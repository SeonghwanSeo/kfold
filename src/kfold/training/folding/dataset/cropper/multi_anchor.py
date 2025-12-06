import random
from collections import defaultdict

import numpy as np

from kfold.data.structure import TokenizedStructure
from kfold.utils.registry import DATA_CROPPER

from .base import BaseCropper
from .utils import pick_token


@DATA_CROPPER.register()
class MultiAnchorCropper(BaseCropper):
    """Cropper that selects tokens based on multiple anchor tokens.
    When the number of anchor tokens is 1, this is equivalent to AF3's cropping strategy.
    """

    class Config(BaseCropper.Config):
        """Configuration for the MultiAnchorCropper.

        Parameters
        ----------
        w_contiguous : float
            Weight for contiguous cropping.
        w_spatial : float
            Weight for spatial cropping.
        w_spatial_interface : float
            Weight for spatial interface cropping.
        max_chains : int
            Maximum number of chains to consider for contiguous cropping.
        anchor_distribution : str
            Distribution to sample number of anchors from.
            Choices: 'uniform', 'linear', 'squared', 'exponential'.
        max_anchors : int
            Maximum number of anchor tokens.
        max_anchor_distance : float
            Maximum distance between anchor tokens during spatial cropping.
        """

        w_contiguous: float = 0.3
        w_spatial: float = 0.2
        w_spatial_interface: float = 0.5
        max_chains: int = 20
        anchor_distribution: str = "exponential"
        max_anchors: int = 4
        max_anchor_distance: float = 100.0

    def __init__(self, config: Config):
        self.config = config

        assert (
            self.config.w_contiguous
            + self.config.w_spatial
            + self.config.w_spatial_interface
            == 1.0
        ), "Weights must sum to 1."

        self.w_contiguous: float = config.w_contiguous
        self.w_spatial: float = config.w_spatial
        self.w_spatial_interface: float = config.w_spatial_interface
        self.max_chains: int = config.max_chains
        self.anchor_distribution: str = config.anchor_distribution
        self.min_anchors: int = 1
        self.max_anchors: int = config.max_anchors
        self.max_anchor_distance: float = config.max_anchor_distance

    def get_token_indices(  # noqa: PLR0915
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        asym_ids: tuple[int, ...] | None,
    ) -> np.ndarray:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        asym_ids : tuple[int, ...] | None
            The chain IDs to center the crop on. If None, a random chain

        Returns
        -------
        token_indices: np.ndarray
            The selected token indices.
        """
        v = np.random.rand()
        if v < self.w_contiguous:
            # Contiguous cropping
            crop_indices = self.crop_contiguous(struct, max_tokens)
        elif v < self.w_contiguous + self.w_spatial:
            # Spatial cropping
            crop_indices = self.crop_spatial(struct, max_tokens, asym_ids)
        else:  # Spatial interface cropping
            crop_indices = self.crop_spatial_interface(struct, max_tokens, asym_ids)

        # Ensure sorted order and limit to max_tokens
        crop_indices.sort()
        if len(crop_indices) > max_tokens:
            crop_indices = crop_indices[:max_tokens]

        return crop_indices

    def crop_contiguous(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
    ) -> np.ndarray:
        """Select an anchor token using contiguous cropping.
        See Algorithm 1 in the AlphaFold Multimer paper.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.

        Returns
        -------
        token_index : np.ndarray
            The selected token index.
        """

        # Compute the number of tokens and start indices per chain
        chain_sizes: dict[int, int] = {
            int(struct.chain.asym_id[i]): int(struct.chain.num_tokens[i])
            for i in range(struct.num_chains)
        }

        # Sample chains up to max_chains
        asym_ids = struct.chain.asym_id
        selected_asym_ids = np.random.permutation(asym_ids)[: self.max_chains]

        is_selected = np.zeros(struct.num_tokens, dtype=bool)

        # Line 1
        n_added: int = 0
        # Line 2
        # NOTE: This differs from the original algorithm which uses max_tokens.
        n_remaining: int = sum(chain_sizes[asym_id] for asym_id in selected_asym_ids)

        # Line 3-13
        for asym_id in selected_asym_ids:
            n_k = chain_sizes[asym_id]
            # Line 4
            n_remaining -= n_k

            # Sample length of crop for current chain
            # Line 5
            max_crop = min(max_tokens - n_added, n_k)
            # Line 6
            min_crop = min(n_k, max(0, max_tokens - n_added - n_remaining))
            # Line 7
            crop_size = np.random.randint(min_crop, max_crop + 1)
            # Line 8
            n_added += crop_size

            # Line 9-12
            if crop_size > 0:
                selected_tokens = self.do_crop_chain_contiguously(
                    struct, asym_id, crop_size
                )
                is_selected[selected_tokens] = True

            if n_added >= max_tokens:
                break

        return np.where(is_selected)[0]

    def crop_spatial(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_ids: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Select anchor tokens using spatial cropping.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        asym_ids : tuple[int, ...] | None
            The chain IDs to bias the anchor selection towards.

        Returns
        -------
        token_index : np.ndarray
            The selected token index.
        """
        # For spatial cropping, get the token center coordinates
        tokens = struct.token.token_index  # =np.arange(n_tokens)
        center_idx = struct.token.center_index  # (n_tokens, 3)

        holo_coords = struct.atom.coords[..., 0, :]  # (n_tokens, 24, 3)
        center_coords = holo_coords[tokens, center_idx, :]  # (n_tokens, 3)
        resolved_mask = struct.atom.resolved_mask[tokens, center_idx]  # (n_tokens,)

        if resolved_mask.sum() <= max_tokens:
            # If all resolved tokens fit in the budget, return all
            return np.where(resolved_mask)[0]

        # Sample number of anchors and their budgets
        num_anchors: int = self.sample_num_anchors()
        budgets: list[int] = self.sample_budgets_per_anchor(num_anchors, max_tokens)

        anchor_tokens: list[int] = []
        is_selected = np.zeros(struct.num_tokens, dtype=bool)
        is_remaining = resolved_mask.copy()
        for i in range(num_anchors):
            # === Select anchor token === #
            asym_id: int | None = None  # No bias by default
            anchor_mask = resolved_mask
            if i == 0 and bias_asym_ids is not None:
                # If bias is given, sample the first anchor from the biased chains
                asym_id = random.choice(bias_asym_ids)
            elif len(anchor_tokens) > 0:
                # Pick anchor token not to far from previous anchor
                prev_anchor = anchor_tokens[-1]
                prev_anchor_coords = center_coords[prev_anchor]  # (3,)
                dists = np.linalg.norm(center_coords - prev_anchor_coords, axis=1)
                # Anchor already selected is allowed
                anchor_mask = anchor_mask & (dists < self.max_anchor_distance)

            anchor = pick_token(struct, asym_id=asym_id, mask=anchor_mask)

            # === Crop spatially around the anchor token === #
            crop_size = budgets[i]
            neighbor_indices = self.do_crop_complex_spatially(
                struct, anchor, crop_size, center_coords, resolved_mask=is_remaining
            )
            is_selected[neighbor_indices] = True
            is_remaining[neighbor_indices] = False
            anchor_tokens.append(anchor)

        return np.where(is_selected)[0]

    def crop_spatial_interface(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_ids: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Select anchor tokens using spatial cropping.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        bias_asym_ids : tuple[int, ...] | None
            The chain IDs to bias the anchor selection towards.

        Returns
        -------
        token_index : np.ndarray
            The selected token index.
        """
        # For spatial cropping, get the token center coordinates
        tokens = struct.token.token_index  # =np.arange(n_tokens)
        center_idx = struct.token.center_index  # (n_tokens, 3)

        holo_coords = struct.atom.coords[..., 0, :]  # (n_tokens, 24, 3)
        center_coords = holo_coords[tokens, center_idx, :]  # (n_tokens, 3)
        resolved_mask = struct.atom.resolved_mask[tokens, center_idx]  # (n_tokens,)

        if resolved_mask.sum() <= max_tokens:
            # If all resolved tokens fit in the budget, return all
            return np.where(resolved_mask)[0]

        # Get all valid interfaces
        record = struct.metadata
        assert record is not None, "Metadata is required for interface cropping."

        all_chains: set[int] = set(struct.chain.asym_id.tolist())
        all_interfaces: list[tuple[int, int]] = [
            interface.asym_ids for interface in record.interfaces if interface.valid
        ]
        all_interfaces = [v for v in all_interfaces if set(v).issubset(all_chains)]
        if bias_asym_ids is not None and len(bias_asym_ids) == 2:
            # Ensure the biased interface is included
            all_interfaces.append(bias_asym_ids)
        all_interfaces = sorted(set(all_interfaces))
        num_interfaces = len(all_interfaces)

        if num_interfaces == 0:
            # No valid interfaces, fall back to regular spatial cropping
            return self.crop_spatial(struct, max_tokens, bias_asym_ids)

        # Collect neighboring chains for each chain
        chain_to_neighbors: dict[int, list[int]] = defaultdict(list)
        for interface in set(all_interfaces):
            i1, i2 = interface
            chain_to_neighbors[i1].append(i2)
            chain_to_neighbors[i2].append(i1)

        # Sample number of anchors and their budgets
        num_anchors: int = self.sample_num_anchors()
        budgets: list[int] = self.sample_budgets_per_anchor(num_anchors, max_tokens)

        is_selected = np.zeros(struct.num_tokens, dtype=bool)
        is_remaining = resolved_mask.copy()
        visited_chains: set[int] = set()
        for i in range(num_anchors):
            # === Select interface to sample from === #
            if i == 0 and bias_asym_ids is not None:
                # Pick first interface randomly or from biased chains
                if len(bias_asym_ids) == 1:
                    asym_id = bias_asym_ids[0]
                    # Find an interface containing the biased chain
                    candidate_interfaces = [
                        interface for interface in all_interfaces if asym_id in interface
                    ]
                    interface = random.choice(candidate_interfaces)
                else:
                    assert len(bias_asym_ids) == 2
                    interface = bias_asym_ids
            elif len(visited_chains) > 0:
                # Pick an interface connected to visited chains
                i1 = random.choice(list(visited_chains))
                i2 = random.choice(chain_to_neighbors[i1])
                interface = (min(i1, i2), max(i1, i2))
            else:
                # Pick a random interface
                interface = random.choice(all_interfaces)

            # === Select anchor token in the interface === #
            crop_size = budgets[i]
            # Anchor already selected is allowed
            anchor = pick_token(struct, asym_id=interface, mask=resolved_mask)

            # Crop spatially around the anchor token
            crop_size = budgets[i]
            neighbor_indices = self.do_crop_complex_spatially(
                struct, anchor, crop_size, center_coords, resolved_mask=is_remaining
            )
            is_selected[neighbor_indices] = True
            is_remaining[neighbor_indices] = False
            visited_chains.update(interface)

        return np.where(is_selected)[0]

    def do_crop_chain_contiguously(
        self, struct: TokenizedStructure, asym_id: int, crop_size: int
    ) -> np.ndarray:
        """Crop a contiguous segment from a specific chain."""
        chain_i = np.where(struct.chain.asym_id == asym_id)[0]
        assert chain_i.size == 1, f"Chain {asym_id} not found."
        chain_i = chain_i[0]

        chain_size = int(struct.chain.num_tokens[chain_i])
        crop_start = np.random.randint(0, chain_size - crop_size + 1, 1).item()

        chain_start = int(struct.chain.token_starts[chain_i])
        global_start = chain_start + crop_start
        return np.arange(global_start, global_start + crop_size)

    def do_crop_complex_spatially(
        self,
        struct: TokenizedStructure,
        anchor_token: int,
        crop_size: int,
        center_coords: np.ndarray | None = None,
        resolved_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Crop tokens spatially around an anchor token.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        anchor_token : int
            The anchor token index.
        crop_size : int
            The number of tokens to crop.
        center_coords : np.ndarray | None, optional
            Precomputed center coordinates of tokens.
        resolved_mask : np.ndarray | None, optional
            Precomputed resolved mask of tokens.

        Returns
        -------
        token_indices : np.ndarray
            The selected token indices.
        """
        tokens = struct.token.token_index  # =np.arange(num_tokens)
        center_idx = struct.token.center_index  # (num_tokens, 3)
        if center_coords is None:
            center_coords = struct.atom.coords[tokens, center_idx]  # (num_tokens, 3)
        if resolved_mask is None:
            resolved_mask = struct.atom.resolved_mask[tokens, center_idx]  # (num_tokens,)

        crop_size = min(crop_size, resolved_mask.sum())

        # Compute distances to all tokens
        anchor_coord = center_coords[anchor_token]  # (3,)
        dists = np.linalg.norm(center_coords - anchor_coord, axis=1)  # (num_tokens,)
        dists[~resolved_mask] = np.inf
        # Get tokens within budget (this includes the anchor token itself)
        neighbor_indices = np.argsort(dists)[:crop_size]
        return neighbor_indices

    # === Helper functions === #
    def sample_num_anchors(self) -> int:
        """Sample the number of anchor tokens."""
        distribution = self.anchor_distribution
        min_anchors = self.min_anchors
        max_anchors = self.max_anchors
        candidates = np.arange(min_anchors, max_anchors + 1)
        n_candidates = len(candidates)
        match distribution:
            case "uniform":
                w = np.ones(n_candidates)
            case "linear":
                w = np.linspace(1.0, 0.0, n_candidates + 1)[:-1]
            case "squared":
                w = np.linspace(1.0, 0.0, n_candidates + 1)[:-1]
                w = w**2
            case "exponential":
                w = np.exp(-candidates)
            case _:
                raise ValueError(f"Unknown anchor distribution: {distribution}")
        w /= w.sum()
        return np.random.choice(candidates, p=w)

    def sample_budgets_per_anchor(
        self,
        num_anchors: int,
        total_budget: int,
        min_per_anchor: int | None = None,
    ) -> list[int]:
        """Sample the token budgets for each anchor token.

        Parameters
        ----------
        num_anchors : int
            The number of anchor tokens.
        total_budget : int
            The total token budget.
        min_per_anchor : int (optional)
            The minimum number of tokens per anchor.
            Default: set to total_budget // num_anchors // 4

        Returns
        -------
        budgets : list[int]
            The token budgets for each anchor token.
            The budgets are sorted in descending order.
        """
        if num_anchors == 1:
            return [total_budget]

        if min_per_anchor is None:
            # Set default minimum per anchor (25% of average)
            min_per_anchor = total_budget // num_anchors // 4

        if min_per_anchor * num_anchors >= total_budget:
            raise ValueError(
                "Minimum per anchor too high for total budget and number of anchors."
            )

        budgets = [min_per_anchor] * num_anchors
        remaining_budget = total_budget - sum(budgets)

        splits = np.random.randint(0, remaining_budget + 1, size=num_anchors - 1)
        splits = np.concatenate(([0], np.sort(splits), [remaining_budget]))
        allocation = splits[1:] - splits[:-1]
        for i in range(num_anchors):
            budgets[i] += allocation[i].item()

        # sort budgets in descending order
        budgets.sort(reverse=True)

        return budgets
