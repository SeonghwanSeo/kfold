"""
Multi-Anchor Cropping Strategy for Apo-to-Holo Structure Modeling.

This module implements an extended cropping strategy designed for scenarios
where Apo structures (e.g., from AF2/ESMFold) are used as templates or inputs
to predict Holo complexes.

**Motivation**
In Apo-to-Holo co-folding, a critical challenge is preventing the model from
learning trivial identity mappings. Standard single-center spatial cropping
(as used in AlphaFold-Multimer/AF3) often yields a dense, locally rigid crop.
In such cases, the model can minimize loss simply by copying the input Apo
structure via residual connections, failing to learn global conformational
changes or inter-domain rearrangements.

The 'Multi-Anchor' strategy mitigates this by distributing the token budget
across multiple spatially distinct regions. This forces the model to reason
about the geometric relationships *between* disconnected or distant regions,
thereby encouraging the learning of global structural transitions.

**Algorithm Description**
1. Contiguous Cropping:
    - Same as AF-M/AF3's contiguous cropping strategy.

2. Spatial Cropping (Multi-Anchor):
    - Selects N anchor tokens (default: 4) and partitions the total token budget.
    - The first anchor is sampled based on chain bias or uniformly.
    - Subsequent anchors are sampled from resolved tokens within a defined
      radius (default: 100 Å) of previous anchors to ensure partial connectivity
      while maximizing coverage.

3. Spatial Interface Cropping (Multi-Anchor):
    - Selects anchors specifically from tokens involved in chain interfaces.
    - Unlike random sampling, this strategy traverses the 'interaction graph'.
      Subsequent anchors are chosen from interfaces connected to already
      selected chains, preserving the biological context of the complex assembly.

**References**
- AlphaFold-Multimer (Evans et al., 2021):
    - Algorithm 1: Contiguous Cropping
    - Algorithm 2: Spatial Cropping logic
- AlphaFold 3 (Abramson et al., 2024):
    - Section 2.7: Cropping strategies (Contiguous, Spatial, Spatial Interface)
"""

from collections import defaultdict

import numpy as np

from kfold.data.tokenized import TokenizedStructure
from kfold.utils.registry import DATA_CROPPER

from . import utils
from .base import BaseCropper


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
        anchor_distribution : str
            Distribution to sample number of anchors from.
            Choices: 'uniform', 'linear', 'squared', 'exponential'.
        min_anchors : int
            Minimum number of anchor tokens.
        max_anchors : int
            Maximum number of anchor tokens.
        max_anchor_distance : float
            Maximum distance between anchor tokens during spatial cropping.
        """

        w_contiguous: float = 0.2
        w_spatial: float = 0.4
        w_spatial_interface: float = 0.4
        anchor_distribution: str = "exponential"
        min_anchors: int = 1
        max_anchors: int = 1
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
        self.anchor_distribution: str = config.anchor_distribution
        self.min_anchors: int = config.min_anchors
        self.max_anchors: int = config.max_anchors
        self.max_anchor_distance: float = config.max_anchor_distance

    def get_token_indices(  # noqa: PLR0915
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        bias_asym_id : int | tuple[int, int] | None
            The chain ID(s) to center the crop on. If None, a random chain or interface
            will be selected.
        rng : np.random.Generator
            The random number generator

        Returns
        -------
        token_indices: np.ndarray
            The selected token indices.
        """
        v = rng.random()
        if v < self.w_contiguous:
            # Contiguous cropping
            crop_indices = self.crop_contiguous(struct, max_tokens, rng=rng)
        elif v < self.w_contiguous + self.w_spatial:
            # Spatial cropping
            crop_indices = self.crop_spatial(struct, max_tokens, bias_asym_id, rng=rng)
        else:  # Spatial interface cropping
            crop_indices = self.crop_spatial_interface(
                struct, max_tokens, bias_asym_id, rng=rng
            )

        # Ensure sorted order and limit to max_tokens
        crop_indices.sort()
        if len(crop_indices) > max_tokens:
            crop_indices = crop_indices[:max_tokens]

        return crop_indices

    def crop_contiguous(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select an anchor token using contiguous cropping.
        See Algorithm 1 in the AlphaFold Multimer paper.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        rng : np.random.Generator
            The random number generator.

        Returns
        -------
        token_index : np.ndarray
            The selected token index.
        """

        # Compute the number of tokens and start indices per chain
        asym_id_to_chain_idx: dict[int, int] = {
            v: i for i, v in enumerate(struct.chain.asym_id.tolist())
        }
        chain_sizes: dict[int, int] = {
            asym_id: int(struct.chain.num_tokens[chain_idx])
            for asym_id, chain_idx in asym_id_to_chain_idx.items()
        }

        # Randomly permute the chain order
        asym_ids = struct.chain.asym_id
        selected_asym_ids = rng.permutation(asym_ids)

        # Line 1
        n_added: int = 0
        # Line 2
        # NOTE: This differs from the original algorithm which uses max_tokens.
        n_remaining: int = sum(chain_sizes[asym_id] for asym_id in selected_asym_ids)

        is_selected = np.zeros(struct.num_tokens, dtype=bool)

        # Line 3-13
        for asym_id in selected_asym_ids:
            if n_added >= max_tokens:
                break

            n_k = chain_sizes[asym_id]
            # Line 4
            n_remaining -= n_k

            # Sample length of crop for current chain
            # Line 5
            max_crop = min(max_tokens - n_added, n_k)
            # Line 6
            min_crop = min(n_k, max(0, max_tokens - n_added - n_remaining))
            # Line 7
            crop_size = int(rng.integers(min_crop, max_crop + 1))
            # Line 8
            n_added += crop_size

            if crop_size == 0:
                continue

            # Line 9
            crop_start = int(rng.integers(0, n_k - crop_size + 1))

            # Line 11
            chain_idx = asym_id_to_chain_idx[asym_id]
            chain_st = int(struct.chain.token_start[chain_idx])
            crop_start += chain_st
            selected_tokens = np.arange(crop_start, crop_start + crop_size)

            # Line 12
            is_selected[selected_tokens] = True

        return np.where(is_selected)[0]

    def crop_spatial(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select anchor tokens using spatial cropping.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        bias_asym_id : int | tuple[int, int] | None
            The chain IDs to bias the anchor selection towards.
        rng : np.random.Generator
            The random number generator.

        Returns
        -------
        token_indices : np.ndarray
            The selected token indices.
        """
        # For spatial cropping, get the token center coordinates
        tokens = struct.token.token_index  # =np.arange(n_tokens)
        center_idx = struct.token.center_index  # (n_tokens, 3)

        holo_coords = struct.atom.coords  # (n_tokens, 24, 3)
        center_coords = holo_coords[tokens, center_idx, :]  # (n_tokens, 3)
        resolved_mask = struct.atom.resolved_mask[tokens, center_idx]  # (n_tokens,)

        if resolved_mask.sum() <= max_tokens:
            # If all resolved tokens fit in the budget, return all
            return np.where(resolved_mask)[0]

        # Sample number of anchors and their budgets
        num_anchors: int = self.sample_num_anchors(rng)
        budgets: list[int] = self.sample_budgets_per_anchor(num_anchors, max_tokens, rng)

        anchor_tokens: list[int] = []
        is_selected = np.zeros(struct.num_tokens, dtype=bool)
        is_remaining = resolved_mask.copy()
        for i in range(num_anchors):
            # Select anchor token
            if i == 0:
                # Pick first anchor randomly or from biased chain(s)
                anchor = utils.pick_token(struct, bias_asym_id, is_remaining, rng)
            else:
                # Pick anchor token not too far from previous anchor
                prev_anchor = anchor_tokens[-1]
                prev_anchor_coords = center_coords[prev_anchor]  # (3,)
                dists = np.linalg.norm(center_coords - prev_anchor_coords, axis=1)
                cutoff_mask = dists < self.max_anchor_distance
                anchor_mask = is_remaining & cutoff_mask
                if not np.any(anchor_mask):
                    # Fallback to allow picking from all remaining tokens
                    anchor_mask = resolved_mask & cutoff_mask
                anchor = utils.pick_token(struct, None, anchor_mask, rng)
            anchor_tokens.append(anchor)

            # Crop spatially around the anchor token
            crop_size = budgets[i]
            neighbor_indices = self.get_closest_tokens(
                struct, anchor, crop_size, center_coords, mask=is_remaining
            )
            is_selected[neighbor_indices] = True
            is_remaining[neighbor_indices] = False

        return np.where(is_selected)[0]

    def crop_spatial_interface(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select anchor tokens using spatial cropping.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        bias_asym_id : int | tuple[int, int] | None
            The chain ID(s) to bias the anchor selection towards.

        Returns
        -------
        token_indices : np.ndarray
            The selected token indices.
        """
        # For spatial cropping, get the token center coordinates
        tokens = struct.token.token_index  # =np.arange(n_tokens)
        center_idx = struct.token.center_index  # (n_tokens, 3)

        holo_coords = struct.atom.coords  # (n_tokens, 24, 3)
        center_coords = holo_coords[tokens, center_idx, :]  # (n_tokens, 3)
        resolved_mask = struct.atom.resolved_mask[tokens, center_idx]  # (n_tokens,)

        if resolved_mask.sum() <= max_tokens:
            # If all resolved tokens fit in the budget, return all
            return np.where(resolved_mask)[0]

        # Get all valid interfaces
        all_interfaces = self.get_valid_interfaces(struct)

        if len(all_interfaces) == 0:
            # No valid interfaces, fall back to regular spatial cropping
            return self.crop_spatial(struct, max_tokens, bias_asym_id, rng=rng)

        # Collect neighboring chains for each chain
        chain_to_neighbors: dict[int, list[int]] = defaultdict(list)
        for interface in set(all_interfaces):
            i1, i2 = interface
            chain_to_neighbors[i1].append(i2)
            chain_to_neighbors[i2].append(i1)

        # Sample number of anchors and their budgets
        num_anchors: int = self.sample_num_anchors(rng)
        budgets: list[int] = self.sample_budgets_per_anchor(num_anchors, max_tokens, rng)

        is_selected = np.zeros(struct.num_tokens, dtype=bool)
        is_remaining = resolved_mask.copy()
        visited_chains: list[int] = []
        for i in range(num_anchors):
            # Select an interface to sample anchor from
            if i == 0:
                if bias_asym_id is None:
                    # Pick a random interface
                    interface_id = utils.random_choice(all_interfaces, rng=rng)
                elif isinstance(bias_asym_id, int):
                    # Find an interface containing the biased chain
                    chain_id = bias_asym_id
                    candidate_interfaces = [
                        iface for iface in all_interfaces if chain_id in iface
                    ]
                    if not candidate_interfaces:
                        # Fallback to random interface
                        candidate_interfaces = all_interfaces
                    interface_id = utils.random_choice(candidate_interfaces, rng=rng)
                else:
                    # Pick the biased interface
                    if bias_asym_id in all_interfaces:
                        interface_id = bias_asym_id
                    else:
                        # Fallback to random interface
                        interface_id = utils.random_choice(all_interfaces, rng=rng)
            else:
                # Pick an interface connected to visited chains
                assert len(visited_chains) > 0, "No visited chains"
                candidates: set[tuple[int, int]] = set()
                for asym_id1 in visited_chains:
                    neighbors = chain_to_neighbors[asym_id1]
                    for asym_id2 in neighbors:
                        interface_id = (min(asym_id1, asym_id2), max(asym_id1, asym_id2))
                        candidates.add(interface_id)
                if len(candidates) == 0:
                    # No connected interfaces, fallback to all interfaces
                    candidates = set(all_interfaces)
                interface_id = utils.random_choice(sorted(candidates), rng=rng)

            # Select anchor token from the interface
            # NOTE: Since re-sample from visited interfaces, we allow picking from
            # all resolved tokens even if already selected.
            anchor = utils.pick_interface_token(struct, interface_id, resolved_mask, rng)

            # Collect spatial neighbors around the anchor token among remaining tokens
            crop_size = budgets[i]
            neighbor_indices = self.get_closest_tokens(
                struct, anchor, crop_size, center_coords, mask=is_remaining
            )
            is_selected[neighbor_indices] = True
            is_remaining[neighbor_indices] = False

            visited_chains = np.unique(struct.token.asym_id[is_selected]).tolist()

        return np.where(is_selected)[0]

    def get_closest_tokens(
        self,
        struct: TokenizedStructure,
        anchor_token: int,
        crop_size: int,
        center_coords: np.ndarray | None = None,
        mask: np.ndarray | None = None,
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
        mask : np.ndarray | None, optional
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
        if mask is None:
            mask = struct.atom.resolved_mask[tokens, center_idx]  # (num_tokens,)

        if mask.sum() <= crop_size:
            # If all resolved tokens fit in the budget, return all
            return np.where(mask)[0]

        # Compute distances to all tokens
        anchor_coord = center_coords[anchor_token]  # (3,)
        dists = np.linalg.norm(center_coords - anchor_coord, axis=1)  # (num_tokens,)
        dists[~mask] = np.inf
        # Get tokens within budget (this includes the anchor token itself)
        neighbor_indices = np.argpartition(dists, crop_size - 1)[:crop_size]
        neighbor_indices.sort()
        return neighbor_indices

    # === Helper functions === #
    def sample_num_anchors(self, rng: np.random.Generator) -> int:
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
                w = np.linspace(1.0, 0.0, n_candidates + 1)[:-1] ** 2
            case "exponential":
                w = np.exp(-candidates)
            case _:
                raise ValueError(f"Unknown anchor distribution: {distribution}")
        w /= w.sum()
        return int(rng.choice(candidates, p=w))

    def sample_budgets_per_anchor(
        self,
        num_anchors: int,
        total_budget: int,
        rng: np.random.Generator,
        min_per_anchor: int | None = None,
    ) -> list[int]:
        """Sample the token budgets for each anchor token.

        Parameters
        ----------
        num_anchors : int
            The number of anchor tokens.
        total_budget : int
            The total token budget.
        rng : np.random.Generator
            The random number generator.
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

        budgets: list[int] = [min_per_anchor] * num_anchors
        remaining_budget = total_budget - sum(budgets)

        splits = rng.integers(0, remaining_budget + 1, size=num_anchors - 1)
        splits = np.concatenate(([0], np.sort(splits), [remaining_budget]))
        allocation = splits[1:] - splits[:-1]
        for i in range(num_anchors):
            budgets[i] += allocation[i].item()

        # sort budgets in descending order
        budgets.sort(reverse=True)

        return budgets

    @staticmethod
    def get_valid_interfaces(struct: TokenizedStructure) -> list[tuple[int, int]]:
        """Get all valid interfaces in the structure.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.

        Returns
        -------
        interface_ids : list[tuple[int, int]]
            The valid interfaces in the structure.
        """
        metadata = struct.metadata
        assert metadata is not None, "Structure metadata is required"
        all_chains: set[int] = set(struct.chain.asym_id.tolist())
        all_interfaces: list[tuple[int, int]] = [
            tuple(interface.asym_ids)
            for interface in metadata.interfaces
            if interface.is_valid
        ]
        all_interfaces = [v for v in all_interfaces if set(v).issubset(all_chains)]
        return sorted(set(all_interfaces))
