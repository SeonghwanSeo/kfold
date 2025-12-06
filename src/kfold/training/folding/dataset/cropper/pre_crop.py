import numpy as np

from kfold.data.structure import TokenizedStructure
from kfold.utils.registry import DATA_CROPPER

from . import utils
from .base import BaseCropper


@DATA_CROPPER.register()
class PreCropper(BaseCropper):
    """Pre-Cropper that selects up to `max_chains` chains.
    See AlphaFold 3 SI Section 2.5.4.

    NOTE: The output of this cropper is treated as the "original" structure
    rather than a training crop. It effectively samples neighboring chains to
    reduce the complex to a manageable bioassembly size before further processing.

    NOTE: Unlike the original AF3 description, this cropper is polymer-centric.
    Small molecules (<6 atoms) do not count toward the chain limit. For example,
    20 polymer chains + 3 small molecules/metals are counted as 20 chains, not 23.
    """

    class Config(BaseCropper.Config):
        """Configuration for the MultiAnchorCropper.

        Parameters
        ----------
        max_chains : int
            Maximum number of chains to consider for contiguous cropping.
        min_chain_atom_count : int
            The minimum number of atoms in a chain to be considered
            for max chain counting. If 0, all chains are counted.
        """

        max_chains: int = 20
        min_chain_atom_count: int = 6

    def __init__(self, config: Config):
        self.max_chains: int = config.max_chains
        self.min_chain_atom_count: int = config.min_chain_atom_count

    def crop(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> TokenizedStructure:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
            NOTE: NOT USED in this cropper.
        asym_ids : tuple[int, ...] | None, optional
            The chain IDs to center the crop on. If None, a random chain
            NOTE: NOT USED in this cropper.

        Returns
        -------
        cropped_struct: TokenizedStructure
            The sub-complex structure with limited number of chains.
        """
        rng = rng or np.random.default_rng()

        if struct.num_chains <= self.max_chains:
            # No cropping needed
            return struct

        # Sample the chains to include in the crop
        sampled_asym_ids = self.sample_chains(struct, bias_asym_id, rng=rng)
        token_mask = np.isin(struct.token.asym_id, sampled_asym_ids)
        selected_token_indices = struct.token.token_index[token_mask]

        # Extract the sub-complex structure with the selected chains
        partial_struct = struct.crop(selected_token_indices)

        # Re-assign token index to be consecutive
        partial_struct = struct.reassign_token_indices()

        return partial_struct

    def get_token_indices(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise ValueError("PreCropper does not support get_token_indices")

    def sample_chains(
        self,
        struct: TokenizedStructure,
        bias_asym_id: int | tuple[int, int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()

        all_tokens = struct.token.token_index
        center_idx = struct.token.center_index

        # Check specific atom resolvability (center atoms only)
        # Assuming resolved_mask is shape (N_tokens, N_atoms)
        resolved_mask = struct.atom.resolved_mask[all_tokens, center_idx]

        # 1. Select Anchor Token
        valid_interfaces = self.get_valid_interfaces(struct)

        if not valid_interfaces:
            # Fallback if no interfaces found
            anchor = utils.pick_token(struct, bias_asym_id, resolved_mask, rng=rng)
        else:
            if bias_asym_id is None:
                interface_id = utils.random_choice(valid_interfaces, rng=rng)
            elif isinstance(bias_asym_id, int):
                candidate_interfaces = [
                    iface for iface in valid_interfaces if bias_asym_id in iface
                ]
                # Fallback if bias chain is not in any valid interface
                if not candidate_interfaces:
                    candidate_interfaces = valid_interfaces
                interface_id = utils.random_choice(candidate_interfaces, rng=rng)
            else:
                # Handle tuple case (specific interface request)
                interface_id = bias_asym_id

            anchor = utils.pick_interface_token(
                struct, interface_id, resolved_mask, rng=rng
            )

        # 2. Compute distances from anchor to all other tokens
        # SI: "based on minimum distance between any tokens centre atom"
        holo_coords = struct.atom.coords[:, :, 0, :]  # (Ntoken, 24, 3)

        # Anchor token center coordinate
        anchor_coord = holo_coords[anchor, center_idx[anchor], :]  # (3,)

        # All token centers
        all_token_centers = holo_coords[all_tokens, center_idx, :]  # (Ntoken, 3)

        dists = np.linalg.norm(all_token_centers - anchor_coord, axis=-1)
        dists[~resolved_mask] = np.inf

        # 3. Select chains based on nearest distances
        neighbors = np.argsort(dists)
        selected_asym_ids: list[int] = []
        chain_counts: int = 0

        all_asym_ids = struct.token.asym_id
        for idx in neighbors:
            if np.isinf(dists[idx]):
                break
            if chain_counts >= self.max_chains:
                break
            asym_id = all_asym_ids[idx]
            if asym_id in selected_asym_ids:
                continue
            selected_asym_ids.append(asym_id)
            chain_idx = np.where(struct.chain.asym_id == asym_id)[0][0]
            if self.min_chain_atom_count <= struct.chain.num_atoms[chain_idx]:
                # Only count chains with sufficient atoms
                chain_counts += 1

        return np.array(selected_asym_ids)

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
        record = struct.metadata
        assert record is not None, "Structure metadata is required"
        all_chains: set[int] = set(struct.chain.asym_id.tolist())
        all_interfaces: list[tuple[int, int]] = [
            interface.asym_ids for interface in record.interfaces if interface.valid
        ]
        all_interfaces = [v for v in all_interfaces if set(v).issubset(all_chains)]
        return sorted(set(all_interfaces))
