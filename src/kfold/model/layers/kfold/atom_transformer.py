from collections.abc import Callable

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.atom_transformer import AtomEmbedder
from kfold.model.layers.alphafold3.utils import broadcast_tokens_to_atoms
from kfold.model.layers.primitives import LinearNoBias


class AtomEmbedderWithApo(AtomEmbedder):
    """Input embedding module with apo structure embedding for atom attention encoder."""

    def __init__(
        self,
        channel_s: int,
        channel_z: int | None,
        channel_atom: int,
        channel_atompair: int,
        use_structure: bool = False,
    ):
        """Initialize the atom attention encoder.

        Parameters
        ----------
        channel_s : int
            The single representation dimension.
        channel_z : int | None
            The pair representation dimension.
        channel_atom : int
            The atom single representation dimension.
        channel_atompair : int
            The atom pair representation dimension.
        use_apo : bool, optional
            Whether to use apo structure embedding, by default True.
        use_structure : bool, optional
            Whether to use structure information, by default True.

        """
        super().__init__(
            channel_s, channel_z, channel_atom, channel_atompair, use_structure
        )
        # Apo position embeddings
        self.embed_apo_offset = LinearNoBias(
            3, channel_atompair, init="default", precision=32
        )
        self.embed_apo_inv_dist = LinearNoBias(1, channel_atompair, init="default")
        self.embed_apo_mask = LinearNoBias(1, channel_atompair, init="default")

    def embed_atom_pairs(self, f_input: FoldingInput, to_qk: Callable) -> torch.Tensor:
        """Get atom pair representation from reference molecule conformer and
        apo conformer.

        Parameters
        ----------
        f_input : FoldingInputk
            The folding input.

        Returns
        -------
        p : torch.Tensor
            The atom pair representation, shape [B, W, Lq, Lk, c_atompair]
        """
        p = super().embed_atom_pairs(f_input, to_qk)  # [B, W, Lq, Lk, c_atompair]
        p = p + self.apo_embedding(f_input, to_qk).to(p.dtype)
        return p

    def apo_embedding(self, f_input: FoldingInput, to_qk: Callable) -> torch.Tensor:
        """Get apo conformer embedding

        Parameters
        ----------
        f_input : FoldingInputk
            The folding input.

        Returns
        -------
        p_apo: torch.Tensor
            The apo conformer embedding, shape [B, W, Lq, Lk, c_atompair]
        """
        # Mask unresolved apo atoms
        mask_q, mask_k = to_qk(f_input.atom.apo_mask, dim=-1)
        v = mask_q[..., :, None] & mask_k[..., None, :]

        # Mask with chain identity (Apo structure is defined per chain)
        asym_id = broadcast_tokens_to_atoms(
            f_input.token.asym_id.unsqueeze(-1), f_input.atom.token_index
        ).squeeze(-1)  # [B, La]
        asym_id_q, asym_id_k = to_qk(asym_id, dim=-1)
        v &= asym_id_q[..., :, None] == asym_id_k[..., None, :]  # [B, W, Lq, Lk]

        # Mask with distance cutoff in sequence (10 neighbor residues)
        # NOTE: (SeonghwanSeo) In spatial cropping, the residue indices may not be
        # continuous. To capture local geometry, we only consider atoms from residues
        # that are within 5 residues in sequence.
        residue_idx = broadcast_tokens_to_atoms(
            f_input.token.residue_index.unsqueeze(-1), f_input.atom.token_index
        ).squeeze(-1)  # [B, La]
        residx_q, residx_k = to_qk(residue_idx, dim=-1)
        v &= abs(residx_q[..., :, None] - residx_k[..., None, :]) <= 5

        # Final apo mask
        v = v.float().unsqueeze(-1)  # [B, W, Lq, Lk, 1]

        # Shape: [B, La, 3] -> [B, W, Lq, 3], [B, W, Lk, 3]
        apo_pos_q, apo_pos_k = to_qk(f_input.atom.apo_coords, dim=-2)
        with torch.autocast(v.device.type, enabled=False):
            # NOTE: (SeonghwanSeo) Since apo structure is much larger than ref_pos,
            # d_inv is adopted instead of d_inv_sq for better representation.
            # Shape: [B, W, Lq, Lk, 3], [B, W, Lq, Lk, 1]
            apo_d_offset = apo_pos_q[..., :, None, :] - apo_pos_k[..., None, :, :]
            apo_d_inv = 1.0 / (1.0 + apo_d_offset.norm(dim=-1, keepdim=True))
            apo_d_offset = apo_d_offset * apo_d_inv.sqrt()  # scale offsets

        # Shape: [B, W, Lq, Lk, c_atompair]
        p = self.embed_apo_offset(apo_d_offset)
        p = p + self.embed_apo_inv_dist(apo_d_inv)
        p = p + self.embed_apo_mask(v)
        p = p * v
        return p
