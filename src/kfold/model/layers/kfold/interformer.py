"""Modified Pairformer architecture for KFold.

Unlike the standard AlphaFold3 Pairformer, which primarily passes information
from the pairwise representation `z` to the single representation `s` (z -> s),
this architecture enables a fully bidirectional information flow (s <-> z).

NOTE: This bidirectional coupling is designed to capture the dynamic interplay
between evolutionary pre-trained sequence features (single representation)
and residue-residue interaction features (pairwise representation).

Motivation:
This is motivated by the need to effectively integrate evolutionary
information and interaction features, which is already performed in AlphaFold3's
MSAModule: `m -> z` and `z -> m`.

Approach:
This is achieved by integrating a `PairwiseProdDiff` module—inspired by
ESMFold—which updates the pairwise embeddings using both the element-wise
difference and product of the single embeddings.
"""

import math
from functools import partial

import torch
import torch.nn as nn

from kfold.model.layers.alphafold3.transformers import AttentionPairBias
from kfold.model.layers.alphafold3.transition import Transition
from kfold.model.layers.primitives import (
    DropoutColumnwise,
    DropoutRowwise,
    GeoNorm,
    LayerNorm,
    Linear,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from kfold.utils.checkpointing import checkpoint_blocks


class InterformerStack(nn.Module):
    """Interformer stack."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads_attn: int = 16,
        num_heads_tri_attn: int = 4,
        num_blocks: int = 48,
        dropout: float = 0.25,
        use_separate_projections: bool = True,
        skip_tri_attn: bool = False,
        use_qk_norm: bool = False,
        blocks_per_ckpt: int | None = None,
        use_geonorm: bool = True,
        geonorm_decay: str = "harmonic",
        geonorm_clamp: float = math.pi / 4,
    ) -> None:
        """Initialize the Interformer module."""
        super().__init__()
        self.num_blocks: int = num_blocks
        self.blocks_per_ckpt: int | None = blocks_per_ckpt
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                InterformerBlock(
                    channel_s,
                    channel_z,
                    num_heads_attn,
                    num_heads_tri_attn,
                    dropout,
                    use_separate_projections,
                    skip_tri_attn,
                    use_qk_norm,
                    use_geonorm=use_geonorm,
                    geonorm_decay=geonorm_decay,
                    geonorm_clamp=geonorm_clamp,
                )
            )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        intra_mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The sequence embeddings
        z : torch.Tensor
            The pairwise embeddings
        mask : torch.Tensor
            The token mask
        intra_mask : torch.Tensor
            The intra-chain mask
        chunk_size_tri_attn : int | None, optional
            The chunk size for triangle attention, by default None
        use_cuequiv_kernels : bool, optional
            Whether to use CuEQuiv kernels, by default False

        Returns
        -------
        torch.Tensor
            The updated sequence embeddings.
        torch.Tensor
            The updated pairwise embeddings.

        """
        if self.training:
            assert chunk_size_tri_attn is None, (
                "During training, chunk_size_tri_attn must be None."
            )

        pair_mask = mask[..., None] & mask[..., None, :]

        blocks = [
            partial(
                b,
                single_mask=mask,
                intra_mask=intra_mask,
                pair_mask=pair_mask,
                chunk_size_tri_attn=chunk_size_tri_attn,
                use_cuequiv_kernels=use_cuequiv_kernels,
                layer_number=layer_number,
                layer_total=self.num_blocks,
            )
            for layer_number, b in enumerate(self.blocks)
        ]
        blocks_per_ckpt = self.blocks_per_ckpt

        if self.training and torch.is_grad_enabled():
            s, z = checkpoint_blocks(
                blocks,
                (s, z),
                blocks_per_ckpt,
                use_reentrant=False,
            )
        else:
            for block in blocks:
                s, z = block(s, z)

        # Line 10
        return s, z


class PairwiseProdDiff(nn.Module):
    """Convert single embeddings to pairwise embeddings.
    Inspired by ESMFold's implementation.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        assert c_out % 2 == 0, "c_out must be even."

        c_hidden = c_out // 2

        self.layernorm = LayerNorm(c_in)
        self.linear_in = Linear(c_in, c_hidden * 2, init="default")
        self.linear_out = Linear(c_hidden * 2, c_out, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Compute pairwise embeddings from single representations using
        element-wise differences and products.

        Parameters
        ----------
        s : torch.Tensor
            The single representation (*, L, c_in).

        Returns
        -------
        torch.Tensor
            The output tensor (*, L, L, c_out).
        """
        s = self.layernorm(s)
        s_i, s_j = torch.chunk(
            self.linear_in(s), 2, dim=-1
        )  # (*, L, c_hid), (*, L, c_hid)

        s_i = s_i.unsqueeze(-2)  # (*, L, 1, c_hidden)
        s_j = s_j.unsqueeze(-3)  # (*, 1, L, c_hidden)

        # Combine Diff (Asymmetry) and Prod (Correlation)
        # NOTE: summation is derived from production operation with linear bias
        # (W1(s_i) + b1) * (W2(s_j) + b2)
        #   = W1(s_i)W2(s_j) + b1*W2(s_j) + b2*W1(s_i) + b1*b2
        z = torch.cat([s_i - s_j, s_i * s_j], dim=-1)  # (*, L, L, c_hidden * 2)

        z = self.linear_out(z)  # (*, L, L, c_out)
        return z


class InterformerBlock(nn.Module):
    """A single Interformer block."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads_attn: int = 16,
        num_heads_tri_attn: int = 4,
        dropout: float = 0.25,
        use_separate_projections: bool = True,
        skip_tri_attn: bool = False,
        use_qk_norm: bool = False,
        use_geonorm: bool = True,
        geonorm_decay: str = "harmonic",
        geonorm_clamp: float = math.pi / 4,
    ) -> None:
        """Initialize the Interformer module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        num_heads_attn : int, optional
            The number of attention heads, by default 16
        num_heads_tri_attn : int, optional
            The number of triangle attention heads, by default 4
        dropout : float, optional
            The dropout rate, by default 0.25
        use_separate_projections : bool, optional
            Whether to use separate projections for intra- and inter-chain
            residue pairs.
        skip_tri_attn : bool, optional
            Whether to skip triangle attention, by default False
        """
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.num_heads_attn: int = num_heads_attn
        self.num_heads_tri_attn: int = num_heads_tri_attn
        self.skip_tri_attn: bool = skip_tri_attn
        self.dropout: float = dropout

        self.use_geonorm: bool = use_geonorm
        if self.use_geonorm:
            # GeoNorm replaces x <- x + g with a geodesic update on the ℓ2 sphere.
            # We keep the internal modules unchanged and only modify residual updates.
            if use_separate_projections:
                self.geonorm_z_proj_intra = GeoNorm(
                    decay=geonorm_decay, clamp=geonorm_clamp
                )
                self.geonorm_z_proj_inter = GeoNorm(
                    decay=geonorm_decay, clamp=geonorm_clamp
                )
            else:
                self.geonorm_z_proj = GeoNorm(decay=geonorm_decay, clamp=geonorm_clamp)

            self.geonorm_z_tri_mul_out = GeoNorm(
                decay=geonorm_decay, clamp=geonorm_clamp
            )
            self.geonorm_z_tri_mul_in = GeoNorm(
                decay=geonorm_decay, clamp=geonorm_clamp
            )
            if not skip_tri_attn:
                self.geonorm_z_tri_att_start = GeoNorm(
                    decay=geonorm_decay, clamp=geonorm_clamp
                )
                self.geonorm_z_tri_att_end = GeoNorm(
                    decay=geonorm_decay, clamp=geonorm_clamp
                )
            self.geonorm_z_transition = GeoNorm(
                decay=geonorm_decay, clamp=geonorm_clamp
            )

            self.geonorm_s_attention = GeoNorm(
                decay=geonorm_decay, clamp=geonorm_clamp
            )
            self.geonorm_s_transition = GeoNorm(
                decay=geonorm_decay, clamp=geonorm_clamp
            )

        self.use_separate_projections: bool = use_separate_projections
        if self.use_separate_projections:
            self.pairwise_proj_intra = PairwiseProdDiff(channel_s, channel_z)
            self.pairwise_proj_inter = PairwiseProdDiff(channel_s, channel_z)
        else:
            self.pairwise_proj = PairwiseProdDiff(channel_s, channel_z)

        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)

        if not self.skip_tri_attn:
            self.tri_att_start = TriangleAttentionStartingNode(
                channel_z, num_heads_tri_attn, inf=1e9
            )
            self.tri_att_end = TriangleAttentionEndingNode(
                channel_z, num_heads_tri_attn, inf=1e9
            )

        self.attention = AttentionPairBias(
            channel_a=channel_s,
            channel_z=channel_z,
            num_heads=num_heads_attn,
            channel_s=None,
            use_single_cond=False,
            qk_norm=use_qk_norm,
        )

        self.transition_s = Transition(channel_s, expansion_factor=4)
        self.transition_z = Transition(channel_z, expansion_factor=4)

        self.dropout_rowwise = DropoutRowwise(dropout)
        self.dropout_columnwise = DropoutColumnwise(dropout)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        intra_mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_kernels: bool = False,
        layer_number: int = 0,
        layer_total: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass."""

        # Information flow from single (s) to pairwise (z)
        # Separate projections for intra- and inter-chain residue pairs
        if self.use_separate_projections:
            g_intra = self.pairwise_proj_intra(s) * intra_mask[..., None]
            if self.use_geonorm:
                z = self.geonorm_z_proj_intra(z, g_intra, layer_number, layer_total)
            else:
                z = z + g_intra

            g_inter = self.pairwise_proj_inter(s) * (~intra_mask)[..., None]
            if self.use_geonorm:
                z = self.geonorm_z_proj_inter(z, g_inter, layer_number, layer_total)
            else:
                z = z + g_inter
        else:
            g_proj = self.pairwise_proj(s)
            if self.use_geonorm:
                z = self.geonorm_z_proj(z, g_proj, layer_number, layer_total)
            else:
                z = z + g_proj

        # Triangle multiplicative update
        g_tri_mul_out = self.dropout_rowwise(
            self.tri_mul_out(
                z,
                pair_mask,
                use_kernels=use_cuequiv_kernels,
            )
        )
        if self.use_geonorm:
            z = self.geonorm_z_tri_mul_out(z, g_tri_mul_out, layer_number, layer_total)
        else:
            z = z + g_tri_mul_out

        g_tri_mul_in = self.dropout_rowwise(
            self.tri_mul_in(
                z,
                mask=pair_mask,
                use_kernels=use_cuequiv_kernels,
            )
        )
        if self.use_geonorm:
            z = self.geonorm_z_tri_mul_in(z, g_tri_mul_in, layer_number, layer_total)
        else:
            z = z + g_tri_mul_in
        if not self.skip_tri_attn:
            # Triangle attention update
            g_tri_att_start = self.dropout_rowwise(
                self.tri_att_start(
                    z,
                    mask=pair_mask,
                    chunk_size=chunk_size_tri_attn,
                    use_kernels=use_cuequiv_kernels,
                )
            )
            if self.use_geonorm:
                z = self.geonorm_z_tri_att_start(
                    z, g_tri_att_start, layer_number, layer_total
                )
            else:
                z = z + g_tri_att_start

            g_tri_att_end = self.dropout_columnwise(
                self.tri_att_end(
                    z,
                    mask=pair_mask,
                    chunk_size=chunk_size_tri_attn,
                    use_kernels=use_cuequiv_kernels,
                )
            )
            if self.use_geonorm:
                z = self.geonorm_z_tri_att_end(
                    z, g_tri_att_end, layer_number, layer_total
                )
            else:
                z = z + g_tri_att_end

        # Transition for pairwise representation
        g_z = self.transition_z(z)
        if self.use_geonorm:
            z = self.geonorm_z_transition(z, g_z, layer_number, layer_total)
        else:
            z = z + g_z

        # Information flow from pairwise (z) to single (s)
        g_s_attn = self.attention(
            s,  # [B, L, C_s]
            None,
            z,  # [B, L, L, C_z]
            attn_mask=single_mask,  # [B, L]
            use_kernels=use_cuequiv_kernels,
        )
        if self.use_geonorm:
            s = self.geonorm_s_attention(s, g_s_attn, layer_number, layer_total)
        else:
            s = s + g_s_attn

        g_s = self.transition_s(s)
        if self.use_geonorm:
            s = self.geonorm_s_transition(s, g_s, layer_number, layer_total)
        else:
            s = s + g_s

        return s, z
