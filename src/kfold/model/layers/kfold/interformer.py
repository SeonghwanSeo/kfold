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

from functools import partial

import torch
import torch.nn as nn

from kfold.model.layers.alphafold3.transformers import AttentionPairBias
from kfold.model.layers.alphafold3.transition import Transition
from kfold.model.layers.primitives import (
    DropoutColumnwise,
    DropoutRowwise,
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
        skip_tri_attn: bool = False,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        """Initialize the Interformer module."""
        super().__init__()
        self.blocks_per_ckpt: int | None = blocks_per_ckpt
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                InterformerBlock(
                    channel_s,
                    channel_z,
                    num_heads_attn,
                    num_heads_tri_attn,
                    skip_tri_attn,
                    dropout,
                )
            )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
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
                pair_mask=pair_mask,
                chunk_size_tri_attn=chunk_size_tri_attn,
                use_cuequiv_kernels=use_cuequiv_kernels,
            )
            for b in self.blocks
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
        skip_tri_attn: bool = False,
        dropout: float = 0.25,
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
        skip_tri_attn : bool, optional
            Whether to skip triangle attention, by default False
        dropout : float, optional
            The dropout rate, by default 0.25
        """
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.num_heads_attn: int = num_heads_attn
        self.num_heads_tri_attn: int = num_heads_tri_attn
        self.skip_tri_attn: bool = skip_tri_attn
        self.dropout: float = dropout

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
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass."""

        # Information flow from single (s) to pairwise (z)
        z = z + self.pairwise_proj(s)

        # Triangle multiplicative update
        z = z + self.dropout_rowwise(
            self.tri_mul_out(
                z,
                pair_mask,
                use_kernels=use_cuequiv_kernels,
            )
        )
        z = z + self.dropout_rowwise(
            self.tri_mul_in(
                z,
                mask=pair_mask,
                use_kernels=use_cuequiv_kernels,
            )
        )
        if not self.skip_tri_attn:
            # Triangle attention update
            z = z + self.dropout_rowwise(
                self.tri_att_start(
                    z,
                    mask=pair_mask,
                    chunk_size=chunk_size_tri_attn,
                    use_kernels=use_cuequiv_kernels,
                )
            )
            z = z + self.dropout_columnwise(
                self.tri_att_end(
                    z,
                    mask=pair_mask,
                    chunk_size=chunk_size_tri_attn,
                    use_kernels=use_cuequiv_kernels,
                )
            )

        # Transition for pairwise representation
        z = z + self.transition_z(z)

        # Information flow from pairwise (z) to single (s)
        s = s + self.attention(
            s,  # [B, L, C_s]
            None,
            z,  # [B, L, L, C_z]
            attn_mask=single_mask,  # [B, L]
            use_kernels=use_cuequiv_kernels,
        )
        s = s + self.transition_s(s)

        return s, z
