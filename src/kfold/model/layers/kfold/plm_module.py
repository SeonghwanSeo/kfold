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
    LinearNoBias,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from kfold.utils.checkpointing import checkpoint_blocks


class PairwiseProdDiff(nn.Module):
    """Convert single embeddings to pairwise embeddings.
    Inspired by ESMFold's implementation.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        assert c_out % 2 == 0, "c_out must be even."
        c_hidden = c_out // 2
        self.layernorm = LayerNorm(c_in, create_offset=False)
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
        s = self.layernorm(s)  # (*, L, c_in)
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


class PLMModule(nn.Module):
    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        channel_plm_input: int = 2560,
        channel_plm: int = 512,
        num_heads_attn: int = 16,
        num_heads_tri_attn: int = 4,
        num_blocks: int = 4,
        dropout_plm: float = 0.15,
        dropout_z: float = 0.25,
        use_separate_projections: bool = True,
        use_qk_norm: bool = False,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.channel_plm_input: int = channel_plm_input
        self.channel_plm: int = channel_plm
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

        self.proj_s_plm = nn.Sequential(
            LayerNorm(channel_plm_input, create_offset=False),
            LinearNoBias(channel_plm_input, channel_plm, init="default"),
        )
        self.proj_s_input = LinearNoBias(channel_s, channel_plm, init="default")

        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                PLMBlock(
                    channel_plm=channel_plm,
                    channel_z=channel_z,
                    num_heads_attn=num_heads_attn,
                    num_heads_tri_attn=num_heads_tri_attn,
                    dropout_plm=dropout_plm,
                    dropout_z=dropout_z,
                    use_separate_projections=use_separate_projections,
                    use_qk_norm=use_qk_norm,
                    is_last_block=(i == num_blocks - 1),
                )
            )

    def forward(
        self,
        z: torch.Tensor,
        s_input: torch.Tensor,
        s_plm: torch.Tensor,
        asym_id: torch.Tensor,
        mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_kernels: bool = False,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pair representations
        s_input : torch.Tensor
            The input single representations
        s_plm : torch.Tensor
            The sequence embeddings
        asym_id : torch.Tensor
            The asymmetry IDs of shape (B, L)
        mask : torch.Tensor
            The token mask of shape (B, L)
        chunk_size_tri_attn : int | None, optional
            The chunk size for triangle attention, by default None.
        use_cuequiv_kernels : bool, optional
            Whether to use cuEQUIV kernels, by default False.

        Returns
        -------
        torch.Tensor
            The updated pair representations
        """
        # Create masks
        pair_mask = mask[..., None] & mask[..., None, :]
        is_same_chain = asym_id[..., None] == asym_id[..., None, :]
        intra_mask = pair_mask & is_same_chain
        inter_mask = pair_mask & (~is_same_chain)

        # Initial linear projection
        s_plm = self.proj_s_plm(s_plm)
        s_plm = s_plm + self.proj_s_input(s_input)

        # PLM Blocks
        blocks = [
            partial(
                b,
                mask=mask,
                pair_mask=pair_mask,
                intra_mask=intra_mask,
                inter_mask=inter_mask,
                use_cuequiv_kernels=use_cuequiv_kernels,
                chunk_size_tri_attn=chunk_size_tri_attn,
            )
            for b in self.blocks
        ]
        if self.training and torch.is_grad_enabled():
            s_plm, z = checkpoint_blocks(
                blocks,
                (s_plm, z),
                self.blocks_per_ckpt,
                use_reentrant=False,
            )
        else:
            for b in blocks:
                s_plm, z = b(s_plm, z)

        return z


class PLMBlock(nn.Module):
    def __init__(
        self,
        channel_plm: int = 512,
        channel_z: int = 128,
        num_heads_attn: int = 16,
        num_heads_tri_attn: int = 4,
        dropout_plm: float = 0.15,
        dropout_z: float = 0.25,
        use_separate_projections: bool = True,
        use_qk_norm: bool = False,
        is_last_block: bool = False,
    ) -> None:
        super().__init__()
        self.channel_plm: int = channel_plm
        self.channel_z: int = channel_z
        self.use_separate_projections: bool = use_separate_projections

        if self.use_separate_projections:
            self.pairwise_proj_intra = PairwiseProdDiff(channel_plm, channel_z)
            self.pairwise_proj_inter = PairwiseProdDiff(channel_plm, channel_z)
        else:
            self.pairwise_proj = PairwiseProdDiff(channel_plm, channel_z)

        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            channel_z, num_heads_tri_attn, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            channel_z, num_heads_tri_attn, inf=1e9
        )

        self.transition_z = Transition(channel_z, expansion_factor=4)

        self.dropout_plm = nn.Dropout(dropout_plm)
        self.dropout_rowwise_z = DropoutRowwise(dropout_z)
        self.dropout_columnwise_z = DropoutColumnwise(dropout_z)

        self.is_last_block: bool = is_last_block
        if not self.is_last_block:
            self.attention = AttentionPairBias(
                channel_a=channel_plm,
                channel_z=channel_z,
                channel_s=None,
                num_heads=num_heads_attn,
                use_single_cond=False,
                qk_norm=use_qk_norm,
            )
            self.transition_plm = Transition(channel_plm, expansion_factor=4)

    def forward(
        self,
        s_plm: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        intra_mask: torch.Tensor,
        inter_mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pair representations
        s_plm : torch.Tensor
            The sequence embeddings
        mask : torch.Tensor
            The token mask of shape (B, L)
        pair_mask : torch.Tensor
            The pair mask of shape (B, L, L)
        intra_mask : torch.Tensor
            The intra-chain mask of shape (B, L, L)
        inter_mask : torch.Tensor
            The inter-chain mask of shape (B, L, L)

        Returns
        -------
        s_plm : torch.Tensor
            The updated sequence embeddings
        z : torch.Tensor
            The updated pair representations
        """
        # Step 1: single to pair
        if self.use_separate_projections:
            z = z + self.pairwise_proj_intra(s_plm) * intra_mask[..., None]
            z = z + self.pairwise_proj_inter(s_plm) * inter_mask[..., None]
        else:
            z = z + self.pairwise_proj(s_plm) * pair_mask[..., None]

        # Step 2: pair to single
        if not self.is_last_block:
            s_plm = s_plm + self.dropout_plm(
                self.attention(
                    s_plm,  # [*, L, C_plm]
                    None,
                    z,  # [*, L, L, C_z]
                    attn_mask=mask,  # [*, L]
                    use_kernels=False,
                )
            )
            s_plm = s_plm + self.transition_plm(s_plm)  # [*, L, C_plm]

        # Step 3: pair to pair
        z = z + self.dropout_rowwise_z(
            self.tri_mul_out(
                z,
                pair_mask,
                use_kernels=use_cuequiv_kernels,
            )
        )
        z = z + self.dropout_rowwise_z(
            self.tri_mul_in(
                z,
                mask=pair_mask,
                use_kernels=use_cuequiv_kernels,
            )
        )
        z = z + self.dropout_rowwise_z(
            self.tri_att_start(
                z,
                mask=pair_mask,
                chunk_size=chunk_size_tri_attn,
                use_kernels=use_cuequiv_kernels,
            )
        )
        z = z + self.dropout_columnwise_z(
            self.tri_att_end(
                z,
                mask=pair_mask,
                chunk_size=chunk_size_tri_attn,
                use_kernels=use_cuequiv_kernels,
            )
        )
        z = z + self.transition_z(z)

        return s_plm, z
