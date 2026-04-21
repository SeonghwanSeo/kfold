from functools import partial

import torch
import torch.nn as nn

from kfold.model.layers.alphafold3.attention_pair_bias import SelfAttentionPairBias
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
from kfold.model.layers.primitives.utils import permute_final_dims
from kfold.utils.checkpointing import checkpoint_blocks
from kfold.utils.torch import add


class PairwiseProdDiff(nn.Module):
    """Convert single embeddings to pairwise embeddings.
    Inspired by ESMFold's implementation.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        assert c_out % 2 == 0, "c_out must be even."
        self.c_in: int = c_in
        self.c_out: int = c_out
        self.c_hid: int = c_out // 2

        self.layernorm = LayerNorm(c_in)
        self.linear_in = LinearNoBias(c_in, 2 * c_out, init="default")
        self.linear_out = Linear(2 * c_out, c_out, init="final")

    def forward(
        self,
        s: torch.Tensor,
        intra_mask: torch.Tensor,
        inter_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise embeddings from single embeddings.

        Parameters
        ----------
        s : torch.Tensor
            The single representation (*, L, c_in).
        intra_mask : torch.Tensor
            The intra-chain mask of shape (*, L, L).
        inter_mask : torch.Tensor
            The inter-chain mask of shape (*, L, L).

        Returns
        -------
        torch.Tensor
            The output tensor (*, L, L, c_out).
        """
        s = self.layernorm(s)  # (*, L, c_in)
        s_i, s_j = self.linear_in(s).chunk(2, dim=-1)  # 2 * (*, L, 2*c_hid)

        dim = self.c_hid
        s_i = s_i[..., :, None, :].unflatten(-1, (2, dim))  # (*, L, 1, 2, c_hid)
        s_j = s_j[..., None, :, :].unflatten(-1, (2, dim))  # (*, 1, L, 2, c_hid)

        # Combine Diff (Asymmetry) and Prod (Correlation)
        z = torch.cat([s_i - s_j, s_i * s_j], dim=-1)  # (*, L, L, 2, c_out)

        # Apply masks
        mask = torch.stack([intra_mask, inter_mask], dim=-1)  # (*, L, L, 2)
        z = z * mask.to(z.dtype)[..., None]  # (*, L, L, 2, c_out)
        z = z.flatten(-2, -1)  # (*, L, L, 2*c_out)

        z = self.linear_out(z)  # (*, L, L, c_out)
        return z


class PLMEmbedder(nn.Module):
    """
    Separated embedder of PLMModule to avoid redundant computation across recycling steps.
    """

    def __init__(
        self,
        channel_s_input: int = 384,
        channel_plm_input: int = 1152,
        channel_plm: int = 768,
    ) -> None:
        super().__init__()
        self.linear_s_input = LinearNoBias(channel_s_input, channel_plm, init="default")
        self.linear_seq_emb = LinearNoBias(channel_plm_input, channel_plm, init="default")

    def forward(
        self,
        s_input: torch.Tensor,
        seq_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        s_input : torch.Tensor
            The input single representations of shape (B, L, C_s)
        seq_emb : torch.Tensor
            The sequence embeddings from PLM of shape (B, L, C_s_plm)

        Returns
        -------
        torch.Tensor
            The fused single representations of shape (B, L, C_s * 2)
        """
        s_input_proj = self.linear_s_input(s_input)  # (B, L, C_s_plm)
        s_seq_emb_proj = self.linear_seq_emb(seq_emb)  # (B, L, C_s_plm)
        s_fused = s_input_proj + s_seq_emb_proj  # (B, L, C_s_plm)
        return s_fused


class PLMModule(nn.Module):
    def __init__(
        self,
        channel_z: int = 128,
        channel_plm: int = 768,
        num_heads_attn: int = 16,
        num_heads_tri_attn: int = 4,
        num_blocks: int = 4,
        dropout_plm: float = 0.15,
        dropout_z: float = 0.25,
        use_qk_norm: bool = False,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                PLMBlock(
                    channel_z=channel_z,
                    channel_plm=channel_plm,
                    num_heads_attn=num_heads_attn,
                    num_heads_tri_attn=num_heads_tri_attn,
                    dropout_plm=dropout_plm,
                    dropout_z=dropout_z,
                    use_qk_norm=use_qk_norm,
                    is_last_block=(i == num_blocks - 1),
                )
            )
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

    def forward(
        self,
        z: torch.Tensor,
        s_plm: torch.Tensor,
        asym_id: torch.Tensor,
        mask: torch.Tensor,
        use_cuequiv_kernels: bool = False,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pair representations of shape (B, L, L, C_z)
        s_plm : torch.Tensor
            The sequence embeddings from PLM of shape (B, L, C_s_plm)
        asym_id : torch.Tensor
            The asymmetry IDs of shape (B, L)
        mask : torch.Tensor
            The token mask of shape (B, L)
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

        # PLM Blocks
        blocks = [
            partial(
                b,
                mask=mask,
                pair_mask=pair_mask,
                intra_mask=intra_mask,
                inter_mask=inter_mask,
                use_cuequiv_kernels=use_cuequiv_kernels,
            )
            for b in self.blocks
        ]
        s_plm, z = checkpoint_blocks(
            blocks,
            (s_plm, z),
            self.blocks_per_ckpt,
            use_reentrant=False,
        )
        return z


class PLMBlock(nn.Module):
    def __init__(
        self,
        channel_z: int = 128,
        channel_plm: int = 768,
        num_heads_attn: int = 16,
        num_heads_tri_attn: int = 4,
        dropout_plm: float = 0.15,
        dropout_z: float = 0.25,
        use_qk_norm: bool = False,
        is_last_block: bool = False,
    ) -> None:
        super().__init__()
        self.channel_z: int = channel_z
        self.channel_plm: int = channel_plm

        self.pairwise_proj = PairwiseProdDiff(channel_plm, channel_z)

        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)
        self.tri_att_start = TriangleAttentionStartingNode(channel_z, num_heads_tri_attn)
        self.tri_att_end = TriangleAttentionEndingNode(channel_z, num_heads_tri_attn)

        self.transition_z = Transition(channel_z, expansion_factor=4)

        self.dropout_plm = nn.Dropout(dropout_plm)
        self.dropout_rowwise_z = DropoutRowwise(dropout_z)
        self.dropout_columnwise_z = DropoutColumnwise(dropout_z)

        self.is_last_block: bool = is_last_block
        if not self.is_last_block:
            self.proj_z_to_bias = nn.Sequential(
                LayerNorm(channel_z),
                LinearNoBias(channel_z, num_heads_attn, init="default"),
            )
            self.attention = SelfAttentionPairBias(
                channel_a=channel_plm,
                channel_s=None,
                num_heads=num_heads_attn,
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
        use_cuequiv_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s_plm : torch.Tensor
            The sequence embeddings
        z : torch.Tensor
            The pair representations
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
        _add = partial(add, inplace=not self.training)

        # Step 1: single to pair
        z = _add(z, self.pairwise_proj(s_plm, intra_mask, inter_mask))

        # Step 2: pair to single
        if not self.is_last_block:
            pair_bias = self.proj_z_to_bias(z)  # [*, L, L, H]
            pair_bias = permute_final_dims(pair_bias, (2, 0, 1))  # [*, H, L, L]
            s_plm = s_plm + self.dropout_plm(  # Avoid in-place modification.
                self.attention(
                    a=s_plm,  # [*, L, C_plm]
                    s=None,
                    pair_bias=pair_bias,  # [*, H, L, L]
                    mask=mask,  # [*, L]
                )
            )
            s_plm = _add(s_plm, self.transition_plm(s_plm))

        # Step 3: pair to pair
        z = _add(
            z,
            self.dropout_rowwise_z(
                self.tri_mul_out(z, pair_mask, use_kernels=use_cuequiv_kernels)
            ),
        )
        z = _add(
            z,
            self.dropout_rowwise_z(
                self.tri_mul_in(z, pair_mask, use_kernels=use_cuequiv_kernels)
            ),
        )
        z = _add(
            z,
            self.dropout_rowwise_z(
                self.tri_att_start(z, pair_mask, use_kernels=use_cuequiv_kernels)
            ),
        )
        z = _add(
            z,
            self.dropout_columnwise_z(
                self.tri_att_end(z, pair_mask, use_kernels=use_cuequiv_kernels)
            ),
        )
        z = _add(z, self.transition_z(z))

        return s_plm, z
