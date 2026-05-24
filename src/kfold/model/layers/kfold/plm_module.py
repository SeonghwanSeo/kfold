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


class PLMInputEmbedder(nn.Module):
    """
    Separated embedder of PLMModule to avoid redundant computation across recycling steps.
    """

    def __init__(
        self,
        channel_seq_emb: int,
        channel_seq_attn: int,
        channel_struct_emb: int,
        channel_z: int = 128,
    ) -> None:
        super().__init__()
        # Initialize with uniform weights.
        self.layernorm_seq = LayerNorm(channel_seq_emb, create_offset=False)
        self.layernorm_struct = LayerNorm(channel_struct_emb, create_offset=False)

        # Attention embedding projection to initialize pair representations.
        # NOTE (Seonghwan): LayerNorm is applied for scalability to sequence length,
        # as the scale of attention maps is reduced by sequence length.
        self.proj_seq_attn = nn.Sequential(
            LayerNorm(channel_seq_attn, create_offset=False),
            LinearNoBias(channel_seq_attn, channel_z, init="relu"),
            nn.ReLU(),
            LinearNoBias(channel_z, channel_z, init="final"),
        )

    def forward(
        self,
        seq_emb: torch.Tensor,
        seq_attn: torch.Tensor,
        struct_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s_input : torch.Tensor
            The input single representations
        seq_emb : torch.Tensor
            The hidden states from the sequence encoder
            of shape (B, L, channel_seq)
        seq_attn : torch.Tensor
            The attention maps from the sequence encoder
            of shape (B, L, num_layer_seq_attn, channel_seq_attn)
        struct_emb : torch.Tensor
            The structure embeddings of shape (B, L, channel_struct)

        Returns
        -------
        s_plm: torch.Tensor
            The fused single representations of shape (B, L, C_plm)
        z_plm: torch.Tensor
            The initial pair representations of shape (B, L, L, C_z)
        """
        # Compute s_plm
        seq_emb = self.layernorm_seq(seq_emb)
        struct_emb = self.layernorm_struct(struct_emb)
        s_plm = torch.cat([seq_emb, struct_emb], dim=-1)
        # Compute z_plm for initializing pair representations.
        z_plm = self.proj_seq_attn(seq_attn.flatten(-2))
        return s_plm, z_plm


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
        s_i, s_j = self.linear_in(s).chunk(2, dim=-1)  # 2 * (*, L, c_hid)
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
        channel_s_inputs: int = 384,
        channel_plm_inputs: int = 2688,
        channel_plm: int = 768,
        channel_z: int = 128,
        num_heads_attn: int = 16,
        num_heads_tri_attn: int = 4,
        num_blocks: int = 4,
        dropout_plm: float = 0.15,
        dropout_z: float = 0.25,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.linear_s_inputs = LinearNoBias(channel_s_inputs, channel_plm)
        self.linear_plm_inputs = LinearNoBias(channel_plm_inputs, channel_plm)
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
                    is_last_block=(i == num_blocks - 1),
                )
            )
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

    def forward(
        self,
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        plm_inputs: torch.Tensor,
        asym_id: torch.Tensor,
        mask: torch.Tensor,
        use_cuequiv_kernels: bool = False,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pair representations of shape (B, L, L, C_z)
        s_inputs : torch.Tensor
            The input single representations of shape (B, L, C_s)
        plm_inputs : torch.Tensor
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
        # Fuse inputs to get initial s_plm.
        s_plm = self.linear_s_inputs(s_inputs) + self.linear_plm_inputs(plm_inputs)

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
        is_last_block: bool = False,
    ) -> None:
        super().__init__()
        self.channel_z: int = channel_z
        self.channel_plm: int = channel_plm

        self.pairwise_proj_intra = PairwiseProdDiff(channel_plm, channel_z)
        self.pairwise_proj_inter = PairwiseProdDiff(channel_plm, channel_z)

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
        z = _add(z, self.pairwise_proj_intra(s_plm) * intra_mask[..., None])
        z = _add(z, self.pairwise_proj_inter(s_plm) * inter_mask[..., None])

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
