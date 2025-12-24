from functools import partial

import torch

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.transition import Transition
from kfold.model.layers.primitives import (
    LayerNorm,
    LinearNoBias,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from kfold.model.layers.primitives.dropout import get_dropout_mask
from kfold.model.layers.primitives.utils import permute_final_dims
from kfold.utils.checkpointing import checkpoint_blocks


class OuterProductMean(torch.nn.Module):
    """Outer product mean layer.
    See Section 3 Algorithm 9 of the AlphaFold3 paper.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
    ) -> None:
        """Initialize the outer product mean layer.

        Parameters
        ----------
        c_in : int
            The input dimension.
        c_hidden : int
            The hidden dimension.
        c_out : int
            The output dimension.

        """
        super().__init__()
        self.c_hidden: int = c_hidden
        self.layernorm = LayerNorm(c_in)
        self.linear_a = LinearNoBias(c_in, c_hidden, init="default")
        self.linear_b = LinearNoBias(c_in, c_hidden, init="default")
        # NOTE (SeonghwanSeo): LinearNoBias is used instead of Linear
        # in contrast to AF3, since we do not want the number of
        # ensemble structure members to affect the output bias.
        self.linear_o = LinearNoBias(c_hidden * c_hidden, c_out, init="final")

    def _outer_product_mean(
        self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the outer product mean.

        Parameters
        ----------
        a : torch.Tensor
            The first projected tensor (*, L, E, c_hidden).
        b : torch.Tensor
            The second projected tensor (*, L, E, c_hidden).
        mask : torch.Tensor
            The mask tensor (*, L, E).

        Returns
        -------
        torch.Tensor
            The output tensor (*, L, L, c_out).
        """
        # Compute mask
        pair_mask = mask.unsqueeze(-2) & mask.unsqueeze(-3)  # (*, L, L, E)
        is_valid = pair_mask.any(dim=-1, keepdim=True)  # (*, L, L, 1)
        norm = pair_mask.sum(dim=-1, keepdim=True).to(a.dtype)  # (*, L, L, 1)

        # [*, L, E, c_h] x [*, L, E, c_h] -> [*, L, L, c_h, c_h]
        outer = torch.einsum("...inc,...jnd->...ijcd", a, b)
        # [*, L, L, c_h, c_h] -> [*, L, L, c_h*c_h]
        outer = outer.flatten(start_dim=-2)

        # [*, L, L, Ch**2] -> [*, L, L, c_out]
        z = self.linear_o(outer)

        # Compute mean
        z = (z * is_valid.to(z.dtype)) / norm.clamp(min=1)
        return z

    def _chunk_outer_product_mean(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Compute in chunks to save memory

        Parameters
        ----------
        a : torch.Tensor
            The left tensor (*, L, E, c_hidden).
        b : torch.Tensor
            The right tensor (*, L, E, c_hidden).
        mask : torch.Tensor
            The mask tensor (*, L, E).
        chunk_size : int
            The chunk size for processing, default 128.

        Returns
        -------
        torch.Tensor
            The output tensor (*, L, L, c_out).
        """
        # Compute mask globally [Batch, L, L, 1]
        # This is cheap (boolean/byte ops) compared to the float32 matmuls below
        pair_mask = mask.unsqueeze(-2) & mask.unsqueeze(-3)
        is_valid = pair_mask.any(dim=-1, keepdim=True)
        norm = pair_mask.sum(dim=-1, keepdim=True).to(a.dtype)

        L = a.shape[1]
        z_chunks: list[torch.Tensor] = []

        for st in range(0, L, chunk_size):
            # 1. Slice Inputs along L (dimension 1)
            # Use narrow to be safe against variable batch dimensions
            # narrow(dimension, start, length)
            end = min(st + chunk_size, L)

            # a_chunk: [*, chunk, E, c_h]
            a_chunk = a[..., st:end, :, :]

            # 2. Compute Outer Product Sum
            # Sum over Ensemble (E) dimension
            # [*, chunk, E, c_h] x [*, L, E, c_h] -> [*, chunk, L, c_h, c_h]
            outer_chunk = torch.einsum("...iec,...jed->...ijcd", a_chunk, b)

            # 3. Project (Linear + Bias)
            outer_chunk = outer_chunk.flatten(start_dim=-2)
            z_chunk = self.linear_o(outer_chunk)  # [*, chunk, L, c_out]

            # 4. Normalize & Mask
            # Slice the norm/valid tensors
            norm_chunk = norm[..., st:end, :, :]  # [*, chunk, L, 1]
            valid_chunk = is_valid[..., st:end, :, :]  # [*, chunk, L, 1]

            # Apply division after bias addition (matching AF3 logic)
            z_chunk = (z_chunk * valid_chunk.to(z_chunk.dtype)) / norm_chunk.clamp(min=1)

            z_chunks.append(z_chunk)

        # Concatenate chunks along L dimension
        z = torch.cat(z_chunks, dim=-3)  # [*, L, L, c_out]

        return z

    def forward(
        self,
        e: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        e : torch.Tensor
            The structure ensemble feature tensor (*, L, E, c_in).
        mask : torch.Tensor
            The mask tensor (*, L, E).
        chunk_size : int | None
            The chunk size for processing, default None.

        Returns
        -------
        torch.Tensor
            The output tensor (*, L, L, c_out).
        """
        # Compute projections
        mask_float = mask.to(e)[..., None]  # (*, L, E, 1)

        m_norm = self.layernorm(e)  # (*, L, E, c_in)

        # Line 2
        a = self.linear_a(m_norm) * mask_float  # (*, L, E, c_h)
        b = self.linear_b(m_norm) * mask_float  # (*, L, E, c_h)
        del m_norm  # free memory for float32 tensor (layernorm output)

        # Line 3-4
        if chunk_size is not None:
            assert self.training is False, "Chunking only supported during inference."
            z = self._chunk_outer_product_mean(a, b, mask, chunk_size)
        else:
            z = self._outer_product_mean(a, b, mask)
        return z


class PairWeightedAveraging(torch.nn.Module):
    """Pair weighted averaging layer."""

    def __init__(
        self,
        c_in: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
    ) -> None:
        """Initialize the pair weighted averaging layer.

        Parameters
        ----------
        c_in: int
            The dimension of the input tensor.
        c_z: int
            The dimension of the input pairwise tensor.
        num_heads: int
            The number of heads.
        inf: float
            The value to use for masking, default 1e6.
        """
        super().__init__()
        assert c_in % num_heads == 0, "c_in must be divisible by num_heads."
        c_head = c_in // num_heads

        self.c_in: int = c_in
        self.c_z: int = c_z
        self.c_head: int = c_head
        self.num_heads: int = num_heads
        self.inf: float = inf

        self.layernorm_in = LayerNorm(c_in)
        self.layernorm_z = LayerNorm(c_z)

        self.linear_v = LinearNoBias(c_in, c_head * num_heads, init="default")
        self.linear_g = LinearNoBias(c_in, c_head * num_heads, init="gating")
        self.linear_z = LinearNoBias(c_z, num_heads, init="default")
        self.linear_o = LinearNoBias(c_head * num_heads, c_in, init="final")

    def forward(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        e : torch.Tensor
            The structure ensemble feature tensor (*, L, E, c_in).
        z : torch.Tensor
            The input pairwise tensor (*, L, L, D)
        mask : torch.Tensor
            The pairwise mask tensor (*, L, L)

        Returns
        -------
        torch.Tensor
            The output tensor (*, L, E, D)

        """
        H = self.num_heads
        C = self.c_head

        # Move ensemble dim to front (treated as batch dim)
        # [*, L, E, c_in] -> [*, E, L, c_in]
        e = permute_final_dims(e, (1, 0, 2))  # [*, E, L, c_in]

        # Prepare z
        # [*, L, L, D] -> [*, H, L, L]
        z = self.layernorm_z(z)
        z = self.linear_z(z)  # [*, L, L, H]
        z = permute_final_dims(z, (2, 0, 1))  # [*, H, L, L]
        # Apply mask
        bias = -self.inf * (~mask).to(z.dtype)[..., None, :, :]  # [*, 1, L, L]
        z = z + bias  # (*, H, L, L)
        # Compute attention weights
        w = torch.softmax(z, dim=-1)  # [*, H, L, L]

        # Prepare v
        # [*, E, L, c_in] -> [*, E, L, H, c_h]
        e = self.layernorm_in(e)  # [*, E, L, c_in]
        v: torch.Tensor = self.linear_v(e).unflatten(-1, (H, C))  # [*, E, L, H, c_h]
        v = v.transpose(-3, -2)  # [*, E, H, L, c_h]

        # Compute output
        # [*, H, L(q), L(k)] x [*, E, H, L(k), c_h] -> [*, E, L(q), H, c_h]
        o = torch.einsum("...hqk,...ehkd->...eqhd", w, v)  # [*, E, L, H, c_h]

        # Compute gating
        g = torch.sigmoid(self.linear_g(e)).unflatten(-1, (H, C))  # [*, E, L, H, c_h]
        o = o * g  # [*, E, L, H, c_h]

        # Final linear
        # [*, E, L, H, c_h] -> [*, E, L, c_in]
        o = o.flatten(start_dim=-2)  # [*, E, L, H*c_h]
        o = self.linear_o(o)  # [*, E, L, c_in]

        # Move ensemble dim back to original position
        # [*, E, L, c_in] = [*, L, E, c_in]
        o = permute_final_dims(o, (1, 0, 2))

        return o


class EnsembleModule(torch.nn.Module):
    """Ensemble module to communicate between structure ensemble
    features and pairwise features."""

    def __init__(
        self,
        channel_struct_input: int,
        channel_s: int = 384,
        channel_z: int = 128,
        channel_struct: int = 128,
        channel_hidden_opm: int = 32,
        num_heads_pwa: int = 8,
        num_heads_tri_attn: int = 4,
        num_blocks: int = 4,
        struct_dropout: float = 0.15,
        z_dropout: float = 0.25,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        """Initialize the Ensemble module.

        Parameters
        ----------
        channel_struct_input : int
            The input structure ensemble embedding size.
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        channel_struct : int
            The structure ensemble embedding size.
        channel_hidden_opm : int
            The hidden size for the outer product mean.
        num_heads_pwa : int
            The number of heads for the ensemble attention.
        num_heads_tri_attn : int
            The number of heads for the triangle attention.
        num_blocks : int
            The number of Ensemble blocks.
        struct_dropout : float
            The Ensemble dropout.
        z_dropout : float
            The pairwise dropout.
        blocks_per_ckpt : int | None
            The number of blocks per checkpoint, default None.
        """
        super().__init__()
        self.num_blocks: int = num_blocks
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

        self.linear_struct = LinearNoBias(
            channel_struct_input, channel_struct, init="default"
        )
        self.linear_s_input = LinearNoBias(channel_s, channel_struct, init="default")
        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                EnsembleBlock(
                    channel_struct=channel_struct,
                    channel_z=channel_z,
                    channel_hidden_opm=channel_hidden_opm,
                    num_heads_pwa=num_heads_pwa,
                    num_heads_tri_attn=num_heads_tri_attn,
                    struct_dropout=struct_dropout,
                    z_dropout=z_dropout,
                    is_last_block=(i == num_blocks - 1),
                )
            )

    def forward(
        self,
        f_input: FoldingInput,
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        chunk_size_opm: int | None = None,
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_mul: bool = True,
        use_cuequiv_attn: bool = True,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input features
        z : Tensor
            The pairwise embeddings
        s_inputs : Tensor
            The input single embeddings

        Returns
        -------
        z_upd: Tensor
            The output pairwise embeddings.

        """
        # Set chunk sizes
        if self.training:
            assert chunk_size_opm is None, "During training, chunk_size_opm must be None."
            assert chunk_size_tri_attn is None, (
                "During training, chunk_size_tri_attn must be None."
            )

        # Compute input projections
        e: torch.Tensor = f_input.pretrained.structure_embedding  # [B, L, E, c_struct]
        # FIXME: add a better way to handle missing structure embeddings
        struct_mask = (e != 0).any(-1)  # [B, L, E]

        e = self.linear_struct(e)  # [B, L, E, c_e]
        e = e + self.linear_s_input(s_inputs).unsqueeze(-2)  # [B, L, E, c_e]

        token_mask = f_input.token.pad_mask  # [B, L]
        token_mask = token_mask[:, :, None] & token_mask[:, None, :]  # [B, L, L]

        # Perform blocks
        blocks = [
            partial(
                b,
                token_mask=token_mask,
                struct_mask=struct_mask,
                use_cuequiv_mul=use_cuequiv_mul,
                use_cuequiv_attn=use_cuequiv_attn,
                chunk_size_opm=chunk_size_opm,
                chunk_size_tri_attn=chunk_size_tri_attn,
            )
            for b in self.blocks
        ]

        if self.training and torch.is_grad_enabled():
            e, z = checkpoint_blocks(
                blocks,
                (e, z),
                self.blocks_per_ckpt,
                use_reentrant=False,
            )
        else:
            for block in blocks:
                e, z = block(e, z)

        return z


class EnsembleBlock(torch.nn.Module):
    """Ensemble embedding block."""

    def __init__(
        self,
        channel_struct: int = 64,
        channel_z: int = 128,
        channel_hidden_opm: int = 32,
        num_heads_pwa: int = 8,
        num_heads_tri_attn: int = 4,
        struct_dropout: float = 0.15,
        z_dropout: float = 0.25,
        is_last_block: bool = False,
    ) -> None:
        """Initialize the Ensemble block.

        Parameters
        ----------
        channel_struct : int
            The structure ensemble embedding size, by default 64.
        channel_z : int, optional
            The pairwise embedding size, by default 128.
        channel_hidden_opm : int, optional
            The hidden size for the outer product mean, by default 32.
        num_heads_pwa : int, optional
            The number of heads for the ensemble attention, by default 8.
        num_heads_tri_attn : int, optional
            The number of heads for the triangle attention, by default 4.
        struct_dropout : float, optional
            The dropout rate for the ensemble stack, by default 0.15.
        z_dropout : float, optional
            The dropout rate for the pairwise stack, by default 0.25.
        is_last_block : bool, optional
            Whether this is the last block, by default False.

        """
        super().__init__()
        self.e_dropout: float = struct_dropout
        self.z_dropout: float = z_dropout

        self.outer_product_mean = OuterProductMean(
            c_in=channel_struct,
            c_hidden=channel_hidden_opm,
            c_out=channel_z,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            channel_z, num_heads_tri_attn, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            channel_z, num_heads_tri_attn, inf=1e9
        )

        self.transition_z = Transition(channel_z, expansion_factor=4)

        self.is_last_block: bool = is_last_block
        if not self.is_last_block:
            # NOTE: The ensemble feature update is skipped in the last block
            # since it is not used afterwards.
            self.pair_weighted_averaging = PairWeightedAveraging(
                c_in=channel_struct,
                c_z=channel_z,
                num_heads=num_heads_pwa,
            )
            self.transition_e = Transition(channel_struct, expansion_factor=4)

    def forward(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        token_mask: torch.Tensor,
        struct_mask: torch.Tensor,
        use_cuequiv_mul: bool = True,
        use_cuequiv_attn: bool = True,
        chunk_size_opm: int | None = None,
        chunk_size_tri_attn: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        e : torch.Tensor
            The structure ensemble representation
        z : torch.Tensor
            The pair representation
        token_mask : torch.Tensor
            The token mask
        struct_mask : torch.Tensor
            The structure ensemble mask

        Returns
        -------
        torch.Tensor
            The output ensemble embeddings.
        torch.Tensor
            The output pairwise embeddings.

        """
        # Communication
        z = z + self.outer_product_mean(e, struct_mask, chunk_size=chunk_size_opm)

        # Pairwise stack
        dropout = get_dropout_mask(z, self.z_dropout, self.training)
        z = z + dropout * self.tri_mul_out(
            z, mask=token_mask, use_kernels=use_cuequiv_mul
        )

        dropout = get_dropout_mask(z, self.z_dropout, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=token_mask, use_kernels=use_cuequiv_mul)

        dropout = get_dropout_mask(z, self.z_dropout, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn,
        )

        dropout = get_dropout_mask(z, self.z_dropout, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn,
        )

        z = z + self.transition_z(z)

        if not self.is_last_block:
            # Ensemble stack
            ensb_dropout = get_dropout_mask(e, self.e_dropout, self.training)
            e = e + ensb_dropout * self.pair_weighted_averaging(e, z, token_mask)
            e = e + self.transition_e(e)

        return e, z
