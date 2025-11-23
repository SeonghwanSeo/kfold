import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from kfold.data.model_input import FoldingInput

from . import initialize as init
from .dropout import get_dropout_mask
from .transition import Transition
from .triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from .triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class OuterProductMean(nn.Module):
    """Outer product mean layer."""

    def __init__(self, c_in: int, c_hidden: int, c_out: int) -> None:
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
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)
        init.final_init_(self.proj_o.weight)
        init.final_init_(self.proj_o.bias)

    def forward(
        self,
        m: Tensor,
        mask: Tensor,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        m : torch.Tensor
            The sequence tensor (B, S, N, c_in).
        mask : torch.Tensor
            The mask tensor (B, S, N).

        Returns
        -------
        torch.Tensor
            The output tensor (B, N, N, c_out).

        """
        # Expand mask
        mask = mask.unsqueeze(-1).to(m)

        # Compute projections
        m = self.norm(m)
        a = self.proj_a(m) * mask
        b = self.proj_b(m) * mask

        # Compute outer product mean
        if chunk_size is not None and not self.training:
            # Compute pairwise mask
            for i in range(0, mask.shape[1], 64):
                if i == 0:
                    num_mask = (
                        mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
                    ).sum(1)
                else:
                    num_mask += (
                        mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
                    ).sum(1)
            num_mask = num_mask.clamp(min=1)

            # Compute sequentially in chunks
            for i in range(0, self.c_hidden, chunk_size):
                a_chunk = a[:, :, :, i : i + chunk_size]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, i * self.c_hidden : (i + chunk_size) * self.c_hidden
                ]

                z = torch.einsum("bsic,bsjd->bijcd", a_chunk, b)
                z = z.reshape(*z.shape[:3], -1)
                z = z / num_mask

                # Project to output
                if i == 0:
                    z_out = z.to(m) @ sliced_weight_proj_o.T
                else:
                    z_out = z_out + z.to(m) @ sliced_weight_proj_o.T

            z_out = z_out + self.proj_o.bias  # add bias
            return z_out
        else:
            mask = mask[:, :, None, :] * mask[:, :, :, None]
            num_mask = mask.sum(1).clamp(min=1)
            z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())
            z = z.reshape(*z.shape[:3], -1)
            z = z / num_mask

            # Project to output
            z = self.proj_o(z.to(m))
            return z


class PairWeightedAveraging(nn.Module):
    """Pair weighted averaging layer."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_h: int,
        num_heads: int,
        inf: float = 1e6,
    ) -> None:
        """Initialize the pair weighted averaging layer.

        Parameters
        ----------
        c_m: int
            The dimension of the input sequence.
        c_z: int
            The dimension of the input pairwise tensor.
        c_h: int
            The dimension of the hidden.
        num_heads: int
            The number of heads.
        inf: float
            The value to use for masking, default 1e6.

        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf

        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)

        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        m: Tensor,
        z: Tensor,
        mask: Tensor,
        chunk_heads: bool = False,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        m : torch.Tensor
            The input sequence tensor (B, S, N, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor (B, S, N, D)

        """
        # Compute layer norms
        m = self.norm_m(m)
        z = self.norm_z(z)

        if chunk_heads and not self.training:
            # Compute heads sequentially
            o_chunks = []
            for head_idx in range(self.num_heads):
                sliced_weight_proj_m = self.proj_m.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_g = self.proj_g.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_z = self.proj_z.weight[head_idx : (head_idx + 1), :]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, head_idx * self.c_h : (head_idx + 1) * self.c_h
                ]

                # Project input tensors
                v: Tensor = m @ sliced_weight_proj_m.T
                v = v.reshape(*v.shape[:3], 1, self.c_h)
                v = v.permute(0, 3, 1, 2, 4)

                # Compute weights
                b: Tensor = z @ sliced_weight_proj_z.T
                b = b.permute(0, 3, 1, 2)
                b = b + (1 - mask[:, None]) * -self.inf
                w = torch.softmax(b, dim=-1)

                # Compute gating
                g: Tensor = m @ sliced_weight_proj_g.T
                g = g.sigmoid()

                # Compute output
                o = torch.einsum("bhij,bhsjd->bhsid", w, v)
                o = o.permute(0, 2, 3, 1, 4)
                o = o.reshape(*o.shape[:3], 1 * self.c_h)
                o_chunks = g * o
                if head_idx == 0:
                    o_out = o_chunks @ sliced_weight_proj_o.T
                else:
                    o_out += o_chunks @ sliced_weight_proj_o.T
            return o_out
        else:
            # Project input tensors
            v: Tensor = self.proj_m(m)
            v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
            v = v.permute(0, 3, 1, 2, 4)

            # Compute weights
            b: Tensor = self.proj_z(z)
            b = b.permute(0, 3, 1, 2)
            b = b + (1 - mask[:, None]) * -self.inf
            w = torch.softmax(b, dim=-1)

            # Compute gating
            g: Tensor = self.proj_g(m)
            g = g.sigmoid()

            # Compute output
            o = torch.einsum("bhij,bhsjd->bhsid", w, v)
            o = o.permute(0, 2, 3, 1, 4)
            o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
            o = self.proj_o(g * o)
            return o


class MSAModule(nn.Module):
    """MSA module."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        s_input_dim: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        use_paired_feature: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the MSA module.

        Parameters
        ----------
        msa_s : int
            The MSA embedding size.
        token_z : int
            The token pairwise embedding size.
        s_input_dim : int
            The input sequence dimension.
        msa_blocks : int
            The number of MSA blocks.
        msa_dropout : float
            The MSA dropout.
        z_dropout : float
            The pairwise dropout.
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        use_paired_feature : bool, optional
            Whether to use the paired feature, by default False

        """
        super().__init__()
        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature

        self.s_proj = nn.Linear(s_input_dim, msa_s, bias=False)
        self.msa_proj = nn.Linear(
            33 + 2 + int(use_paired_feature),
            msa_s,
            bias=False,
        )
        self.layers = nn.ModuleList()
        for _ in range(msa_blocks):
            self.layers.append(
                MSALayer(
                    msa_s,
                    token_z,
                    msa_dropout,
                    z_dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                )
            )

    def forward(
        self,
        z: Tensor,
        emb: Tensor,
        f_input: FoldingInput,
        use_kernels: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        emb : Tensor
            The input embeddings
        feats : Dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The output pairwise embeddings.

        """
        # Set chunk sizes
        if not self.training:
            if z.shape[1] > 384:
                chunk_heads_pwa = True
                chunk_size_transition_z = 64
                chunk_size_transition_msa = 32
                chunk_size_outer_product = 4
                chunk_size_tri_attn = 128
            else:
                chunk_heads_pwa = False
                chunk_size_transition_z = None
                chunk_size_transition_msa = None
                chunk_size_outer_product = None
                chunk_size_tri_attn = 512
        else:
            chunk_heads_pwa = False
            chunk_size_transition_z = None
            chunk_size_transition_msa = None
            chunk_size_outer_product = None
            chunk_size_tri_attn = None

        # Load relevant features

        msa = torch.zeros(
            (f_input.batch_size, 2048, f_input.num_tokens, 33),
            device=f_input.token.res_type.device,
        )  # [B, 2048, L, 33]
        msa[:, 0, :, 1:] = f_input.token.res_type  # [B, 2048, L, 33]
        has_deletion = msa[..., :1] * 0.0  # [B, 2048, L, 1]
        deletion_value = has_deletion  # [B, 2048, L, 1]
        is_paired = deletion_value.clone()  # [B, 2048, L, 1]
        is_paired[:, 0] = 1.0
        msa_mask = is_paired.squeeze(-1)  # [B, 2048, L]
        token_mask = f_input.token.pad_mask.float()  # [B, L]
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

        # Compute input projections
        m = self.msa_proj(m)
        m = m + self.s_proj(emb).unsqueeze(1)

        # Perform MSA blocks
        if self.training:
            for i in range(self.msa_blocks):
                z, m = torch.utils.checkpoint.checkpoint(
                    self.layers[i],
                    z,
                    m,
                    token_mask,
                    msa_mask,
                    use_kernels,
                    chunk_heads_pwa,
                    chunk_size_transition_z,
                    chunk_size_transition_msa,
                    chunk_size_outer_product,
                    chunk_size_tri_attn,
                    use_reentrant=False,
                )
        else:
            for i in range(self.msa_blocks):
                z, m = self.layers[i](
                    z,
                    m,
                    token_mask,
                    msa_mask,
                    use_kernels,
                    chunk_heads_pwa,
                    chunk_size_transition_z,
                    chunk_size_transition_msa,
                    chunk_size_outer_product,
                    chunk_size_tri_attn,
                )
        return z


class MSALayer(nn.Module):
    """MSA module."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ) -> None:
        """Initialize the MSA module.

        Parameters
        ----------

        msa_s : int
            The MSA embedding size.
        token_z : int
            The pair representation dimension.
        msa_dropout : float
            The MSA dropout.
        z_dropout : float
            The pair dropout.
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4

        """
        super().__init__()
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.msa_transition = Transition(dim=msa_s, hidden=msa_s * 4)
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.z_transition = Transition(
            dim=token_z,
            hidden=token_z * 4,
        )
        self.outer_product_mean = OuterProductMean(
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
        )

    def forward(
        self,
        z: Tensor,
        m: Tensor,
        token_mask: Tensor,
        msa_mask: Tensor,
        use_kernels: bool = False,
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z: int | None = None,
        chunk_size_transition_msa: int | None = None,
        chunk_size_outer_product: int | None = None,
        chunk_size_tri_attn: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pair representation
        m : Tensor
            The msa representation
        token_mask : Tensor
            The token mask
        msa_mask : Dict[str, Tensor]
            The MSA mask

        Returns
        -------
        Tensor
            The output pairwise embeddings.
        Tensor
            The output MSA embeddings.

        """
        # Communication to MSA stack
        msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)
        m = m + msa_dropout * self.pair_weighted_averaging(
            m, z, token_mask, chunk_heads_pwa
        )
        m = m + self.msa_transition(m, chunk_size_transition_msa)

        # Communication to pairwise stack
        z = z + self.outer_product_mean(m, msa_mask, chunk_size_outer_product)

        # Compute pairwise stack
        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=token_mask, use_kernels=use_kernels)

        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=token_mask, use_kernels=use_kernels)

        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_kernels,
        )

        dropout = get_dropout_mask(self.z_dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_kernels,
        )

        z = z + self.z_transition(z, chunk_size_transition_z)

        return z, m
