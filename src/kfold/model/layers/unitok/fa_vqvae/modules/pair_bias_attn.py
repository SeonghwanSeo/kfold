"""
Code adopted from La-Proteina (https://github.com/NVIDIA-Digital-Bio/la-proteina).
"""

# MIT License

# Copyright (c) 2022 MattMcPartlon

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from einops import rearrange
from torch import Tensor, einsum, nn


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: int | None = None,
        **kawrgs,  # noqa
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()

        if exists(pair_dim) and pair_dim > 0:
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
            self.use_pair_bias = True
        else:
            self.to_bias, self.pair_norm = None, None
            self.use_pair_bias = False

    def pair_bias_attn_layer(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape [B, L, A, dim_token]
        :param pair_feats: pair features of shape [B, L, A, A, dim_pair]
        :param pair_mask: optional boolean tensor of atom adjacencies, shape [B, L, A, A]
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        if self.use_pair_bias and exists(pair_feats):
            pair_feats = self.pair_norm(pair_feats)
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)  # [B, L, A, inner_dim]
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        # [B, L, A, A, heads] -> [B, heads, L, A, A]
        if self.use_pair_bias:
            b = rearrange(self.to_bias(pair_feats), "b ... h -> b h ...")
        else:
            b = torch.zeros(
                q.shape[0],
                self.heads,
                q.shape[1],
                q.shape[2],
                q.shape[2],
                device=q.device,
            )
        q, k, v, g = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g)
        )  # [B, L, A, inner_dim] -> [B, heads, L, A, dim_head]
        attn_feats = self._attn(q, k, v, b, pair_mask)  # [B, heads, L, A, dim_head]
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h l n d -> b l n (h d)", h=h
        )  # [B, L, A, inner_dim]
        return self.to_out_node(attn_feats)  # [B, L, A, dim_out] or [B, L, inner_dim]

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None,
        pair_mask: Tensor | None,
    ) -> Tensor:
        return self.pair_bias_attn_layer(node_feats, pair_feats, pair_mask)

    def _attn(self, q, k, v, b, pair_mask: Tensor | None) -> Tensor:
        """Perform attention update"""
        sim = (
            einsum("b h l i d, b h l j d -> b h l i j", q, k) * self.scale
        )  # [B, heads, L, A, A]
        if pair_mask is not None:
            sim = sim.masked_fill(~pair_mask.unsqueeze(1), max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)  # [B, heads, L, A, A]
        return einsum(
            "b h l i j, b h l j d -> b h l i d", attn, v
        )  # [B, heads, L, A, dim_head]


class MultiHeadPairBiasedAttention(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(
        self,
        dim_token,
        dim_pair,
        nheads,
        use_qkln,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.norm = torch.nn.LayerNorm(dim_token)
        self.dim_pair = dim_pair

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_token, dim_token * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_token * 2, dim_token),
        )
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence representation, shape [B, L, A, dim_token]
            pair_rep: Pair represnetation, shape [B, L, A, A, dim_pair]
            mask: Binary mask, shape [B, L, A]

        Returns:
            Updated sequence representation, shape [B, L, A, dim_token].
        """
        if self.dim_pair > 0:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(3)  # [B, L, A, A]
        else:
            pair_mask = None
        x = x + self.mlp(self.norm(x))  # [B, L, A, dim_token]
        x = self.mha(x, pair_rep, pair_mask)
        return x
