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
from torch import nn


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_heads: int,
        qk_ln: bool,
        qk_bias: bool,
    ):
        super().__init__()
        c_head = c_s // n_heads
        self.node_dim = c_s
        self.pair_dim = c_z
        self.n_heads = n_heads
        self.scale = c_head**-0.5

        self.to_qkv = nn.Linear(c_s, c_s * 3, bias=qk_bias)
        self.to_g = nn.Linear(c_s, c_s)
        self.to_out_node = nn.Linear(c_s, c_s)
        self.node_norm = nn.LayerNorm(c_s)
        self.q_layer_norm = nn.LayerNorm(c_s) if qk_ln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(c_s) if qk_ln else nn.Identity()
        self.to_bias = nn.Linear(c_z, n_heads, bias=False)
        self.pair_norm = nn.LayerNorm(c_z)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-head scalar Attention Layer

        :param s: scalar features of shape [B, L, A, c_s]
        :param z: pair features of shape [B, L, A, A, c_z]
        :param mask: optional boolean tensor of atom adjacencies, shape [B, L, A]
        :return:
        """
        H = self.n_heads

        s = self.node_norm(s)
        z = self.pair_norm(z)
        q, k, v = self.to_qkv(s).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(s)
        # [B, L, A, A, heads] -> [B, heads, L, A, A]
        b = rearrange(self.to_bias(z), "b ... h -> b h ...")
        q, k, v, g = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=H), (q, k, v, g)
        )
        attn_feats = self._attn(q, k, v, b, mask)  # [B, heads, L, A, dim_head]
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h l n d -> b l n (h d)", h=H
        )  # [B, L, A, inner_dim]
        return self.to_out_node(attn_feats)  # [B, L, A, dim_out] or [B, L, inner_dim]

    def _attn(self, q, k, v, b, mask: torch.Tensor) -> torch.Tensor:
        """Perform attention update"""
        sim = (
            torch.einsum("b ... i d, b ... j d -> b ... i j", q, k) * self.scale
        )  # [B, heads, L, A, A]
        mask = rearrange(mask, "b ... l j -> b ... () l () j")  # [B, 1, L, 1, A]
        sim = sim.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(sim + b, dim=-1).nan_to_num(0.0)  # [B, heads, L, A, A]
        return torch.einsum("b ... i j, b ... j d -> b ... i d", attn, v)


class MultiHeadPairBiasedAttention(nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_heads: int,
        use_qk_ln: bool = True,
        use_qk_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(c_s)
        self.mlp = nn.Sequential(
            nn.Linear(c_s, c_s * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_s * 2, c_s),
        )
        self.mha = PairBiasAttention(
            c_s=c_s,
            c_z=c_z,
            n_heads=n_heads,
            qk_ln=use_qk_ln,
            qk_bias=use_qk_bias,
        )

    def forward(self, s, z, mask):
        """
        Args:
            s: Input sequence representation, shape [B, L, A, dim_token]
            z: Pair represnetation, shape [B, L, A, A, dim_pair]
            mask: Binary mask, shape [B, L, A]

        Returns:
            Updated sequence representation, shape [B, L, A, dim_token].
        """
        s = s + self.mlp(self.norm(s))
        s = self.mha(s, z, mask)
        return s


class MultiheadAttnAndTransition(torch.nn.Module):
    """
    Layer that applies mha and transition to a sequence representation.
    Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        c_s: Token dimension in sequence representation.
        c_z: Dimension of pair representation.
        n_heads: Number of attention heads.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mhba = MultiHeadPairBiasedAttention(
            c_s=c_s,
            c_z=c_z,
            n_heads=n_heads,
        )

        self.norm = torch.nn.LayerNorm(c_s)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(c_s, c_s * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(c_s * 2, c_s),
        )

    def forward(self, s, z, mask):
        """
        Args:
            s: Single representation, shape [b, n, c_s]
            z: Pair representation (shape [b, n, n, c_z])
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        r = self.mhba(s, z, mask)
        s = s + r

        r = self.mlp(self.norm(s))
        s = s + r
        return s * mask[..., None]
