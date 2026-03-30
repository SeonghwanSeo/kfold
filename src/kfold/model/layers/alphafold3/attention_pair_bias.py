"""Attention Pair Bias layer.
See Section 3.7 Algorithm 24 of the AlphaFold3 paper.

We implement two variants of this layer:
1. SelfAttentionPairBias: the attention is performed on the same input tensor.
2. CrossAttentionPairBias: the attention is performed on two different input tensors.
"""

import torch
import torch.nn as nn
from einops import rearrange

from kfold.model.layers.primitives import AdaLN, LayerNorm, Linear, LinearNoBias
from kfold.model.layers.primitives.attention import attention_pair_bias


class AttentionPairBias(nn.Module):
    def __init__(
        self,
        channel_a: int,
        num_heads: int,
        *,
        qk_norm: bool = False,
        zero_init_out: bool = False,
        inf: float = 1e6,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        channel_a : int
            The atom/token dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6
        """
        super().__init__()
        assert channel_a % num_heads == 0
        self.channel_a: int = channel_a
        self.num_heads: int = num_heads
        self.head_dim: int = channel_a // num_heads
        self.inf: float = inf

        self.linear_q = Linear(channel_a, channel_a, init="default")
        self.linear_k = LinearNoBias(channel_a, channel_a, init="default")
        self.linear_v = LinearNoBias(channel_a, channel_a, init="default")
        self.linear_g = LinearNoBias(channel_a, channel_a, init="gating")

        if qk_norm:
            self.layernorm_q = LayerNorm(channel_a, create_offset=True)
            self.layernorm_k = LayerNorm(channel_a, create_offset=True)
        else:
            self.layernorm_q = nn.Identity()
            self.layernorm_k = nn.Identity()

        if zero_init_out:
            self.linear_out = LinearNoBias(channel_a, channel_a, init="final")
        else:
            self.linear_out = LinearNoBias(channel_a, channel_a, init="default")

    def _prev_qkv(
        self, a_q: torch.Tensor, a_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the query, key, and value tensors from the input tensors."""
        # [*, L, c] -> [*, L, c]
        q = self.layernorm_q(self.linear_q(a_q))
        k = self.layernorm_k(self.linear_k(a_k))
        v = self.linear_v(a_k)
        # [*, L, c] -> [*, H, L, c_h]
        H = self.num_heads
        q, k, v = map(lambda t: rearrange(t, "... l (h d) -> ... h l d", h=H), (q, k, v))
        return q, k, v

    def _attention(
        self,
        a: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair_bias: torch.Tensor,
        mask: torch.Tensor,
        use_kernels: bool,
    ) -> torch.Tensor:
        # Attention Pair Bias with cuequivariance kernels
        return attention_pair_bias(
            s=a,
            q=q,
            k=k,
            v=v,
            pair_bias=pair_bias,
            mask=mask,
            w_proj_g=self.linear_g.weight,
            b_proj_g=self.linear_g.bias,
            w_proj_o=self.linear_out.weight,
            b_proj_o=self.linear_out.bias,
            num_heads=self.num_heads,
            inf=self.inf,
            use_kernels=use_kernels,
        )


class SelfAttentionPairBias(AttentionPairBias):
    def __init__(
        self,
        channel_a: int,
        num_heads: int,
        channel_s: int | None,
        qk_norm: bool = False,
        inf: float = 1e9,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        channel_a : int
            The atom/token dimension.
        num_heads : int
            The number of heads.
        channel_s : int
            The single conditioning dimension.
        qk_norm : bool, optional
            Whether to apply LayerNorm to Q and K.
        inf : float, optional
            The inf value, by default 1e6
        """
        self.use_single_conditioning: bool = channel_s is not None

        super().__init__(
            channel_a,
            num_heads,
            qk_norm=qk_norm,
            zero_init_out=(not self.use_single_conditioning),
            inf=inf,
        )

        if self.use_single_conditioning:
            assert channel_s is not None
            self.adaln_a = AdaLN(channel_a, channel_s)
            self.linear_ada_out = Linear(channel_s, channel_a, init="gating_closed")
            self.sigmoid = nn.Sigmoid()
        else:
            self.layernorm_a = LayerNorm(channel_a, create_offset=True)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor | None,
        pair_bias: torch.Tensor,
        mask: torch.Tensor,
        use_kernels: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        a : torch.Tensor
            The input tensor (..., L, c_a).
        s : torch.Tensor | None
            The single conditioning tensor (..., L, c_s).
        pair_bias : torch.Tensor
            The pair representation tensor (..., H, L, L)
        mask : torch.Tensor
            The attention mask tensor (..., L)
        use_kernels : bool, optional
            Whether to use custom kernel for attention, by default False

        Returns
        -------
        a : torch.Tensor
            The output tensor (..., L, c_a)
        """
        if self.use_single_conditioning:
            assert s is not None
            a = self.adaln_a(a, s)
        else:
            assert s is None
            a = self.layernorm_a(a)

        q, k, v = self._prev_qkv(a, a)
        a = self._attention(a, q, k, v, pair_bias, mask, use_kernels)

        if self.use_single_conditioning:
            assert s is not None
            a = self.sigmoid(self.linear_ada_out(s)) * a
        return a


class CrossAttentionPairBias(AttentionPairBias):
    def __init__(
        self,
        channel_a: int,
        num_heads: int,
        channel_s: int | None,
        qk_norm: bool = False,
        inf: float = 1e9,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        channel_s : int
            The atom/token dimension.
        channel_z : int
            The input pair bias dimension.
        num_heads : int
            The number of heads.
        channel_s : int
            The single conditioning dimension.
        inf : float, optional
            The inf value, by default 1e6
        """
        self.use_single_conditioning: bool = channel_s is not None

        super().__init__(
            channel_a,
            num_heads,
            qk_norm=qk_norm,
            zero_init_out=(not self.use_single_conditioning),
            inf=inf,
        )

        if self.use_single_conditioning:
            assert channel_s is not None
            self.adaln_a_q = AdaLN(channel_a, channel_s)
            self.adaln_a_k = AdaLN(channel_a, channel_s)
            self.linear_ada_out = Linear(channel_s, channel_a, init="gating_closed")
            self.sigmoid = nn.Sigmoid()
        else:
            self.layernorm_a_q = LayerNorm(channel_a, create_offset=True)
            self.layernorm_a_k = LayerNorm(channel_a, create_offset=True)

    def forward(
        self,
        a_q: torch.Tensor,
        a_k: torch.Tensor,
        s_q: torch.Tensor | None,
        s_k: torch.Tensor | None,
        pair_bias: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        a_q : torch.Tensor
            The query input tensor (..., W, Lq, c_a).
        a_k : torch.Tensor
            The key/value input tensor (..., W, Lk, c_a).
        s_q : torch.Tensor | None
            The query single conditioning tensor (..., W, Lq, c_s).
        s_k : torch.Tensor | None
            The key/value single conditioning tensor (..., W, Lk, c_s).
        pair_bias : torch.Tensor
            The pair representation tensor (..., W, H, Lq, Lk)
        mask : torch.Tensor
            The attention mask tensor (..., W, Lk)


        Returns
        -------
        a_q : torch.Tensor
            The output query tensor. (..., W, Lq, c_a)
        """
        if self.use_single_conditioning:
            assert s_q is not None and s_k is not None
            a_q = self.adaln_a_q(a_q, s_q)
            a_k = self.adaln_a_k(a_k, s_k)
        else:
            assert s_q is None and s_k is None
            a_q = self.layernorm_a_q(a_q)
            a_k = self.layernorm_a_k(a_k)

        # Prepare attention pair bias input
        a = rearrange(a_q, "... w l d -> ... (w l) d")  # flatten for compatibility

        # [*, W, Lq/k, c] -> [*, W, H, Lq/k, c_h]
        q, k, v = self._prev_qkv(a_q, a_k)

        a = self._attention(a, q, k, v, pair_bias, mask, use_kernels=False)

        # Reshape back to original shape
        a_q = rearrange(a, "... (w l) d -> ... w l d", w=a_q.shape[-3])

        if self.use_single_conditioning:
            assert s_q is not None
            a_q = self.sigmoid(self.linear_ada_out(s_q)) * a_q
        return a_q
