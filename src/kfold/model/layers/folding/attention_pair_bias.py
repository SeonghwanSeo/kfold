# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention Pair Bias layer.
See Section 3.7 Algorithm 24 of the AlphaFold3 paper.

We implement two variants of this layer:
1. SelfAttentionPairBias: the attention is performed on the same input tensor.
2. CrossAttentionPairBias: the attention is performed on two different input tensors.
"""

import torch
import torch.nn as nn

from kfold.model.primitives import AdaLN, LayerNorm, Linear, LinearNoBias
from kfold.model.primitives.attention import attention_pair_bias


class AttentionPairBias(nn.Module):
    def __init__(
        self,
        channel_a: int,
        num_heads: int,
        *,
        zero_init_out: bool = False,
        inf: float = 1e9,
        kernel_backend: str = "torch",
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        channel_a : int
            The atom/token dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e9
        """
        super().__init__()
        assert channel_a % num_heads == 0
        self.channel_a: int = channel_a
        self.num_heads: int = num_heads
        self.head_dim: int = channel_a // num_heads
        self.inf: float = inf
        self.kernel_backend = kernel_backend

        self.linear_q = Linear(channel_a, channel_a, init="default")
        self.linear_k = LinearNoBias(channel_a, channel_a, init="default")
        self.linear_v = LinearNoBias(channel_a, channel_a, init="default")
        self.linear_g = LinearNoBias(channel_a, channel_a, init="gating")

        if zero_init_out:
            self.linear_out = LinearNoBias(channel_a, channel_a, init="final")
        else:
            self.linear_out = LinearNoBias(channel_a, channel_a, init="default")


class SelfAttentionPairBias(AttentionPairBias):
    def __init__(
        self,
        channel_a: int,
        num_heads: int,
        channel_s: int | None,
        call_site: str,
        inf: float = 1e9,
        kernel_backend: str = "torch",
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
        inf : float, optional
            The inf value, by default 1e9
        """
        self.use_single_conditioning: bool = channel_s is not None
        if call_site not in {"diffusion_global", "confidence"}:
            raise ValueError(f"Unsupported self APB call site: {call_site!r}")
        self.call_site = call_site

        super().__init__(
            channel_a,
            num_heads,
            zero_init_out=(not self.use_single_conditioning),
            inf=inf,
            kernel_backend=kernel_backend,
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

        a = attention_pair_bias(
            self,
            a,
            a,
            pair_bias,
            mask,
            kernel_backend=self.kernel_backend,
            call_site=self.call_site,
        )

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
        inf: float = 1e9,
        kernel_backend: str = "torch",
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
            The inf value, by default 1e9
        """
        self.use_single_conditioning: bool = channel_s is not None

        # cuEq does not support local windows, which use the Torch backend.
        super().__init__(
            channel_a,
            num_heads,
            zero_init_out=(not self.use_single_conditioning),
            inf=inf,
            kernel_backend="torch" if kernel_backend == "cuequiv" else kernel_backend,
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

        a_q = attention_pair_bias(
            self,
            a_q,
            a_k,
            pair_bias,
            mask,
            kernel_backend=self.kernel_backend,
            call_site="diffusion_local",
        )

        if self.use_single_conditioning:
            assert s_q is not None
            a_q = self.sigmoid(self.linear_ada_out(s_q)) * a_q
        return a_q
