# started from code from https://github.com/jwohlwend/boltz, MIT License,
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

import math
from functools import partial

import torch
import torch.nn as nn

try:
    from cuequivariance_torch.primitives.triangle import triangle_attention
except ImportError:
    triangle_attention = None

from .linear import LinearNoBias
from .normalization import LayerNorm
from .utils import chunk_layer, flatten_final_dims, permute_final_dims


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    d = t.dtype
    if d is torch.bfloat16:
        with torch.autocast("cuda", enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


@torch.compiler.disable
def kernel_triangular_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tri_bias: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if triangle_attention is None:
        raise ImportError(
            "cuequivariance_torch is not installed. "
            "Please install cuequivariance_torch to use the kernel implementation."
        )
    return triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        inf: float = 1e9,
    ):
        """Initialize the attention layer.

        Parameters
        ----------
        c_q : int
            Input dimension of query data
        c_k : int
            Input dimension of key data
        c_v : int
            Input dimension of value data
        c_hidden : int
            Per-head hidden dimension
        no_heads : int
            Number of attention heads
        gating : bool, default=True
            Whether the output should be gated using query data

        """
        super().__init__()

        self.c_q: int = c_q
        self.c_k: int = c_k
        self.c_v: int = c_v
        self.c_hidden: int = c_hidden
        self.no_heads: int = no_heads
        self.gating: bool = gating
        self.inf: float = inf

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = LinearNoBias(
            self.c_q, self.c_hidden * self.no_heads, init="default"
        )
        self.linear_k = LinearNoBias(
            self.c_k, self.c_hidden * self.no_heads, init="default"
        )
        self.linear_v = LinearNoBias(
            self.c_v, self.c_hidden * self.no_heads, init="default"
        )
        self.linear_o = LinearNoBias(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = LinearNoBias(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        tri_bias: torch.Tensor,
        mask: torch.Tensor,
        use_kernels: bool = False,
    ) -> torch.Tensor:
        """Compute attention.

        Parameters
        ----------
        q_x : torch.Tensor
            [*, Q, C_q] query data
        kv_x : torch.Tensor
            [*, K, C_k] key data
        tri_bias : torch.Tensor
            [*, H, Q, K] triangular bias
        mask : torch.Tensor
            [*, Q, K] mask
        use_kernels : bool, default=False
            Whether to use optimized CUDA kernels

        Returns
        -------
            [*, Q, C_q] attention update

        """
        # Attention kernel applies scaling internally
        q, k, v = self._prep_qkv(
            q_x,
            kv_x,
            apply_scale=not use_kernels,
        )

        if use_kernels:
            scale = 1.0 / math.sqrt(self.c_hidden)
            o = kernel_triangular_attn(
                q,
                k,
                v,
                tri_bias=tri_bias,
                mask=mask.bool(),
                scale=scale,
            )
            o = o.transpose(-2, -3)
        else:
            if mask.dtype == torch.bool:
                mask_bias = -self.inf * (~mask).to(q.dtype)
            else:
                mask_bias = self.inf * (mask - 1)
            biases = [mask_bias, tri_bias]
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


class TriangleAttention(nn.Module):
    """See Section 3.4 Algorithm 14 in the AlphaFold3 paper."""

    def __init__(
        self,
        c_in: int,
        no_heads: int,
        starting: bool,
        inf: float = 1e9,
    ) -> None:
        super().__init__()
        assert c_in % no_heads == 0, (
            f"c_in ({c_in}) must be divisible by no_heads ({no_heads})"
        )

        self.c_in: int = c_in
        self.c_hidden: int = c_in // no_heads
        self.no_heads: int = no_heads
        self.starting: bool = starting
        self.inf: float = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = LinearNoBias(c_in, self.no_heads, init="default")

        self.mha = MultiHeadAttention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        tri_bias: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
        use_kernels: bool = False,
    ) -> torch.Tensor:
        """Compute triangle attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [*, I, J, C_in]
        biases : list[torch.Tensor]
            List of bias tensors of shape [*, H, I, J]
        chunk_size : int
            Size of chunks for memory efficient computation
        use_kernels : bool, default=False
            Whether to use optimized CUDA kernels

        Returns
        -------
        torch.Tensor
            Output tensor of shape [*, I, J, C_in]

        """
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "tri_bias": tri_bias,
            "mask": mask,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_kernels=use_kernels,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_kernels: bool = False,
    ) -> torch.Tensor:
        """Compute triangle attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [*, I, J, C_in]
        mask : torch.Tensor, optional
            Attention mask of shape [*, I, J]
        chunk_size : int, optional
            Size of chunks for memory efficient computation
        use_kernels : bool, default=False
            Whether to use optimized CUDA kernels

        Returns
        -------
        torch.Tensor
            Output tensor of shape [*, I, J, C_in]

        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask = mask[..., :, None, None, :]

        with torch.autocast(x.device.type, dtype=torch.float32, enabled=use_kernels):
            # NOTE: casting to float32 for cuequiv kernels.
            triangle_bias = self.linear(x)

            # [*, H, I, J]
            triangle_bias = permute_final_dims(triangle_bias, (2, 0, 1))

            # [*, 1, H, I, J]
            triangle_bias = triangle_bias.unsqueeze(-4)

        if chunk_size is not None and not use_kernels:
            x = self._chunk(
                x,
                triangle_bias,
                mask,
                chunk_size,
                use_kernels=use_kernels,
            )
        else:
            x = self.mha(
                x,
                x,
                triangle_bias,
                mask,
                use_kernels=use_kernels,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements Algorithm 14
class TriangleAttentionStartingNode(TriangleAttention):
    """See Section 3.4 Algorithm 14 in the AlphaFold3 paper."""

    def __init__(self, c_in: int, no_heads: int, inf: float = 1e9) -> None:
        super().__init__(c_in, no_heads, starting=True, inf=inf)


class TriangleAttentionEndingNode(TriangleAttention):
    """See Section 3.4 Algorithm 15 in the AlphaFold3 paper."""

    def __init__(self, c_in: int, no_heads: int, inf: float = 1e9) -> None:
        super().__init__(c_in, no_heads, starting=False, inf=inf)
