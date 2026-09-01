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

import torch
import torch.nn as nn

try:
    from cuequivariance_torch.primitives.triangle import (
        triangle_attention as cueq_triangle_attention,
    )
except ImportError:
    cueq_triangle_attention = None

from kfold.utils.kernels import KernelBackend

from .linear import LinearNoBias
from .normalization import LayerNorm
from .utils import flatten_final_dims, permute_final_dims


def _triton_compute_dtype(x: torch.Tensor) -> torch.dtype:
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


def _triton_cache_key(module: nn.Module, x: torch.Tensor, dtype: torch.dtype) -> tuple:
    parameters = (
        module.layer_norm.weight,
        module.layer_norm.bias,
        module.mha.linear_q.weight,
        module.mha.linear_k.weight,
        module.mha.linear_v.weight,
        module.linear.weight,
        module.mha.linear_g.weight,
        module.mha.linear_o.weight,
    )
    return (x.device.type, x.device.index, dtype) + tuple(
        value
        for parameter in parameters
        for value in (
            parameter.data_ptr(),
            None if torch.is_inference(parameter) else parameter._version,
        )
    )


@torch.compiler.disable
def triton_triangular_attn(
    module: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if torch.is_grad_enabled():
        raise RuntimeError("The Triton triangle attention backend is inference-only.")
    if not x.is_cuda:
        raise RuntimeError("The Triton triangle attention backend requires CUDA.")

    from kfold.utils.kernels.triton.triangle_attention import forward, precompute

    compute_dtype = _triton_compute_dtype(x)
    key = _triton_cache_key(module, x, compute_dtype)
    cached_key = getattr(module, "_triton_attention_cache_key", None)
    cached = getattr(module, "_triton_attention_cache", None)

    with torch.autocast(x.device.type, enabled=False):
        if cached is None or cached_key != key:

            def cast(parameter: torch.Tensor) -> torch.Tensor:
                return parameter.to(device=x.device, dtype=compute_dtype)

            cached = precompute(
                cast(module.layer_norm.weight),
                cast(module.layer_norm.bias),
                cast(module.mha.linear_q.weight).t().contiguous(),
                cast(module.mha.linear_k.weight).t().contiguous(),
                cast(module.mha.linear_v.weight).t().contiguous(),
                cast(module.linear.weight).t().contiguous(),
                torch.zeros(
                    module.no_heads,
                    device=x.device,
                    dtype=compute_dtype,
                ),
                module.no_heads,
                module.c_hidden,
            )
            cached["gate_weight"] = cast(module.mha.linear_g.weight)
            cached["output_weight"] = cast(module.mha.linear_o.weight)
            module._triton_attention_cache = cached
            module._triton_attention_cache_key = key

        length, channel = x.shape[-2:]
        batch_shape = x.shape[:-3]
        flat_x = x.to(compute_dtype).reshape(-1, length, length, channel)
        flat_mask = torch.broadcast_to(mask.bool(), x.shape[:-1]).reshape(
            -1, length, length
        )
        output = forward(
            flat_x,
            cached,
            mask=flat_mask,
            scale=1.0 / math.sqrt(module.c_hidden),
            eps=module.layer_norm.eps,
            W_proj_g=cached["gate_weight"],
            B_proj_g=None,
            W_proj_o=cached["output_weight"],
            B_proj_o=None,
        )
        output = output.reshape(batch_shape + (length, length, channel))
    return output.to(x.dtype)


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

    a = a.softmax(dim=-1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


@torch.compiler.disable
def cueq_triangular_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if cueq_triangle_attention is None:
        raise ImportError(
            "cuequivariance_torch is not installed. "
            "Please install cuequivariance_torch to use the kernel implementation."
        )
    # cuEquivariance 0.10 raises its eager Torch-fallback threshold to 200 for
    # head dimensions below 32, regardless of CUEQ_TRIATTN_FALLBACK_THRESHOLD.
    # Requesting auxiliary outputs forces the supported native kernel path; the
    # benchmark preflight turns any remaining unsupported fallback into an error.
    output, _, _ = cueq_triangle_attention(
        q,
        k,
        v,
        bias.float(),
        mask=mask,
        scale=scale,
        return_aux=True,
    )
    return output


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
        backend: KernelBackend = KernelBackend.TORCH,
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
        if backend not in (KernelBackend.TORCH, KernelBackend.CUEQUIVARIANCE):
            raise ValueError(f"Unsupported multi-head attention backend: {backend!r}")
        self.backend = backend

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
        Returns
        -------
            [*, Q, C_q] attention update

        """
        # Attention kernel applies scaling internally
        q, k, v = self._prep_qkv(
            q_x,
            kv_x,
            apply_scale=self.backend is not KernelBackend.CUEQUIVARIANCE,
        )

        if self.backend is KernelBackend.CUEQUIVARIANCE:
            scale = 1.0 / math.sqrt(self.c_hidden)
            o = cueq_triangular_attn(
                q,
                k,
                v,
                bias=tri_bias,
                mask=mask,
                scale=scale,
            )
            o = o.transpose(-2, -3)
        else:
            mask_bias = -self.inf * (~mask.bool()).to(q.dtype)
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
        backend: KernelBackend = KernelBackend.TORCH,
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
        self.backend = backend

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = LinearNoBias(c_in, self.no_heads, init="default")

        attention_backend = (
            KernelBackend.CUEQUIVARIANCE
            if backend is KernelBackend.CUEQUIVARIANCE
            else KernelBackend.TORCH
        )
        self.mha = MultiHeadAttention(
            self.c_in,
            self.c_in,
            self.c_in,
            self.c_hidden,
            self.no_heads,
            backend=attention_backend,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute triangle attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [*, I, J, C_in]
        mask : torch.Tensor, optional
            Attention mask of shape [*, I, J]
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

        if self.backend is KernelBackend.TRITON:
            x = triton_triangular_attn(self, x, mask)
            if not self.starting:
                x = x.transpose(-2, -3)
            return x

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask = mask[..., :, None, None, :]

        triangle_bias = self.linear(x)

        # [*, H, I, J]
        triangle_bias = permute_final_dims(triangle_bias, (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        x = self.mha(
            x,
            x,
            triangle_bias,
            mask,
        )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements Algorithm 14
class TriangleAttentionStartingNode(TriangleAttention):
    """See Section 3.4 Algorithm 14 in the AlphaFold3 paper."""

    def __init__(
        self,
        c_in: int,
        no_heads: int,
        inf: float = 1e9,
        backend: KernelBackend = KernelBackend.TORCH,
    ) -> None:
        super().__init__(c_in, no_heads, starting=True, inf=inf, backend=backend)


class TriangleAttentionEndingNode(TriangleAttention):
    """See Section 3.4 Algorithm 15 in the AlphaFold3 paper."""

    def __init__(
        self,
        c_in: int,
        no_heads: int,
        inf: float = 1e9,
        backend: KernelBackend = KernelBackend.TORCH,
    ) -> None:
        super().__init__(c_in, no_heads, starting=False, inf=inf, backend=backend)
