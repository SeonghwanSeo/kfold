import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import initialize as init

LinearNoBias = partial(nn.Linear, bias=False)


def add(x: torch.Tensor, y: float | torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.add_(y)
    else:
        return x + y


def sub(x: torch.Tensor, y: float | torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.sub_(y)
    else:
        return x - y


def div(x: torch.Tensor, y: float | torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.div_(y)
    else:
        return x / y


def mul(x: torch.Tensor, y: float | torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.mul_(y)
    else:
        return x * y


class Transition(nn.Module):
    """Perform a two-layer MLP.
    See Section 3.3 Algorithm 11 Transition layer
    """

    def __init__(
        self,
        channel: int,
        expansion_factor: int,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        channel: int
            The dimension of the input
        expansion_factor: int
            The expansion factor for the hidden dimension

        """
        super().__init__()

        model_dim = channel * expansion_factor
        self.model_dim = model_dim
        self.layernorm = nn.LayerNorm(channel, eps=1e-5)
        self.linear_no_bias_a = LinearNoBias(channel, model_dim)
        self.linear_no_bias_b = LinearNoBias(channel, model_dim)
        self.linear_no_bias_out = LinearNoBias(model_dim, channel)

        init.bias_init_one_(self.layernorm.weight)
        init.bias_init_zero_(self.layernorm.bias)

        init.lecun_normal_init_(self.linear_no_bias_a.weight)
        init.lecun_normal_init_(self.linear_no_bias_b.weight)
        init.final_init_(self.linear_no_bias_out.weight)

    def forward(self, x: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        """Perform a forward pass.
        See Section 3.3 Algorithm 11 Transition layer

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)
        chunk_size: Optional[int]
            The chunk size for memory-efficient computation, default None

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        # Line 1
        x = self.layernorm(x)

        if chunk_size is None or self.training:
            # Line 2
            a = self.linear_no_bias_a(x)

            # Line 3
            b = self.linear_no_bias_b(x)

            # Line 4
            x = self.linear_no_bias_out(F.silu(a) * b)
            return x
        else:
            # Compute in chunks
            for i in range(0, self.model_dim, chunk_size):
                lin_a_slice = self.linear_no_bias_a.weight[i : i + chunk_size, :]
                lin_b_slice = self.linear_no_bias_b.weight[i : i + chunk_size, :]
                lin_o_slice = self.linear_no_bias_out.weight[:, i : i + chunk_size]
                x_chunk = F.silu(x @ lin_a_slice.T) * (x @ lin_b_slice.T)
                if i == 0:
                    x_out = x_chunk @ lin_o_slice.T
                else:
                    x_out = x_out + x_chunk @ lin_o_slice.T  # type: ignore
            return x_out  # type: ignore


class AdaLN(nn.Module):
    """Adaptive Layer Normalization
    See Section 3.7 Algorithm 26 Adaptive LayerNorm
    """

    def __init__(self, channel_a: int, channel_s: int):
        """Initialize the adaptive layer normalization.

        Parameters
        ----------
        channel_a : int
            The input dimension.
        channel_s : int
            The single condition dimension.

        """
        super().__init__()
        self.a_norm = nn.LayerNorm(channel_a, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(channel_s, bias=False)
        self.s_scale = nn.Linear(channel_s, channel_a)
        self.s_bias = LinearNoBias(channel_s, channel_a)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """see Section 3.7 Algorithm 26 Adaptive LayerNorm"""
        # Line 1
        a = self.a_norm(a)
        # Line 2
        s = self.s_norm(s)
        # Line 3
        a = torch.sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float | bool | None = None,
    use_high_precision: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    dtype = query.dtype if not use_high_precision else torch.float32

    if scale is True:
        scale = math.sqrt(query.shape[-1])
    if scale is not None:
        query = div(query, scale, inplace=inplace)

    with torch.autocast("cuda", dtype=dtype):
        # Compute attention weights
        attn = torch.einsum("...qc,...kc->...qk", query, key)

        # Add attention bias
        if bias is not None:
            attn = add(attn, bias, inplace=inplace)

        # Softmax normalization
        with torch.autocast("cuda", dtype=torch.float32):
            attn = attn.softmax(dim=-1)

        # Compute output
    out = torch.einsum("...qk,...kc->...qc", attn.to(value.dtype), value)

    return out
