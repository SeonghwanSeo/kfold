# started from code from https://github.com/jwohlwend/boltz, MIT License,

import torch
from torch import nn

try:
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
except ImportError:
    triangle_multiplicative_update = None

from kfold.utils.kernels import KernelBackend

from .linear import LinearNoBias
from .normalization import LayerNorm


@torch.compiler.disable
def cueq_triangluar_mult(
    x: torch.Tensor,
    direction: str,
    mask: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    eps: float,
):
    if triangle_multiplicative_update is None:
        raise ImportError(
            "cuequivariance_torch is not installed. "
            "Please install cuequivariance_torch to use the kernel implementation."
        )
    return triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


def _triton_compute_dtype(x: torch.Tensor) -> torch.dtype:
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


def _triton_cache_key(module: nn.Module, x: torch.Tensor, dtype: torch.dtype) -> tuple:
    parameters = (
        module.layernorm_in.weight,
        module.layernorm_in.bias,
        module.linear_p_in.weight,
        module.linear_g_in.weight,
        module.layernorm_out.weight,
        module.layernorm_out.bias,
        module.linear_p_out.weight,
        module.linear_g_out.weight,
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
def triton_triangluar_mult(
    module: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    if torch.is_grad_enabled():
        raise RuntimeError(
            "The Triton triangle multiplication backend is inference-only."
        )
    if not x.is_cuda:
        raise RuntimeError("The Triton triangle multiplication backend requires CUDA.")

    from kfold.utils.kernels.triton.triangle_multiplication import forward, precompute

    compute_dtype = _triton_compute_dtype(x)
    key = _triton_cache_key(module, x, compute_dtype)
    cached_key = getattr(module, "_triton_multiplication_cache_key", None)
    cached = getattr(module, "_triton_multiplication_cache", None)

    with torch.autocast(x.device.type, enabled=False):
        if cached is None or cached_key != key:

            def cast(parameter: torch.Tensor) -> torch.Tensor:
                return parameter.to(device=x.device, dtype=compute_dtype)

            cached = precompute(
                cast(module.layernorm_in.weight),
                cast(module.layernorm_in.bias),
                cast(module.linear_p_in.weight),
                cast(module.linear_g_in.weight),
                cast(module.layernorm_out.weight),
                cast(module.layernorm_out.bias),
                cast(module.linear_p_out.weight),
                cast(module.linear_g_out.weight),
            )
            module._triton_multiplication_cache = cached
            module._triton_multiplication_cache_key = key

        output = forward(
            x.to(compute_dtype),
            cached,
            direction=direction,
            mask=mask.bool(),
            eps=module.layernorm_in.eps,
        )
    return output.to(x.dtype)


class TriangleMultiplicationOutgoing(nn.Module):
    """TriangleMultiplicationOutgoing.
    See Section 3.4 Algorithm 12
    """

    direction = "outgoing"

    def __init__(
        self,
        dim: int = 128,
        backend: KernelBackend = KernelBackend.TORCH,
    ) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()
        self.backend = backend

        self.layernorm_in = LayerNorm(dim)
        self.linear_p_in = LinearNoBias(dim, 2 * dim, init="default")
        self.linear_g_in = LinearNoBias(dim, 2 * dim, init="gating")

        self.layernorm_out = LayerNorm(dim)
        self.linear_p_out = LinearNoBias(dim, dim, init="final")
        self.linear_g_out = LinearNoBias(dim, dim, init="gating")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)
        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if self.backend is KernelBackend.TRITON:
            return triton_triangluar_mult(self, x, mask, direction="outgoing")
        if self.backend is KernelBackend.CUEQUIVARIANCE:
            return cueq_triangluar_mult(
                x,
                direction=self.direction,
                mask=mask,
                norm_in_weight=self.layernorm_in.weight,
                norm_in_bias=self.layernorm_in.bias,
                p_in_weight=self.linear_p_in.weight,
                g_in_weight=self.linear_g_in.weight,
                norm_out_weight=self.layernorm_out.weight,
                norm_out_bias=self.layernorm_out.bias,
                p_out_weight=self.linear_p_out.weight,
                g_out_weight=self.linear_g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.layernorm_in(x)
        x_in = x
        x = self.linear_p_in(x) * self.linear_g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bikd,bjkd->bijd", a, b)

        # Output gating
        x = self.linear_p_out(self.layernorm_out(x)) * self.linear_g_out(x_in).sigmoid()

        return x


class TriangleMultiplicationIncoming(nn.Module):
    """TriangleMultiplicationIncoming.
    See Section 3.4 Algorithm 13
    """

    direction = "incoming"

    def __init__(
        self,
        dim: int = 128,
        backend: KernelBackend = KernelBackend.TORCH,
    ) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()
        self.backend = backend

        self.layernorm_in = LayerNorm(dim, eps=1e-5)
        self.linear_p_in = LinearNoBias(dim, 2 * dim, init="default")
        self.linear_g_in = LinearNoBias(dim, 2 * dim, init="gating")

        self.layernorm_out = LayerNorm(dim)
        self.linear_p_out = LinearNoBias(dim, dim, init="final")
        self.linear_g_out = LinearNoBias(dim, dim, init="gating")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)
        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if self.backend is KernelBackend.TRITON:
            return triton_triangluar_mult(self, x, mask, direction="incoming")
        if self.backend is KernelBackend.CUEQUIVARIANCE:
            return cueq_triangluar_mult(
                x,
                direction=self.direction,
                mask=mask,
                norm_in_weight=self.layernorm_in.weight,
                norm_in_bias=self.layernorm_in.bias,
                p_in_weight=self.linear_p_in.weight,
                g_in_weight=self.linear_g_in.weight,
                norm_out_weight=self.layernorm_out.weight,
                norm_out_bias=self.layernorm_out.bias,
                p_out_weight=self.linear_p_out.weight,
                g_out_weight=self.linear_g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.layernorm_in(x)
        x_in = x
        x = self.linear_p_in(x) * self.linear_g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bkid,bkjd->bijd", a, b)

        # Output gating
        x = self.linear_p_out(self.layernorm_out(x)) * self.linear_g_out(x_in).sigmoid()

        return x
