# started from code from https://github.com/jwohlwend/boltz, MIT License,
import torch
from torch import nn

try:
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
except ImportError:
    triangle_multiplicative_update = None

from .primitives import LayerNorm, LinearNoBias


@torch.compiler.disable
def kernel_triangular_mult(
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


class TriangleMultiplicationOutgoing(nn.Module):
    """TriangleMultiplicationOutgoing.
    See Section 3.4 Algorithm 12
    """

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = LayerNorm(dim, eps=1e-5)
        self.p_in = LinearNoBias(dim, 2 * dim, init="default")
        self.g_in = LinearNoBias(dim, 2 * dim, init="gating")

        self.norm_out = LayerNorm(dim)
        self.p_out = LinearNoBias(dim, dim, init="final")
        self.g_out = LinearNoBias(dim, dim, init="gating")

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, use_kernels: bool = False
    ) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)
        use_kernels: bool
            Whether to use the kernel

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if use_kernels:
            return kernel_triangular_mult(
                x,
                direction="outgoing",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bikd,bjkd->bijd", a, b)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x


class TriangleMultiplicationIncoming(nn.Module):
    """TriangleMultiplicationIncoming.
    See Section 3.4 Algorithm 13
    """

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = LayerNorm(dim, eps=1e-5)
        self.p_in = LinearNoBias(dim, 2 * dim, init="default")
        self.g_in = LinearNoBias(dim, 2 * dim, init="gating")

        self.norm_out = LayerNorm(dim)
        self.p_out = LinearNoBias(dim, dim, init="final")
        self.g_out = LinearNoBias(dim, dim, init="gating")

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, use_kernels: bool = False
    ) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)
        use_kernels: bool
            Whether to use the kernel

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if use_kernels:
            return kernel_triangular_mult(
                x,
                direction="incoming",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bkid,bkjd->bijd", a, b)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x
