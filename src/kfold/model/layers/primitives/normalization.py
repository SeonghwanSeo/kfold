import math

import torch
import torch.nn as nn

from .linear import Linear, LinearNoBias


# TODO (SeonghwanSeo): we may consider kernel fusion for LayerNorm
class LayerNorm(nn.Module):
    """Basic LayerNorm layer with learnable scale and offset.
    NOTE: This supports using bias only.
    """

    def __init__(
        self,
        normalized_shape: int,
        create_scale: bool = True,
        create_offset: bool = True,
        eps=1e-5,
    ):
        super().__init__()
        self.normalized_shape: int = normalized_shape
        self.eps: float = eps
        if create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.weight = None
        if create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.bias = None

    def forward(self, x) -> torch.Tensor:
        d = x.dtype
        if d is torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                weight = self.weight.to(dtype=d) if self.weight is not None else None
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                out = nn.functional.layer_norm(
                    input=x,
                    normalized_shape=(self.normalized_shape,),
                    weight=weight,
                    bias=bias,
                    eps=self.eps,
                )
        else:
            out = nn.functional.layer_norm(
                input=x,
                normalized_shape=(self.normalized_shape,),
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )
        return out


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
        self.layernorm_a = LayerNorm(channel_a, create_scale=False, create_offset=False)
        self.layernorm_s = LayerNorm(channel_s, create_scale=True, create_offset=False)
        self.linear_g = Linear(channel_s, channel_a, init="gating")
        self.linear_bias = LinearNoBias(channel_s, channel_a, init="final")
        self.sigmoid = nn.Sigmoid()

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """see Section 3.7 Algorithm 26 Adaptive LayerNorm"""
        # Line 1
        a = self.layernorm_a(a)
        # Line 2
        s = self.layernorm_s(s)
        # Line 3
        a = self.sigmoid(self.linear_g(s)) * a + self.linear_bias(s)
        return a


class GeoNorm(nn.Module):
    """GeoNorm geodesic residual update.

    This is based on GeoNorm (arXiv:2601.22095), which replaces the
    projection-style normalization + residual addition with a geodesic
    update on an ℓ2-sphere via the exponential map.

    Unlike LayerNorm/RMSNorm, GeoNorm is not a unary normalization op.
    It combines the current state `x` (residual stream) and an update
    direction `g` (e.g. Attention(x) or FFN(x)) into an updated state.

    The core update (Eq. 3-5 + Appendix D) is:
      1) Project `g` onto the tangent space at `x`.
      2) Compute an angle θ proportional to ||v|| / ||x|| (clamped).
      3) Move along the great-circle geodesic: x' = x cosθ + u ||x|| sinθ
         where u is the unit tangent direction.

    Notes for this codebase:
    - We compute in fp32 for numerical stability and cast back.
    - If ||x|| is near-zero (common for padded tokens), we fall back to
      standard residual addition (x + g) to avoid undefined geometry.
    """

    def __init__(
        self,
        clamp: float = math.pi / 4,
        decay: str = "harmonic",
        eps: float = 1e-8,
        zero_norm_fallback: str = "residual",  # "residual" | "keep"
    ) -> None:
        super().__init__()
        # Learnable affine on the (pre-decay) angle.
        # (Appendix D uses scalar scale/bias.)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

        self.clamp: float = float(clamp)
        self.decay: str = str(decay).lower()
        self.eps: float = float(eps)

        if zero_norm_fallback not in {"residual", "keep"}:
            raise ValueError(
                "zero_norm_fallback must be one of: 'residual', 'keep'"
            )
        self.zero_norm_fallback = zero_norm_fallback

    def _apply_decay(
        self,
        theta: torch.Tensor,
        layer_number: int,
        layer_total: int,
    ) -> torch.Tensor:
        """Apply layer-wise step-size decay schedule."""
        if self.decay in {"none", "no", ""}:
            return theta
        if self.decay == "harmonic":
            return theta / float(layer_number + 1)
        if self.decay == "sqrt":
            return theta / math.sqrt(layer_number + 1)
        if self.decay == "linear":
            # Linear decay to 0 as we approach the last layer.
            # Use max(1, layer_total) to avoid divide-by-zero.
            denom = float(max(1, layer_total))
            return theta * float(layer_total - layer_number) / denom

        raise ValueError(
            f"Unsupported GeoNorm decay schedule: {self.decay}. "
            "Use one of: none, harmonic, sqrt, linear."
        )

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        layer_number: int,
        layer_total: int,
    ) -> torch.Tensor:
        if x.shape != g.shape:
            raise ValueError(
                f"GeoNorm requires x and g to have the same shape, got "
                f"x={tuple(x.shape)} and g={tuple(g.shape)}"
            )

        # Compute in float32 for stability (esp. on fp16/bf16).
        x_dtype = x.dtype
        x_f = x.float()
        g_f = g.float()

        # ||x|| and (||x|| + eps)^2
        x_norm = torch.linalg.vector_norm(x_f, ord=2, dim=-1, keepdim=True)
        x_norm_sq = (x_norm + self.eps) ** 2

        # Project update direction onto tangent space at x:
        # v = g - <x, g> / ||x||^2 * x
        dot = (x_f * g_f).sum(dim=-1, keepdim=True)
        v = g_f - (dot / x_norm_sq) * x_f

        v_norm = torch.linalg.vector_norm(v, ord=2, dim=-1, keepdim=True) + self.eps
        unit_tangent = v / v_norm

        # Base angle (Appendix D): θ = ||v|| / ||x|| (clamped)
        theta = torch.clamp(v_norm / (x_norm + self.eps), max=self.clamp)

        # Learnable affine on theta before decay (Appendix D).
        theta = theta * self.scale + self.bias

        # Layer-wise decay schedule (paper suggests harmonic/sqrt/linear).
        theta = self._apply_decay(
            theta,
            layer_number=layer_number,
            layer_total=layer_total,
        )
        theta = torch.clamp(theta, max=self.clamp)

        # Geodesic update on the sphere (Eq. 3):
        # x' = x cosθ + unit_tangent * ||x|| sinθ
        out = x_f * torch.cos(theta) + unit_tangent * x_norm * torch.sin(theta)

        # If the residual stream is near-zero, the tangent space is ill-defined.
        # This happens frequently on padded tokens / empty pair slots.
        # Fall back to a simple residual update or keep x unchanged.
        near_zero = (x_norm < 1e-6)
        if near_zero.any():
            if self.zero_norm_fallback == "residual":
                out = torch.where(near_zero, x_f + g_f, out)
            elif self.zero_norm_fallback == "keep":
                out = torch.where(near_zero, x_f, out)

        return out.to(dtype=x_dtype)
