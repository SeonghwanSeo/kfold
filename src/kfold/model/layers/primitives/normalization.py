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
