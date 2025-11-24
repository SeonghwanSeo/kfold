import torch
import torch.nn as nn

from .linear import Linear, LinearNoBias


# TODO (SeonghwanSeo): we may consider kernel fusion for LayerNorm
class LayerNorm(nn.LayerNorm): ...


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
        self.layernorm_a = LayerNorm(channel_a, elementwise_affine=False, bias=False)
        self.layernorm_s = LayerNorm(channel_s, elementwise_affine=True, bias=False)
        self.linear_g = Linear(channel_s, channel_a, init="final")
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
