import torch
import torch.nn as nn

from .linear import LinearNoBias


class SwiGLU(nn.Module):
    """SiLU Gated Linear Unit (SwiGLU) activation function."""

    def __init__(self, channel_in: int, channel_out: int, init: str = "relu"):
        super().__init__()
        self.linear = LinearNoBias(channel_in, channel_out * 2, init=init)
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return self.swish(a) * b
