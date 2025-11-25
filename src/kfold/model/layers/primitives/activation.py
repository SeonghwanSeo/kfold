import torch
import torch.nn as nn

from .linear import LinearNoBias


class SwiGLU(nn.Module):
    """SiLU Gated Linear Unit (SwiGLU) activation function."""

    def __init__(self, channel_in: int, channel_out: int):
        super().__init__()
        self.linear_a = LinearNoBias(channel_in, channel_out, init="relu")
        self.linear_b = LinearNoBias(channel_in, channel_out, init="relu")
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.linear_a(x)
        b = self.linear_b(x)
        return self.swish(a) * b
