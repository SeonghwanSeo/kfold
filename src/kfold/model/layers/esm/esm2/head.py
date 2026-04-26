import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sqrt_2 = math.sqrt(2.0)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / sqrt_2))


class RobertaLMHead(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int, weight: torch.Tensor):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x
