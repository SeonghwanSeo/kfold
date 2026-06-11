"""Basic layers without initialization"""

import torch.nn as nn

enable_init: bool = False


class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        if enable_init:
            super().reset_parameters()


class LayerNorm(nn.LayerNorm):
    def reset_parameters(self) -> None:
        if enable_init:
            super().reset_parameters()


class Embedding(nn.Embedding):
    def reset_parameters(self) -> None:
        if enable_init:
            super().reset_parameters()
