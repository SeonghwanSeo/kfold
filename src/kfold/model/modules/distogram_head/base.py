from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from kfold.utils.registry import DISTOGRAM_HEAD, BaseConfig


@dataclass
class BaseDistogramHeadConfig(BaseConfig):
    """Base configuration class for distogram head modules."""

    c_z: int = 128
    num_bins: int = 64


@DISTOGRAM_HEAD.register(config_cls=BaseDistogramHeadConfig)
class BaseDistogramHead(torch.nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of distogram head module.

        Parameters
        ----------
        z : torch.Tensor
            Tensor of shape (B, N, N, c_z) containing pair feature.

        Returns
        -------
        logits: torch.Tensor
            Tensor of shape (B, N, N, num_bins) containing distogram logits.
        """


@DISTOGRAM_HEAD.register(config_cls=BaseDistogramHeadConfig)
class DistogramHead(BaseDistogramHead):
    def __init__(self, cfg: BaseDistogramHeadConfig):
        super().__init__(cfg)
        self.linear = torch.nn.Linear(cfg.c_z, cfg.num_bins)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of distogram head module.

        Parameters
        ----------
        z : torch.Tensor
            Tensor of shape (B, N, N, c_z) containing pair feature

        Returns
        -------
        logits: torch.Tensor
            Tensor of shape (B, N, N, num_bins) containing distogram logits.
        """
        logits = self.linear(z)
        # symmetrize logits
        logits = logits + logits.permute(0, 2, 1, 3)
        return logits
