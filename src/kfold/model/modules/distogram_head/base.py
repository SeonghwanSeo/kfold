from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from kfold.utils.registry import DISTOGRAM_HEAD, BaseConfig


@DISTOGRAM_HEAD.register()
class BaseDistogramHead(torch.nn.Module, ABC):
    """Base class for distogram head modules.
    See Section 3 Algorithm 1 Main Inference Loop: Line [17]
    """

    @dataclass
    class Config(BaseConfig):
        """Base configuration class for distogram head modules.

        Parameters
        ----------
        channel_z : int
            The channel of pair representation.
        num_bins : int
            The number of distance bins.
        """

        channel_z: int = 128
        num_bins: int = 64

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
