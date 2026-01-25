from abc import ABC, abstractmethod

import torch

import kfold.constants as C
from kfold.utils.registry import INTERACTION_HEAD, BaseConfig


@INTERACTION_HEAD.register()
class BaseInteractionHead(torch.nn.Module, ABC):
    """Base class for interaction head modules."""

    class Config(BaseConfig):
        channel_z: int = 128
        num_pair_types: int = C.NUM_PAIR_INTERACTION_TYPES

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of interaction head module."""
