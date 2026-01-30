"""Base class and config for interaction prediction heads."""

from abc import ABC, abstractmethod

import torch

import kfold.constants as C
from kfold.utils.registry import INTERACTION_HEAD, BaseConfig


@INTERACTION_HEAD.register()
class BaseInteractionHead(torch.nn.Module, ABC):
    """Base class for interaction head modules."""

    class Config(BaseConfig):
        """Configuration for interaction heads.

        Attributes
        ----------
        channel_z : int
            Input pair representation channel dimension.
        num_pair_types : int
            Number of pair interaction types to predict.
        """

        channel_z: int = 128
        num_pair_types: int = C.NUM_PAIR_INTERACTION_TYPES

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict interaction logits from pair representations.

        Parameters
        ----------
        z : torch.Tensor
            Pair representation tensor of shape [B, L, L, channel_z].

        Returns
        -------
        torch.Tensor
            Interaction logits of shape [B, L, L, num_pair_types].
        """
