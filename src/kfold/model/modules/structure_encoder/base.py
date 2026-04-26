from abc import ABC, abstractmethod

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import STRUCTURE_ENCODER, BaseConfig


@STRUCTURE_ENCODER.register()
class BaseStructureEncoder(torch.nn.Module, ABC):
    class Config(BaseConfig): ...

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of structure representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations.
        """
