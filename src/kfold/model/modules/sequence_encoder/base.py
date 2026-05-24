from abc import ABC, abstractmethod

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import SEQUENCE_ENCODER, BaseConfig


@SEQUENCE_ENCODER.register()
class BaseSequenceEncoder(torch.nn.Module, ABC):
    class Config(BaseConfig): ...

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def n_heads(self) -> int: ...

    @property
    @abstractmethod
    def d_model(self) -> int: ...

    @abstractmethod
    def forward(self, f_input: FoldingInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations,
            where D is the model dimension.
        attention: torch.Tensor
            Tensor of shape (B, Ntoken, Ntoken, N, H) containing attention weights,
            where N is number of layers and H is number of heads.
        """
