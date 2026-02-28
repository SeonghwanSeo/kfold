from abc import ABC, abstractmethod

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import SEQUENCE_ENCODER, BaseConfig


@SEQUENCE_ENCODER.register()
class BaseSequenceEncoder(torch.nn.Module, ABC):
    class Config(BaseConfig):
        return_attn: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.return_attn: bool = cfg.return_attn

    @property
    def d_attn(self) -> int:
        """Dimension of attention weights returned by the sequence encoder."""
        raise NotImplementedError("Subclasses must implement d_attn property.")

    @abstractmethod
    def forward(self, f_input: FoldingInput) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations.
        attention: torch.Tensor | None
            Tensor of shape (B, Ntoken, Ntoken, N*H) containing attention weights,
            where N is number of layers and H is number of heads.
        """
