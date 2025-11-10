from abc import ABC, abstractmethod

import torch

from kfold.utils.registry import SEQUENCE_ENCODER


@SEQUENCE_ENCODER.register()
class BaseSequenceEncoder(torch.nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_id: torch.Tensor,
        chain_id: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        sequence_tokens : torch.Tensor
            Tensor of shape (B, L) containing sequence tokens.
        sequence_ids : torch.Tensor
            Tensor of shape (B, L) containing sequence id.
        chain_ids : torch.Tensor
            Tensor of shape (B, L) containing chain ids.
        return_attention : bool, optional
            Whether to return attention weights. Default is False.

        Returns
        -------
        x: torch.Tensor
            Tensor of shape (B, L, D) containing sequence feature.
        attention: torch.Tensor | None
            Tensor of shape (B, N, H, L, L) containing attention weights,
            where N is number of layers and H is number of heads.
        """
