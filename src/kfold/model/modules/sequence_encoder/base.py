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
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor of shape (B, L) containing sequence tokens.
        attn_mask: torch.Tensor
            Attention mask of shape (B, L), where True indicates valid tokens.
        pos_id: torch.Tensor
            Position ids of shape (B, L) for rotary positional embeddings.

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, L, D) containing sequence representations.
        attention: torch.Tensor | None
            Tensor of shape (B, N, H, L, L) containing attention weights,
            where N is number of layers and H is number of heads.
        """
