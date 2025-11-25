import torch

from kfold.model.layers.primitives import LinearNoBias
from kfold.utils.registry import DISTOGRAM_HEAD

from .base import BaseDistogramHead


@DISTOGRAM_HEAD.register()
class DistogramHead(BaseDistogramHead):
    def __init__(self, cfg: BaseDistogramHead.Config):
        super().__init__(cfg)
        self.linear = LinearNoBias(cfg.channel_z, cfg.num_bins, init="final")

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
