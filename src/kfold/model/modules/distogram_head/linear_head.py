import torch

from kfold.model.layers.primitives import LinearNoBias
from kfold.model.layers.primitives.utils import permute_final_dims
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
            Tensor of shape (*, N, N, c_z) containing pair feature

        Returns
        -------
        logits: torch.Tensor
            Tensor of shape (*, N, N, num_bins) containing distogram logits.
        """
        logits = self.linear(z)
        # symmetrize logits
        logits = logits + permute_final_dims(logits, (1, 0, 2))
        return logits
