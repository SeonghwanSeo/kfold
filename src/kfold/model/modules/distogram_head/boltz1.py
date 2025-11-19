import torch

from kfold.utils.registry import DISTOGRAM_HEAD

from .base import BaseDistogramHead


@DISTOGRAM_HEAD.register()
class Boltz1DistogramHead(BaseDistogramHead):
    def __init__(self, cfg: BaseDistogramHead.Config):
        super().__init__(cfg)
        self.distogram = torch.nn.Linear(cfg.channel_z, cfg.num_bins)

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
        z = z + z.transpose(1, 2)
        return self.distogram(z)
