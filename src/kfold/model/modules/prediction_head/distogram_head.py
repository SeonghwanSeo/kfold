import torch

from kfold.model.layers.primitives import LinearNoBias
from kfold.model.layers.primitives.utils import permute_final_dims
from kfold.utils.registry import DISTOGRAM_HEAD, BaseConfig


@DISTOGRAM_HEAD.register()
class DistogramHead(torch.nn.Module):
    class Config(BaseConfig):
        """Base configuration class for distogram head modules.

        Parameters
        ----------
        channel_z : int
            The channel of pair representation.
        num_bins : int
            The number of distance bins.
        """

        channel_z: int = 128
        num_bins: int = 64
        min_dist: float = 2.0
        max_dist: float = 22.0

    def __init__(self, cfg: Config):
        super().__init__()
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
