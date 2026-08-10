from dataclasses import dataclass

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives import LinearNoBias
from kfold.model.primitives.utils import permute_final_dims
from kfold.utils.config import configurable


@configurable
class DistogramHead(torch.nn.Module):
    @dataclass(kw_only=True)
    class Config:
        """Base configuration class for distogram head modules.

        Parameters
        ----------
        channel_z : int
            The channel of pair representation.
        num_bins : int
            The number of distance bins.
        """

        channel_z: int = 256
        num_bins: int = 64
        min_dist: float = 2.0
        max_dist: float = 22.0

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_bins: int = cfg.num_bins
        bin_size = (cfg.max_dist - cfg.min_dist) / cfg.num_bins
        first_bin = cfg.min_dist + bin_size
        last_bin = cfg.max_dist - bin_size
        bin_boundaries = torch.linspace(first_bin, last_bin, cfg.num_bins - 1)
        self.register_buffer(
            "bin_boundaries",
            bin_boundaries,
            persistent=False,
        )
        self.contact_bin = int((bin_boundaries < 8.0).sum().item()) - 1

        self.linear = LinearNoBias(cfg.channel_z, cfg.num_bins, init="final")

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of distogram head module.

        Parameters
        ----------
        z : torch.Tensor
            Tensor of shape (*, N, N, c_z) containing pair feature

        Returns
        -------
        distogram_out: dict[str, torch.Tensor]
            Distogram logits and distance-bin boundaries.
        """
        logits = self.linear(z)
        # symmetrize logits
        logits = logits + permute_final_dims(logits, (1, 0, 2))
        return {
            "logits": logits,
            "bin_boundaries": self.bin_boundaries,
        }

    def forward_inference(
        self,
        f_input: FoldingInput,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of distogram head module.

        Parameters
        ----------
        z : torch.Tensor
            Tensor of shape (*, N, N, c_z) containing pair feature

        Returns
        -------
        distogram_out: dict[str, torch.Tensor]
            Distogram logits, distance-bin boundaries, and contact probabilities.
        """
        mask = f_input.token.pad_mask  # [B, N]
        pair_mask = mask[..., None, :] & mask[..., :, None]  # [B, N, N]
        distogram_out = self(z)
        logits = distogram_out["logits"]  # [B, N, N, num_bins]
        p = torch.softmax(logits, dim=-1)  # [B, N, N, num_bins]
        p.masked_fill_(~pair_mask[..., None], 0.0)
        p_contact = p[..., : self.contact_bin + 1].sum(dim=-1)
        return distogram_out | {"prob_contact": p_contact}
