import torch

from kfold.data.types.model_input import FoldingInput
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
        self.num_bins: int = cfg.num_bins
        self.min_dist: float = cfg.min_dist
        self.max_dist: float = cfg.max_dist

        min_d, max_d = self.min_dist, self.max_dist
        bin_size: float = (max_d - min_d) / self.num_bins  # =0.3125
        self.first_bin: float = min_d + bin_size  # =2.3125
        self.last_bin: float = max_d - bin_size  # =21.6875
        self.contact_bin = int((8.0 - self.first_bin) / bin_size)  # = 18

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
        distogram: torch.Tensor
            Tensor of shape (*, N, N, num_bins) containing distogram logits.
        p_contact: torch.Tensor
            Tensor of shape (*, N, N) containing contact probabilities,
            i.e., the probability that the distance between two residues is
            less than 8Å.
        """
        mask = f_input.token.pad_mask  # [B, N]
        pair_mask = mask[..., None, :] & mask[..., :, None]  # [B, N, N]
        distogram = self(z)  # [B, N, N, num_bins]
        p = torch.softmax(distogram, dim=-1)  # [B, N, N, num_bins]
        p.masked_fill_(~pair_mask[..., None], 0.0)
        p_contact = p[..., : self.contact_bin + 1].sum(dim=-1)
        return {"distogram": distogram, "prob_contact": p_contact}
