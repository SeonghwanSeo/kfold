import torch

from kfold.data.model_input import FoldingInput


class DistogramLoss(torch.nn.Module):
    def __init__(
        self,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
    ) -> None:
        super().__init__()
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.num_bins: int = num_bins

        bin_size: float = (max_dist - min_dist) / num_bins
        first_bin = min_dist + bin_size  # =2.3125
        last_bin = max_dist - bin_size  # =21.6875

        boundaries = torch.linspace(first_bin, last_bin, num_bins - 1)  # [num_bins - 1]
        self.register_buffer("boundaries", boundaries, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the  distogram loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, Lt, Lt, num_bins) containing distogram logits.
        f_input : FoldingInput
            The input features containing the target distogram and masks.

        Returns
        -------
        disto_loss : torch.Tensor
            The computed distogram loss of shape (B,).
        """

        with torch.autocast("cuda", enabled=False), torch.no_grad():
            boundaries: torch.Tensor = self.boundaries  # [num_bins - 1] # type: ignore
            pdist_disto = torch.cdist(
                f_input.token.disto_coords,
                f_input.token.disto_coords,
            )  # [B, Lt, Lt]
            target_distogram = (pdist_disto.unsqueeze(-1) > boundaries).sum(dim=-1).long()

        # Compute the distogram loss
        B, L, L = target_distogram.shape
        disto_loss = torch.nn.functional.cross_entropy(
            logits.view(B * L * L, self.num_bins),
            target_distogram.view(B * L * L),
            reduction="none",
        ).view(B, L, L)

        # Mask out invalid distogram
        mask = f_input.token.disto_mask  # [B, Lt]
        pair_mask = mask[:, None, :] & mask[:, :, None]  # [B, Lt, Lt]
        pair_mask.diagonal(dim1=-2, dim2=-1).zero_()  # zero out diagonal
        disto_loss = disto_loss * pair_mask  # [B, Lt, Lt]

        # Compute mean loss
        disto_loss = disto_loss.sum((-1, -2)) / pair_mask.sum((-1, -2)).clamp(1)  # [B,]
        return disto_loss
