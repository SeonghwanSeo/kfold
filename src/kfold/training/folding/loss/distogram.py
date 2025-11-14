import torch

from kfold.data.model_input import FoldingInput


class DistogramLoss(torch.nn.Module):
    def __init__(self, min_dist: float, max_dist: float, num_bins: int) -> None:
        super().__init__()
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.num_bins: int = num_bins

        boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)  # [num_bins - 1]
        self.register_buffer("boundaries", boundaries)

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
            The computed distogram loss of shape.
        """

        dist_repr_atoms = torch.cdist(
            f_input.token.disto_coords,
            f_input.token.disto_coords,
        )  # [B, Lt, Lt]

        target_distogram = (
            (dist_repr_atoms.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        )

        B, L, L = target_distogram.shape

        target_distogram = target_distogram.squeeze(1)  # [B, Lt, Lt]

        # Compute the distogram loss
        disto_loss = torch.nn.functional.cross_entropy(
            logits.view(B * L * L, self.num_bins),
            target_distogram.view(B * L * L),
            reduction="none",
        ).view(B, L, L)

        # Mask out invalid distogram
        mask = f_input.token.disto_mask  # [B, Lt]
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [B, Lt, Lt]
        pair_mask.diagonal(dim1=-2, dim2=-1).zero_()  # zero out diagonal
        pair_mask = pair_mask.to(dtype=logits.dtype)
        disto_loss = disto_loss * pair_mask  # [B, Lt, Lt]

        # Compute mean loss
        disto_loss = disto_loss.sum((-1, -2)) / pair_mask.sum((-1, -2)).clamp(1)  # [B,]
        return disto_loss.mean()
