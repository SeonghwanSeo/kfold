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
            The computed distogram loss of shape (B,).
        """

        target_distogram = self.compute_distogram(
            atom_coords=f_input.atom.label_coords,
            disto_index=f_input.token.disto_index,
        )  # [B, Nholo, Lt, Lt]

        B, Nholo, L, L = target_distogram.shape
        assert Nholo == 1, "Currently only supports single holo coordinates."

        target_distogram = target_distogram.squeeze(1)  # [B, Lt, Lt]

        # Get mask
        mask = f_input.token.disto_mask  # [B, Lt]
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [B, Lt, Lt]
        pair_mask.diagonal(dim1=-2, dim2=-1).zero_()  # zero out diagonal
        pair_mask = pair_mask.to(dtype=logits.dtype)

        # Compute the distogram loss
        disto_loss = torch.nn.functional.cross_entropy(
            logits.view(B * L * L, self.num_bins),
            target_distogram.view(),
            reduction="none",
        ).view(B, L, L)

        # Mask out invalid positions
        disto_loss = disto_loss * pair_mask  # [B, Lt, Lt]

        # Compute mean loss
        disto_loss = disto_loss.sum((-1, -2)) / pair_mask.sum((-1, -2)).clamp(1)
        return disto_loss  # [B,]

    def compute_distogram(
        self,
        atom_coords: torch.Tensor,
        disto_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the target distogram from pairwise distances.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Tensor of shape (B, Nholo, Latom, 3) containing atom coordinates.
        disto_index : torch.Tensor
            Tensor of shape (B, Ltoken) containing the indices of atoms used for distogram

        Returns
        -------
        target_distogram : torch.Tensor
            Tensor of shape (B, Nholo, Lt, Lt) containing the target distogram.
        """
        batch_size = atom_coords.shape[0]
        batch_indices = torch.arange(batch_size, device=atom_coords.device).unsqueeze(-1)

        disto_coords = atom_coords[batch_indices, :, disto_index]  # [B, Nholo, Ltoken, 3]
        disto_pair_dist = torch.cdist(
            disto_coords, disto_coords
        )  # [B, Nholo, Ltoken, Ltoken]
        target_distogram = (
            (disto_pair_dist.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        )
        return target_distogram
