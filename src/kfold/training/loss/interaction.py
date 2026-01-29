"""Interaction loss with distance-gated pseudo-labels."""

import torch

import kfold.constants as C
from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.input_embedder.pretrained_embedder_with_interaction import (
    compute_pair_interactions,
)


class InteractionLoss(torch.nn.Module):
    """Binary interaction loss over pair types with distance gating."""

    def __init__(
        self, distance_threshold: float = 7.5, inter_chain_only: bool = False
    ) -> None:
        super().__init__()
        self.distance_threshold = distance_threshold
        self.inter_chain_only = inter_chain_only

    def forward(
        self,
        logits: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute pair interaction loss from pseudo-labels.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted interaction logits of shape [B, L, L, K].
        f_input : FoldingInput
            Model input containing token coordinates and interaction primitives.

        Returns
        -------
        torch.Tensor
            Per-sample loss of shape [B].
        """
        with torch.autocast("cuda", enabled=False):
            # Use float32 distances for stable gating near the threshold.
            pdist = torch.cdist(
                f_input.token.disto_coords,
                f_input.token.disto_coords,
            )
            within_threshold = pdist <= self.distance_threshold
            pair_interactions = compute_pair_interactions(
                f_input.token.interaction_type,
                f_input.token.chain_type,
            )
            target = pair_interactions * within_threshold.unsqueeze(-1)

        # Exclude legacy ligand/ion tokens that carry "all-ones" placeholder
        # primitives to avoid training on dense all-negative pairs.
        interaction_type = f_input.token.interaction_type
        ligand_mask = f_input.token.is_ligand
        num_active = interaction_type[..., 1:].sum(-1)
        placeholder_ligand = ligand_mask & (num_active == (C.NUM_INTERACTION_TYPES - 1))

        token_mask = (
            f_input.token.pad_mask & f_input.token.disto_mask & ~placeholder_ligand
        )
        pair_mask = token_mask[:, None, :] & token_mask[:, :, None]
        if self.inter_chain_only:
            # Optionally ignore intra-chain pairs.
            asym_id = f_input.token.asym_id
            if f_input.token.is_batched:
                inter_chain = asym_id[:, :, None] != asym_id[:, None, :]
            else:
                inter_chain = asym_id[:, None] != asym_id[None, :]
            pair_mask = pair_mask & inter_chain
        # Self pairs are not meaningful interaction targets.
        pair_mask.diagonal(dim1=-2, dim2=-1).zero_()

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target.float(),
            reduction="none",
        )
        pair_mask_expanded = pair_mask.unsqueeze(-1)
        loss = loss * pair_mask_expanded

        # Class imbalance is extreme (very sparse positives), which drives the
        # mean loss toward ~0 even when the head is learning. Balance the scale
        # by averaging positive and negative terms separately, then combining
        # them. This keeps the loss in a distogram-like range without changing
        # external loss weights.
        pos_mask = (target > 0).to(dtype=loss.dtype) * pair_mask_expanded
        neg_mask = (target <= 0).to(dtype=loss.dtype) * pair_mask_expanded

        pos_count_raw = pos_mask.sum((-1, -2, -3))
        neg_count_raw = neg_mask.sum((-1, -2, -3))
        pos_count = pos_count_raw.clamp_min(1.0)
        neg_count = neg_count_raw.clamp_min(1.0)

        pos_loss = (loss * pos_mask).sum((-1, -2, -3)) / pos_count
        neg_loss = (loss * neg_mask).sum((-1, -2, -3)) / neg_count

        has_pos = pos_count_raw > 0
        balanced_loss = 0.5 * (pos_loss + neg_loss)
        return torch.where(has_pos, balanced_loss, neg_loss)
