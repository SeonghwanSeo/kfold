import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.input_embedder.pretrained_embedder_with_interaction import (
    compute_pair_interactions,
)


class InteractionLoss(torch.nn.Module):
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
        """Compute pair interaction loss for protein-protein pairs."""
        with torch.autocast("cuda", enabled=False):
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

        token_mask = f_input.token.pad_mask & f_input.token.disto_mask
        pair_mask = token_mask[:, None, :] & token_mask[:, :, None]
        if self.inter_chain_only:
            asym_id = f_input.token.asym_id
            if f_input.token.is_batched:
                inter_chain = asym_id[:, :, None] != asym_id[:, None, :]
            else:
                inter_chain = asym_id[:, None] != asym_id[None, :]
            pair_mask = pair_mask & inter_chain
        pair_mask.diagonal(dim1=-2, dim2=-1).zero_()

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target.float(),
            reduction="none",
        )
        loss = loss * pair_mask.unsqueeze(-1)

        num_pairs = pair_mask.sum((-1, -2)).clamp_min(1)
        denom = num_pairs * logits.shape[-1]
        loss = loss.sum((-1, -2, -3)) / denom
        return loss
