"""Simple linear interaction head with symmetric logits."""

import torch

from kfold.model.layers.primitives import LinearNoBias
from kfold.utils.registry import INTERACTION_HEAD

from .base import BaseInteractionHead


@INTERACTION_HEAD.register()
class InteractionHead(BaseInteractionHead):
    """Project pair representations to interaction logits and symmetrize."""

    def __init__(self, cfg: BaseInteractionHead.Config) -> None:
        super().__init__(cfg)
        self.linear = LinearNoBias(cfg.channel_z, cfg.num_pair_types, init="final")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return symmetric interaction logits of shape [B, L, L, K]."""
        logits = self.linear(z)
        # Average the two directions to keep the logit scale unchanged.
        logits = 0.5 * (logits + logits.permute(0, 2, 1, 3))
        return logits
