import torch

from kfold.model.layers.primitives import LinearNoBias
from kfold.utils.registry import INTERACTION_HEAD

from .base import BaseInteractionHead


@INTERACTION_HEAD.register()
class InteractionHead(BaseInteractionHead):
    def __init__(self, cfg: BaseInteractionHead.Config) -> None:
        super().__init__(cfg)
        self.linear = LinearNoBias(cfg.channel_z, cfg.num_pair_types, init="final")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        logits = logits + logits.permute(0, 2, 1, 3)
        return logits
