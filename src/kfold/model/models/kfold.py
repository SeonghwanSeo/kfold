import torch.nn as nn

from kfold.data.model_input import FoldingInput
from kfold.utils.registry import Registry


class KFold(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.global_config = config
        self.config = config.model

        # Initialize sub-modules here using the config
        model_config = config.modelfig
        self.sequence_emb = Registry.instantiate(model_config.seq_repr)
        self.structure_emb = Registry.instantiate(model_config.struct_repr)
        self.transformer = Registry.instantiate(model_config.transformer)
        self.diffusion = Registry.instantiate(model_config.diffusion)
        self.confidence_head = Registry.instantiate(model_config.confidence_head)
        self.dstogram_head = Registry.instantiate(model_config.dstogram_head)

    def forward(self, input: FoldingInput, mode: str = "train"):
        # Implement the forward pass using the sub-modules
        # FIXME: this is a temporary implementation.
        assert mode == "train", "Only 'train' mode is supported currently."

        seq_features = self.sequence_emb(input)
        struct_features = self.structure_emb(input)

        s, z = self.transformer(input, seq_features, struct_features)

        # diffusion training
        # TODO: we may want to add diffusion_batch?
        apo_coords = input.atom.apo_coords
        label_coords = input.atom.label_coords

        coords = self.diffusion(s, z)

        confidence = self.confidence_head(s, z)
        dstogram = self.dstogram_head(s, z)

        return coords, confidence, dstogram
