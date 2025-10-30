import torch

import kfold.model.modules as submodules
from kfold.data.model_input import FoldingInput
from kfold.utils.registry import Registry


class KFold(torch.nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.config = global_config

        # Initialize sub-modules here using the config
        model_config = global_config.model
        self.sequence_encoder: submodules.seq_encoder.BaseSequenceEncoder = (
            Registry.instantiate(model_config.sequence_encoder)
        )
        # self.structure_encoder: submodules.struct_encoder.BaseStructureEncoder = (
        #     Registry.instantiate(model_config.structure_encoder)
        # )
        # self.transformer: submodules.transformer.BaseTransformer = Registry.instantiate(
        #     model_config.transformer
        # )
        # self.diffusion: submodules.diffusion.BaseDiffusionModule = Registry.instantiate(
        #     model_config.diffusion
        # )
        # self.confidence_head: submodules.confidence_head.BaseConfidenceHead = (
        #     Registry.instantiate(model_config.confidence_head)
        # )
        self.distogram_head: submodules.distogram_head.BaseDistogramHead = (
            Registry.instantiate(model_config.distogram_head)
        )

    def forward(self, input: FoldingInput, mode: str = "train"):
        raise NotImplementedError("Forward pass is not implemented yet.")
