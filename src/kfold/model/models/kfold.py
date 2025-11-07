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
        # self.sequence_encoder: submodules.sequence_encoder.BaseSequenceEncoder = (
        #     Registry.instantiate(model_config.sequence_encoder)
        # )
        # self.structure_encoder: submodules.struct_encoder.BaseStructureEncoder = (
        #     Registry.instantiate(model_config.structure_encoder)
        # )

        self.input_embedder: submodules.input_embedder.BaseInputEmbedder = (
            Registry.instantiate(model_config.input_embedder)
        )

        self.trunk: submodules.trunk.BaseTrunk = Registry.instantiate(model_config.trunk)

        self.score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
            model_config.score_model
        )

        # NOTE: structure module is not a torch.nn.Module
        # This handles diffusion sampling as well
        self.structure_module: submodules.structure_module.BaseStructureModule = (
            Registry.instantiate(model_config.structure_module)
        )

        # Heads
        self.distogram_head: submodules.distogram_head.BaseDistogramHead = (
            Registry.instantiate(model_config.distogram_head)
        )

        # self.confidence_head: submodules.confidence_head.BaseConfidenceHead = (
        #     Registry.instantiate(model_config.confidence_head)
        # )

    def forward(
        self,
        f_input: FoldingInput,
        num_recycles: int | None,
        mode: str = "train",
    ):
        raise NotImplementedError("Forward pass is not implemented yet.")
