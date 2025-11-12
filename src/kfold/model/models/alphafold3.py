import torch
from omegaconf import DictConfig

import kfold.model.modules as submodules
from kfold.utils.registry import MAIN_MODULE, Registry

from .kfold import KFold


@MAIN_MODULE.register()
class AlphaFold3(KFold):
    def __init__(self, global_config: DictConfig):
        torch.nn.Module.__init__(self)
        self.config = global_config

        # Initialize sub-modules here using the config
        model_config = global_config.model

        self.input_embedder: submodules.input_embedder.BaseInputEmbedder = (
            Registry.instantiate(model_config.input_embedder)
        )

        # TODO: add MSA module

        self.trunk: submodules.trunk.BaseTrunk = Registry.instantiate(model_config.trunk)

        self.score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
            model_config.score_model
        )

        # NOTE: structure module is not a torch.nn.Module
        # This handles diffusion sampling as well
        self.structure_module: submodules.structure_module.BaseStructureModule = (
            Registry.instantiate(
                model_config.structure_module, score_model=self.score_model
            )
        )

        # Heads
        self.distogram_head: submodules.distogram_head.BaseDistogramHead = (
            Registry.instantiate(model_config.distogram_head)
        )

        # TODO: add confidence head
        # self.confidence_head: submodules.confidence_head.BaseConfidenceHead = (
        #     Registry.instantiate(model_config.confidence_head)
        # )
