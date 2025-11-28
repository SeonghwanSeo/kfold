from omegaconf import DictConfig

from kfold.utils.registry import MAIN_MODULE

from .base import BaseFoldingModel


@MAIN_MODULE.register()
class KFold(BaseFoldingModel):
    def __init__(self, global_config: DictConfig):
        super().__init__(global_config)
        # self.sequence_encoder: submodules.sequence_encoder.BaseSequenceEncoder = (
        #     Registry.instantiate(model_config.sequence_encoder)
        # )
        # self.structure_encoder: submodules.struct_encoder.BaseStructureEncoder = (
        #     Registry.instantiate(model_config.structure_encoder)
        # )
