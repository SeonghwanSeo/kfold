from kfold.utils.registry import MAIN_MODULE

from .base import BaseFoldingModel, BaseFoldingModelConfig


class KFoldConfig(BaseFoldingModelConfig):
    pass
    # TODO: define encoders
    # sequence_encoder: BaseConfig
    # structure_encoder: BaseConfig


@MAIN_MODULE.register()
class KFold(BaseFoldingModel):
    def __init__(self, config: KFoldConfig):
        super().__init__(config)
        # self.sequence_encoder: submodules.sequence_encoder.BaseSequenceEncoder = (
        #     Registry.instantiate(model_config.sequence_encoder)
        # )
        # self.structure_encoder: submodules.struct_encoder.BaseStructureEncoder = (
        #     Registry.instantiate(model_config.structure_encoder)
        # )
