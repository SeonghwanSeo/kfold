import dataclasses

from kfold.utils.registry import MAIN_MODULE

from .base import BaseFoldingModel, BaseFoldingModelConfig


@dataclasses.dataclass(kw_only=True)
class KFoldConfig(BaseFoldingModelConfig):
    _class_: str = "KFold"
    # TODO: define encoders
    # sequence_encoder: BaseConfig
    # structure_encoder: BaseConfig


@MAIN_MODULE.register()
class KFold(BaseFoldingModel): ...
