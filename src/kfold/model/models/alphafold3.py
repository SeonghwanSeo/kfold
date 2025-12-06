from kfold.utils.registry import MAIN_MODULE

from .base import BaseFoldingModel


@MAIN_MODULE.register()
class AlphaFold3(BaseFoldingModel): ...
