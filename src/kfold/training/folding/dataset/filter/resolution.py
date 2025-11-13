"""Implemented from https://github.com/jwohlwend/boltz"""

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER, BaseConfig

from .base import BaseFilter


@DATA_FILTER.register()
class ResolutionFilter(BaseFilter):
    """A filter that filters complexes based on their resolution.
    NOTE: this works only for RCSB PDB entries.
    """

    class Config(BaseConfig):
        """
        resolution (float): The maximum allowed resolution.
        """

        resolution: float = 4.0

    def __init__(self, config: Config):
        self.resolution: float = config.resolution

    def filter(self, record: Metadata) -> bool:
        exp_record = record.exp
        assert exp_record is not None, "DateFilter only works for RCSB records"
        assert exp_record.resolution is not None, "Resolution is empty"
        return exp_record.resolution <= self.resolution
