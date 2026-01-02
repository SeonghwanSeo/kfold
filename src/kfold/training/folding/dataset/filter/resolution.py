"""Implemented from https://github.com/jwohlwend/boltz"""

from kfold.data.schema import Metadata
from kfold.utils.registry import DATA_FILTER

from .base import BaseFilter


@DATA_FILTER.register()
class ResolutionFilter(BaseFilter):
    """A filter that filters complexes based on their resolution.
    NOTE: this works only for RCSB PDB entries.
    """

    class Config(BaseFilter.Config):
        """
        resolution (float): The maximum allowed resolution.
        """

        resolution: float = 9.0

    def __init__(self, config: Config):
        self.resolution: float = config.resolution

    def filter(self, metadata: Metadata) -> bool:
        exp_record = metadata.exp
        assert exp_record is not None, "ResolutionFilter only works for RCSB records"
        if exp_record.resolution is None:
            return False
        return exp_record.resolution <= self.resolution
