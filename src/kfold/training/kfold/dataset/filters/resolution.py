"""Implemented from https://github.com/jwohlwend/boltz"""

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER

from .filter import DataFilter


@DATA_FILTER.register()
class ResolutionFilter(DataFilter):
    """A filter that filters complexes based on their resolution.
    NOTE: this works only for RCSB PDB entries.
    """

    class Config:
        """
        resolution (float): The maximum allowed resolution.
        """

        resolution: float = 4.0

    def __init__(self, config: Config):
        self.resolution: float = config.resolution

    def filter(self, metadata: Metadata) -> bool:
        record = metadata.rcsb
        assert record is not None, "DateFilter only works for RCSB records"
        assert record.resolution is not None, "Resolution is empty"
        return record.resolution <= self.resolution
