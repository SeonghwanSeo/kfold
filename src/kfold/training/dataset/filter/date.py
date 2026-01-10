"""Implemented from https://github.com/jwohlwend/boltz"""

from datetime import datetime

from kfold.data.types.metadata import Metadata
from kfold.utils.registry import DATA_FILTER

from .base import BaseFilter


@DATA_FILTER.register()
class ReleaseDateFilter(BaseFilter):
    """Filter the data based on their date
    NOTE: this works only for RCSB PDB entries.
    """

    class Config(BaseFilter.Config):
        """
        date (str): The maximum date of PDB entries to filter
        """

        date: str = "2021-09-30"

    def __init__(self, config: Config):
        self.filter_date: datetime = datetime.fromisoformat(config.date)

    def filter(self, metadata: Metadata) -> bool:
        exp_record = metadata.exp
        assert exp_record is not None, "DateFilter only works for RCSB records"
        date = exp_record.release_date
        if date is None or date == "":
            return False
        date = datetime.fromisoformat(date)
        return date <= self.filter_date
