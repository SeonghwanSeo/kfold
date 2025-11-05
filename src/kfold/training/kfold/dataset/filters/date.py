"""Implemented from https://github.com/jwohlwend/boltz"""

from datetime import datetime
from typing import Literal

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER

from .filter import DataFilter


@DATA_FILTER.register()
class ReleaseDateFilter(DataFilter):
    """Filter the data based on their date
    NOTE: this works only for RCSB PDB entries.
    """

    class Config:
        """
        date (str): The maximum date of PDB entries to filter
        ref (str): The reference date to use.
            Can be "deposited", "revised", or "released".
        """

        date: str
        ref: Literal["deposited", "revised", "released"] = "released"

    def __init__(self, config: Config):
        self.filter_date: datetime = datetime.fromisoformat(config.date)
        self.ref: Literal["deposited", "revised", "released"] = config.ref

        if self.ref not in ["deposited", "revised", "released"]:
            raise ValueError(
                "Invalid reference date. Must be deposited, revised, or released"
            )

    def filter(self, metadata: Metadata) -> bool:
        record = metadata.rcsb
        assert record is not None, "DateFilter only works for RCSB records"

        if self.ref == "deposited":
            date = record.deposited
        elif self.ref == "released":
            date = record.released
            if not date:
                date = record.deposited
        elif self.ref == "revised":
            date = record.revised
            if not date and record.released:
                date = record.released
            elif not date:
                date = record.deposited
        else:
            raise ValueError(f"Unknown reference date type: {self.ref}")

        if date is None or date == "":
            return False

        date = datetime.fromisoformat(date)
        return date <= self.filter_date
