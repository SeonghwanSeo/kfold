"""Implemented from https://github.com/jwohlwend/boltz"""

from datetime import datetime
from typing import Literal

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER, BaseConfig

from .base import BaseFilter


@DATA_FILTER.register()
class ReleaseDateFilter(BaseFilter):
    """Filter the data based on their date
    NOTE: this works only for RCSB PDB entries.
    """

    class Config(BaseConfig):
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

    def filter(self, record: Metadata) -> bool:
        exp_record = record.exp
        assert exp_record is not None, "DateFilter only works for RCSB records"

        if self.ref == "deposited":
            date = exp_record.deposited
        elif self.ref == "released":
            date = exp_record.released
            if not date:
                date = exp_record.deposited
        elif self.ref == "revised":
            date = exp_record.revised
            if not date and exp_record.released:
                date = exp_record.released
            elif not date:
                date = exp_record.deposited
        else:
            raise ValueError(f"Unknown reference date type: {self.ref}")

        if date is None or date == "":
            return False

        date = datetime.fromisoformat(date)
        return date <= self.filter_date
