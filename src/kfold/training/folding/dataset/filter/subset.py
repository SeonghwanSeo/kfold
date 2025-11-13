"""Implemented from https://github.com/jwohlwend/boltz"""

from pathlib import Path

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER, BaseConfig

from .base import BaseFilter


@DATA_FILTER.register()
class SubsetFilter(BaseFilter):
    """Filter a data record based on a subset of the data."""

    class Config(BaseConfig):
        """
        Attributes:
            subset_path (str): The path to the subset file.
            exclude (bool): If True, the filter will exclude the subset from the data.
        """

        subset_path: str | Path
        exclude: bool = False

    def __init__(self, config: Config) -> None:
        """Initialize the filter.

        Parameters
        ----------
        subset : str
            The subset of data to consider, one per line.
        exclude: str
            If True, the filter will exclude the subset from the data.
            If False, the filter will include only the subset in the data.
        """
        with Path(config.subset_path).open("r") as f:
            subset = f.read().splitlines()

        self.subset: set[str] = {s.lower() for s in subset}
        self.exclude: bool = config.exclude

    def filter(self, record: Metadata) -> bool:
        is_in_subset = record.id.lower() in self.subset
        return not is_in_subset if self.exclude else is_in_subset
