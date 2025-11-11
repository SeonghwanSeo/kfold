"""Implemented from https://github.com/jwohlwend/boltz"""

from pathlib import Path

from kfold.data.metadata import Metadata

from .filter import DataFilter


class SubsetFilter(DataFilter):
    """Filter a data record based on a subset of the data."""

    def __init__(self, subset_path: str | Path, exclude: bool = False) -> None:
        """Initialize the filter.

        Parameters
        ----------
        subset : str
            The subset of data to consider, one per line.
        exclude: str
            If True, the filter will exclude the subset from the data.
            If False, the filter will include only the subset in the data.
        """
        with Path(subset_path).open("r") as f:
            subset = f.read().splitlines()

        self.subset: set[str] = {s.lower() for s in subset}
        self.exclude: bool = exclude

    def filter(self, metadata: Metadata) -> bool:
        is_in_subset = metadata.id.lower() in self.subset
        return not is_in_subset if self.exclude else is_in_subset
