"""Implemented from https://github.com/jwohlwend/boltz"""

from abc import ABC, abstractmethod

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER


@DATA_FILTER.register()
class DataFilter(ABC):
    """Base class for data filters based on metadata."""

    def __call__(self, metadata: Metadata) -> bool:
        return self.filter(metadata)

    @abstractmethod
    def filter(self, metadata: Metadata) -> bool:
        """Filter a data.

        Parameters
        ----------
        metadata : Metadata
            The object to consider filtering in / out.

        Returns
        -------
        bool
            True if the data passes the filter, False otherwise.
        """
