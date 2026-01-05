"""Implemented from https://github.com/jwohlwend/boltz"""

from abc import ABC, abstractmethod

from kfold.data.types.metadata import Metadata
from kfold.utils.registry import DATA_FILTER, BaseConfig


@DATA_FILTER.register()
class BaseFilter(ABC):
    """Base class for data filters based on metadata."""

    class Config(BaseConfig):
        """Configuration for BaseFilter."""

    def __call__(self, metadata: Metadata) -> bool:
        return self.filter(metadata)

    @abstractmethod
    def filter(self, metadata: Metadata) -> bool:
        """Filter a data.

        Parameters
        ----------
        metadata : Metadata
            The metadata to consider filtering in / out.

        Returns
        -------
        bool
            True if the data passes the filter, False otherwise.
        """
