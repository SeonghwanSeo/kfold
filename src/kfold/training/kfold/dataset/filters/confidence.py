"""Implemented from https://github.com/jwohlwend/boltz"""

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER

from .filter import DataFilter

# TODO: add with confidence score to handle synthetic data


@DATA_FILTER.register()
class ConfidenceFilter(DataFilter):
    """A filter that filters complexes based on their confidence score.
    NOTE: this works only for synthetic data entries.
    """

    class Config:
        """
        plddt_threshold (float): The minimum confidence score (pLDDT).
        """

        plddt_threshold: float = 70.0

    # TODO: add more metrics
    def __init__(self, config: Config):
        self.plddt_threshold: float = config.plddt_threshold

    def filter(self, metadata: Metadata) -> bool:
        record = metadata.prediction
        assert record is not None, "DateFilter only works for synthetic data records"
        assert record.plddt is not None, "Confidence score is empty"
        return record.plddt >= self.plddt_threshold
