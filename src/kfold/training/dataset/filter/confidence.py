"""Implemented from https://github.com/jwohlwend/boltz"""

from kfold.data.schema import Metadata
from kfold.utils.registry import DATA_FILTER

from .base import BaseFilter

# TODO: add with confidence score to handle synthetic data


@DATA_FILTER.register()
class ConfidenceFilter(BaseFilter):
    """A filter that filters complexes based on their confidence score.
    NOTE: this works only for synthetic data entries.
    """

    class Config(BaseFilter.Config):
        """
        plddt_threshold (float): The minimum confidence score (pLDDT).
        """

        plddt_threshold: float = 70.0

    # TODO: add more metrics
    def __init__(self, config: Config):
        self.plddt_threshold: float = config.plddt_threshold

    def filter(self, metadata: Metadata) -> bool:
        pred_record = metadata.prediction
        assert pred_record is not None, (
            "ConfidenceFilter only works for synthetic data records"
        )
        assert pred_record.plddt is not None, "Confidence score is empty"
        return pred_record.plddt >= self.plddt_threshold
