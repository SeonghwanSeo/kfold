"""Confidence supervision policy for experimental RCSB structures."""

from kfold.data.types.metadata import Metadata

from .train_dataset import TrainingDataset


class RCSBTrainingDataset(TrainingDataset):
    """Train on prepared experimental structures using shared input handling."""

    def determine_confidence_train_data(self, metadata: Metadata) -> bool:
        # For RCSB training dataset, we only train confidence head on the
        # high-resolution experimental structures.
        assert metadata.source == "rcsb", (
            f"Expected metadata source to be 'rcsb' for RCSBTrainingDataset,"
            f" but got '{metadata.source}'."
        )
        assert metadata.exp is not None, (
            "Experimental metadata must be available for RCSBTrainingDataset."
        )
        # Train the confidence head only on experimental structures.
        resolution = metadata.exp.resolution
        if resolution is not None and 0.1 <= resolution <= 4.0:
            return True
        return False
