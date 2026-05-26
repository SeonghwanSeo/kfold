"""Dataset classes for training with distillation data"""

from kfold.data.types.metadata import Metadata

from .train_dataset import TrainingDataset

StructInfo = dict


class DistillationDataset(TrainingDataset):
    """Training dataset for distillation"""

    def determine_confidence_train_data(self, metadata: Metadata) -> bool:
        # No confidence training for distillation dataset
        return False
