from .train_dataset import TrainingDataset, TrainingDatasetConfig  # noqa
from .val_dataset import ValidationDataset, ValidationDatasetConfig
from .rcsb import RCSBTrainingDataset
from .distillation import DistillationDataset


def get_training_dataset_cls(
    train_config: TrainingDatasetConfig,
) -> type["TrainingDataset"]:
    match train_config.type:
        case "rcsb":
            return RCSBTrainingDataset
        case "distillation":
            return DistillationDataset
        case _:
            raise ValueError(f"Unsupported training dataset type: {train_config.type}")
