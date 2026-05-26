from .train_dataset import TrainingDataset, TrainingDatasetConfig  # noqa
from .val_dataset import ValidationDataset, ValidationDatasetConfig
from .rcsb import DisorderedPDBTrainingDataset, RCSBTrainingDataset
from .distillation import DistillationDataset
from .monomer import MonomerDistillationDataset, RNAMonomerDistillationDataset
from .homodimer import HomodimerDistillationDataset


def get_training_dataset_cls(
    train_config: TrainingDatasetConfig,
) -> type["TrainingDataset"]:
    match train_config.type:
        case "rcsb":
            return RCSBTrainingDataset
        case "disordered_pdb":
            return DisorderedPDBTrainingDataset
        case "distillation":
            return DistillationDataset
        case "monomer-distillation":
            return MonomerDistillationDataset
        case "rna-monomer-distillation":
            return RNAMonomerDistillationDataset
        case "homodimer-distillation":
            return HomodimerDistillationDataset
        case _:
            raise ValueError(f"Unsupported training dataset type: {train_config.type}")
