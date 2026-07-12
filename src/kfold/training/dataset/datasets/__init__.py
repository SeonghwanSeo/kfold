from .train_dataset import TrainingDataset, TrainingDatasetConfig  # noqa
from .val_dataset import ValidationDataset, ValidationDatasetConfig
from .rcsb import DisorderedPDBTrainingDataset, RCSBTrainingDataset
from .distillation import DistillationDataset
from .monomer import ProteinMonomerDistillationDataset, RNAMonomerDistillationDataset
from .dimer import HeterodimerDistillationDataset, HomodimerDistillationDataset


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
        case "protein-monomer-distillation":
            return ProteinMonomerDistillationDataset
        case "rna-monomer-distillation":
            return RNAMonomerDistillationDataset
        case "homodimer-distillation":
            return HomodimerDistillationDataset
        case "heterodimer-distillation":
            return HeterodimerDistillationDataset
        case _:
            raise ValueError(f"Unsupported training dataset type: {train_config.type}")
