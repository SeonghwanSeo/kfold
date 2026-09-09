"""Custom distillation datasets stored as serialized RefStructure records."""

from kfold.data.types.metadata import Metadata

from .train_dataset import TrainingDataset


class DistillationDataset(TrainingDataset):
    """Load predicted structures through the shared RefStructure pipeline.

    Each ``structure.lmdb`` value must be written by ``RefStructure.save_npz``;
    its UTF-8 key is the structure's metadata ID. ``manifest.msgpack`` contains
    the corresponding ``Metadata.to_dict()`` records used for sampling.

    Chain types, entity IDs, symmetry IDs, and covalent connections come from
    the stored structure, regardless of chain count or dataset name. Apo and
    prior inputs, including protein multimer conditioning, follow the common
    training pipeline. Monomer and multimer apo lookups are optional.
    """

    def determine_confidence_train_data(self, metadata: Metadata) -> bool:
        """Predicted labels do not supervise the confidence head."""
        return False
