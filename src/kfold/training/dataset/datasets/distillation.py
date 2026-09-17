# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
