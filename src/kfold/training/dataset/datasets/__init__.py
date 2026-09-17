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
