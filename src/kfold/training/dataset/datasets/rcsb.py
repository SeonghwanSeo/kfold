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
