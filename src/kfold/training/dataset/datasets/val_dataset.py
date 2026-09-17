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

import dataclasses

from kfold.data.pipelines import featurization, prior_sampling, tokenization
from kfold.data.types.ccd import CCD

from .base import BaseLMDBDataset, DatasetConfig


@dataclasses.dataclass(kw_only=True)
class ValidationDatasetConfig(DatasetConfig): ...


class ValidationDataset(BaseLMDBDataset):
    """Validation dataset without sampling and cropping."""

    def __init__(
        self,
        config: ValidationDatasetConfig,
        ccd: CCD,
        tokenizer: tokenization.Tokenizer,
        featurizer: featurization.InputFeaturizer,
        prior_sampler: prior_sampling.PriorSampler | None,
        safe_load: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        config : ValidationDatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
        """
        super().__init__(
            config,
            ccd,
            tokenizer,
            featurizer,
            prior_sampler,
            safe_load=safe_load,
            train=False,
        )
        self.config: ValidationDatasetConfig = config

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        cfg = self.config
        if cfg.apo_perturb is not None:
            self.logger.warning("Apo perturbation is enabled for validation dataset.")

    def setup(self) -> None:
        """Additional setup for subclasses."""

        def get_num_tokens(m: dict) -> int:
            return sum(c["num_tokens"] for c in m["chains"])

        self.metadatas.sort(key=lambda m: get_num_tokens(m))
