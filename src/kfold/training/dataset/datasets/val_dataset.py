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
