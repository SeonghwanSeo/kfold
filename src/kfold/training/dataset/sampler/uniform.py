import numpy as np

from kfold.data.types.metadata import Metadata
from kfold.utils.registry import DATA_SAMPLER

from .base import BaseSampler, Sample


@DATA_SAMPLER.register()
class UniformSampler(BaseSampler):
    """Random sampler for dataset items.

    This sampler randomly selects items from the dataset, either at the
    complex level or the chain level, based on the configuration.
    """

    class Config(BaseSampler.Config):
        """Configuration for RandomSampler.

        Parameters
        ----------
        level : str
            option: 'complex', 'chain', or 'interface'
        """

        level: str = "complex"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.level = config.level.lower()
        assert self.level in [
            "complex",
            "chain",
            "interface",
        ], f"Unsupported sampling level: {self.level}"

    def get_samples(self, metadatas: list[Metadata]) -> tuple[list[Sample], np.ndarray]:
        samples: list[Sample]
        if self.level == "chain":
            samples = [Sample(m, c.asym_id) for m in metadatas for c in m.chains]
        elif self.level == "interface":
            samples = [
                Sample(m, tuple(iface.asym_ids))
                for m in metadatas
                for iface in m.interfaces
            ]
        else:
            samples = [Sample(m, None) for m in metadatas]

        weights = np.ones(len(samples), dtype=np.float32)
        return samples, weights
