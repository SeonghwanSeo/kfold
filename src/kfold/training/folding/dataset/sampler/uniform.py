# Started from https://github.com/jwohlwend/boltz
from kfold.data.schema import Metadata
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
        chain_level : bool
            If True, sample at chain level; otherwise, sample at complex level.
        """

        chain_level: bool = False

    def __init__(self, config: Config) -> None:
        self.config = config
        self.chain_level = config.chain_level

    def get_samples(self, metadatas: list[Metadata]) -> tuple[list[Sample], None]:
        samples: list[Sample]
        if self.chain_level:
            samples = [Sample(m, None) for m in metadatas]
        else:
            samples = []
            for m in metadatas:
                for chain in m.chains:
                    if not chain.valid:
                        continue
                    samples.append(Sample(m, chain.asym_id))

        weights = None  # Uniform sampling
        return samples, weights
