# Started from https://github.com/jwohlwend/boltz
import numpy as np

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_SAMPLER

from .base import BaseSampler, Sample


@DATA_SAMPLER.register()
class RandomSampler(BaseSampler):
    """Random sampler for dataset items.

    This sampler randomly selects items from the dataset, either at the
    complex level or the chain level, based on the configuration.
    """

    class Config:
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

    def setup(self, records: list[Metadata]) -> None:
        self.items: list[Sample] = []
        if self.chain_level:
            self.items = [Sample(record, None) for record in records]
        else:
            for record in records:
                for chain in record.chains:
                    if not chain.valid:
                        continue
                    self.items.append(Sample(record, (chain.asym_id,)))

        self.weights: np.ndarray = np.ones(len(self.items)) / len(self.items)
