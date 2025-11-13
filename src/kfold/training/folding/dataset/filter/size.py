"""Implement from https://github.com/jwohlwend/boltz"""

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER, BaseConfig

from .base import BaseFilter


@DATA_FILTER.register()
class NumChainFilter(BaseFilter):
    """Filter the data based on num chains"""

    class Config(BaseConfig):
        """
        min_num_chains (int): The minimum number of chains to filter
        max_num_chains (int): The maximum number of chains to filter
        """

        min_num_chains: int = 1
        max_num_chains: int = 300

    def __init__(self, config: Config):
        self.min_num_chains: int = config.min_num_chains
        self.max_num_chains: int = config.max_num_chains

    def filter(self, record: Metadata) -> bool:
        num_chains = record.num_chains
        return self.min_num_chains <= num_chains <= self.max_num_chains


@DATA_FILTER.register()
class NumResidueFilter(BaseFilter):
    """Filter the data based on the number of residues"""

    class Config(BaseConfig):
        """
        Attributes:
            min_num_residues (int): The minimum number of residues to filter
            max_num_residues (int): The maximum number of residues to filter
        """

        min_num_residues: int = 1
        max_num_residues: int = 300

    def __init__(self, config: Config):
        self.min_num_residues: int = config.min_num_residues
        self.max_num_residues: int = config.max_num_residues

    def filter(self, record: Metadata) -> bool:
        num_residues = sum(chain.num_residues for chain in record.chains)
        return self.min_num_residues <= num_residues <= self.max_num_residues
