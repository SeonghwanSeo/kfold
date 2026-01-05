"""Implement from https://github.com/jwohlwend/boltz"""

from kfold.data.types.metadata import Metadata
from kfold.utils.registry import DATA_FILTER

from .base import BaseFilter


@DATA_FILTER.register()
class NumChainFilter(BaseFilter):
    """Filter the data based on num chains"""

    class Config(BaseFilter.Config):
        """
        min_chains (int): The minimum number of chains to filter
        max_chains (int): The maximum number of chains to filter
        """

        min_chains: int = 1
        max_chains: int = 300

    def __init__(self, config: Config):
        self.min_chains: int = config.min_chains
        self.max_chains: int = config.max_chains

    def filter(self, metadata: Metadata) -> bool:
        num_chains = metadata.num_chains
        return self.min_chains <= num_chains <= self.max_chains


@DATA_FILTER.register()
class NumResidueFilter(BaseFilter):
    """Filter the data based on the number of residues"""

    class Config(BaseFilter.Config):
        """
        Attributes:
            min_residues (int): The minimum number of residues to filter
            max_residues (int): The maximum number of residues to filter
        """

        min_residues: int = 1
        max_residues: int = 2048

    def __init__(self, config: Config):
        self.min_residues: int = config.min_residues
        self.max_residues: int = config.max_residues

    def filter(self, metadata: Metadata) -> bool:
        num_residues = sum(chain.num_residues for chain in metadata.chains)
        return self.min_residues <= num_residues <= self.max_residues
