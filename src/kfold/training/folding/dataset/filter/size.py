"""Implement from https://github.com/jwohlwend/boltz"""

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER

from .filter import DataFilter


@DATA_FILTER.register()
class NumChainFilter(DataFilter):
    """Filter the data based on num chains"""

    class Config:
        """
        min_num_chains (int): The minimum number of chains to filter
        max_num_chains (int): The maximum number of chains to filter
        """

        min_num_chains: int = 1
        max_num_chains: int = 300

    def __init__(self, config: Config):
        self.min_num_chains: int = config.min_num_chains
        self.max_num_chains: int = config.max_num_chains

    def filter(self, metadata: Metadata) -> bool:
        num_chains = len(metadata.chains)
        return self.min_num_chains <= num_chains <= self.max_num_chains


class NumResidueFilter(DataFilter):
    """Filter the data based on the number of residues"""

    class Config:
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

    def filter(self, metadata: Metadata) -> bool:
        num_residues = sum(chain.num_residues for chain in metadata.chains)
        return self.min_num_residues <= num_residues <= self.max_num_residues


class NumTokenFilter(DataFilter):
    """Filter the data based on num tokens"""

    class Config:
        """
        Attributes:
            min_num_tokens (int): The minimum number of tokens to filter
            max_num_tokens (int): The maximum number of tokens to filter
        """

        min_num_tokens: int = 1
        max_num_tokens: int = 300

    def __init__(self, config: Config):
        self.min_num_tokens: int = config.min_num_tokens
        self.max_num_tokens: int = config.max_num_tokens

    def filter(self, metadata: Metadata) -> bool:
        num_tokens = sum(chain.num_tokens for chain in metadata.chains)
        return self.min_num_tokens <= num_tokens <= self.max_num_tokens


class NumAtomFilter(DataFilter):
    """Filter the data based on num atoms"""

    class Config:
        """
        Attributes:
            min_num_atoms (int): The minimum number of atoms to filter
            max_num_atoms (int): The maximum number of atoms to filter
        """

        min_num_atoms: int = 1
        max_num_atoms: int = 300

    def __init__(self, config: Config):
        self.min_num_atoms: int = config.min_num_atoms
        self.max_num_atoms: int = config.max_num_atoms

    def filter(self, metadata: Metadata) -> bool:
        num_atoms = sum(chain.num_atoms for chain in metadata.chains)
        return self.min_num_atoms <= num_atoms <= self.max_num_atoms
