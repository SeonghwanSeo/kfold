import kfold.constants as C
from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_FILTER

from .base import BaseFilter


@DATA_FILTER.register()
class CompositionFilter(BaseFilter):
    """Filter the data based on the composition of biomolecular complex

    Rule:
    - Filter out the single-chain data.
    - Filter out the data without at least one protein chain.

    TODO:
    - (SeonghwanSeo) We may want to add ligand ccd filtering in teh future.
      Currently we simply use boltz's preprocessed data, which already filters
      out some common molecules.
    """

    class Config(BaseFilter.Config):
        """
        remove_excluding_ligands (bool): Whether to remove ligands with excluded CCDs.
        """

        remove_excluding_ligands: bool = False

    def __init__(self, config: Config):
        self.remove_excluding_ligands: bool = config.remove_excluding_ligands

    def filter(self, record: Metadata) -> bool:
        chains = record.chains

        if self.remove_excluding_ligands:
            # Remove ligands with excluded CCDs
            chains = [
                chain
                for chain in record.chains
                if not (
                    chain.chain_type == C.chain.ChainType.LIGAND
                    and chain.chain_name in C.ccd.LIGAND_EXCLUSIONS
                )
            ]

        if len(chains) < 2:
            # Remove single-chain data
            return False

        has_protein = any(
            chain.chain_type == C.chain.ChainType.PROTEIN for chain in chains
        )
        if not has_protein:
            return False

        return True
