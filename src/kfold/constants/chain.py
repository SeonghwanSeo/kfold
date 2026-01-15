from enum import IntEnum


class ChainType(IntEnum):
    PROTEIN = 0
    DNA = 1
    RNA = 2
    LIGAND = 3

    @property
    def is_polymer(self) -> bool:
        return self in {ChainType.PROTEIN, ChainType.DNA, ChainType.RNA}

    @property
    def is_nonpolymer(self) -> bool:
        return self in {ChainType.LIGAND}

    @property
    def is_protein(self) -> bool:
        return self is ChainType.PROTEIN

    @property
    def is_nucleic_acid(self) -> bool:
        return self in {ChainType.DNA, ChainType.RNA}

    @property
    def is_dna(self) -> bool:
        return self is ChainType.DNA

    @property
    def is_rna(self) -> bool:
        return self is ChainType.RNA

    @property
    def is_ligand(self) -> bool:
        return self is ChainType.LIGAND

    def __str__(self) -> str:
        match self:
            case ChainType.PROTEIN:
                return "Protein"
            case ChainType.DNA:
                return "DNA"
            case ChainType.RNA:
                return "RNA"
            case ChainType.LIGAND:
                return "Ligand"


# TODO: may want to add some mmcif-related informations for data preprocessing
# e.g., mmcif chain type naming.
