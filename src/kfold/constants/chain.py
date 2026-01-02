from enum import IntEnum


# Same to Boltz's order
class ChainType(IntEnum):
    PROTEIN = 0
    DNA = 1
    RNA = 2
    LIGAND = 3
    ION = 4

    @property
    def is_polymer(self) -> bool:
        return self in {ChainType.PROTEIN, ChainType.DNA, ChainType.RNA}

    @property
    def is_nonpolymer(self) -> bool:
        return self in {ChainType.LIGAND, ChainType.ION}

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

    def __str__(self) -> str:
        if self == ChainType.PROTEIN:
            return "Protein"
        elif self == ChainType.DNA:
            return "DNA"
        elif self == ChainType.RNA:
            return "RNA"
        elif self == ChainType.LIGAND:
            return "Ligand"
        elif self == ChainType.ION:
            return "Ion"
        else:
            raise ValueError(f"Unknown ChainType: {self.value}")


# TODO: may want to add some mmcif-related informations for data preprocessing
# e.g., mmcif chain type naming.
