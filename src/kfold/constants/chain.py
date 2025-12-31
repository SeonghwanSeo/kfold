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
