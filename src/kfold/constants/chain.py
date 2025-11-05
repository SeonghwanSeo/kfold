from enum import IntEnum


# Same to Boltz's order
class ChainType(IntEnum):
    Protein = 0
    DNA = 1
    RNA = 2
    Ligand = 3


# TODO: may want to add some mmcif-related informations for data preprocessing
# e.g., mmcif chain type naming.
