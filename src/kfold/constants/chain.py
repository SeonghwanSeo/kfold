from enum import IntEnum


class ChainType(IntEnum):
    Protein = 0
    RNA = 1
    DNA = 2
    Ligand = 3


# TODO: may want to add some mmcif-related informations for data preprocessing
# e.g., mmcif chain type naming.
