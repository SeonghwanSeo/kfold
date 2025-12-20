from enum import IntEnum

from rdkit import Chem


# Same to Boltz's order
class ConnectionType(IntEnum):
    OTHER = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4


def bond_type_to_connection_type(bond_type: Chem.BondType) -> ConnectionType:
    """Convert RDKit BondType to ConnectionType."""
    match bond_type:
        case Chem.BondType.SINGLE:
            return ConnectionType.SINGLE
        case Chem.BondType.DOUBLE:
            return ConnectionType.DOUBLE
        case Chem.BondType.TRIPLE:
            return ConnectionType.TRIPLE
        case Chem.BondType.AROMATIC:
            return ConnectionType.AROMATIC
        case _:
            return ConnectionType.OTHER
