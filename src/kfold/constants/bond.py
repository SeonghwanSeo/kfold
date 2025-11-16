from enum import IntEnum


# Same to Boltz's order
class ConnectionType(IntEnum):
    OTHER = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4
    # additional types for covalent bonds
    COVALENT = 5
