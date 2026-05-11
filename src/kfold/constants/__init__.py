from . import (  # noqa: F401
    atom,
    bond,
    ccd,
    chain,
    residue,
    sequence,
    training,
)

NUM_RES_TYPES: int = len(residue.ResidueName)
NUM_ATOM_NAME_CHARS: int = 64  # AlphaFold3
NUM_ATOM_ELEMENTS: int = 128  # AlphaFold3

MAX_NUM_ATOMS_PER_TOKEN: int = 24

# See Section 2.7.3 of the AlphaFold3 paper
INTERFACE_CUTOFF: float = 15.0  # Angstroms

ChainType = chain.ChainType
SubChainType = chain.SubChainType
ResidueName = residue.ResidueName
AtomName = atom.AtomName
ConnectionType = bond.ConnectionType
