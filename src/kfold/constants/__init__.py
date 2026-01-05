from . import atom, bond, ccd, chain, constraint, residue, training  # noqa: F401

NUM_RES_TYPES: int = len(residue.ResidueName)
NUM_ATOM_NAME_CHARS: int = 64  # AlphaFold3
NUM_ATOM_ELEMENTS: int = 128  # AlphaFold3

MAX_NUM_ATOMS_PER_TOKEN: int = 24

# FIXME: do we need more pocket contact types?
NUM_CONSTRAINT_TYPES: int = len(constraint.ConstraintType)

# See Section 2.7.3 of the AlphaFold3 paper
INTERFACE_CUTOFF: float = 15.0  # Angstroms

ChainType = chain.ChainType
ResidueName = residue.ResidueName
ConstraintType = constraint.ConstraintType
AtomName = atom.AtomName
ConnectionType = bond.ConnectionType
