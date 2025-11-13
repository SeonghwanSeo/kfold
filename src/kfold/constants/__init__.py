from . import atom, bond, chain, pocket, residue  # noqa: F401 # fmt: skip

NUM_RES_TYPES: int = len(residue.ResidueName)
NUM_ATOM_NAME_CHARS: int = 64  # AlphaFold3
NUM_ATOM_ELEMENTS: int = 128  # AlphaFold3

MAX_NUM_ATOMS_PER_TOKEN: int = 24

# FIXME: do we need more pocket contact types?
NUM_POCKET_CONTACT_TYPES: int = len(pocket.ContactType)

# See Section 2.7.3 of the AlphaFold3 paper
INTERFACE_CUTOFF: float = 15.0  # Angstroms
