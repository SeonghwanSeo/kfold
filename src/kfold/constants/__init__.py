from . import atom, bond, chain, pocket, residue  # noqa: F401 # fmt: skip

NUM_RES_TYPES: int = len(residue.ResidueName)
NUM_ATOM_NAME_CHARS: int = 64  # AlphaFold3
NUM_ATOM_ELEMENTS: int = 128  # AlphaFold3

# FIXME: do we need more pocket contact types?
NUM_POCKET_CONTACT_TYPES: int = len(pocket.ContactType)
