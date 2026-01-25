from . import (  # noqa: F401
    atom,
    bond,
    ccd,
    chain,
    constraint,
    interaction,
    residue,
    training,
)

NUM_RES_TYPES: int = len(residue.ResidueName)
NUM_ATOM_NAME_CHARS: int = 64  # AlphaFold3
NUM_ATOM_ELEMENTS: int = 128  # AlphaFold3

MAX_NUM_ATOMS_PER_TOKEN: int = 24

# FIXME: do we need more pocket contact types?
NUM_CONSTRAINT_TYPES: int = len(constraint.ConstraintType)
NUM_INTERACTION_TYPES: int = interaction.NUM_INTERACTION_TYPES
NUM_PAIR_INTERACTION_TYPES: int = interaction.NUM_PAIR_INTERACTION_TYPES

# See Section 2.7.3 of the AlphaFold3 paper
INTERFACE_CUTOFF: float = 15.0  # Angstroms

ChainType = chain.ChainType
ResidueName = residue.ResidueName
ConstraintType = constraint.ConstraintType
AtomName = atom.AtomName
ConnectionType = bond.ConnectionType
InteractionType = interaction.InteractionType
