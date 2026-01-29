# ruff: noqa: N815
"""PLIP Interaction Type Constants.

PLIP (Protein-Ligand Interaction Profiler) interaction types for
protein-ligand, protein-protein, and protein-nucleic acid interactions.

Reference:
- PLIP 2015: https://pmc.ncbi.nlm.nih.gov/articles/PMC4489249/
- PLIP 2021: https://academic.oup.com/nar/article/49/W1/W530/6266421
- PLIP 2025: https://academic.oup.com/nar/article/53/W1/W463/8128215

Primitive Interaction Types (properties that tokens/atoms can have):
- DUMMY: No interaction capability (padding or unknown)
- HI: Hydrophobic
- HBD: Hydrogen Bond Donor
- HBA: Hydrogen Bond Acceptor
- SBC: Salt Bridge Cation (positive charge)
- SBA: Salt Bridge Anion (negative charge)
- PP: Pi-system (aromatic ring)
- PC: Positive Charge for pi-cation interaction

Pair Interaction Types (formed by primitive type pairs):
1. Hydrophobic Interaction: HI - HI
2. Hydrogen Bond: HBD - HBA (bidirectional)
3. Salt Bridge: SBC - SBA (bidirectional)
4. Pi-Pi Stacking: PP - PP
5. Pi-Cation Interaction: PP - PC (bidirectional)
"""

from enum import IntEnum
from functools import lru_cache

from .chain import ChainType
from .residue import ResidueName

__all__ = [
    "InteractionType",
    "PairInteractionType",
    "NUM_INTERACTION_TYPES",
    "NUM_PAIR_INTERACTION_TYPES",
    "RESIDUE_INTERACTION_FEATURES",
    "get_residue_interaction_type",
]


class InteractionType(IntEnum):
    """Primitive interaction type indices for multi-hot encoding.

    These represent properties that individual tokens/atoms can have.
    The actual interactions are formed by pairs of these primitive types.

    Note: DUMMY is at index 0.
    """

    DUMMY = 0
    HI = 1  # Hydrophobic
    HBD = 2  # Hydrogen Bond Donor
    HBA = 3  # Hydrogen Bond Acceptor
    SBC = 4  # Salt Bridge Cation
    SBA = 5  # Salt Bridge Anion
    PP = 6  # Pi-system (aromatic)
    PC = 7  # Positive Charge for pi-cation

    @property
    def idx(self) -> int:
        """Get the index of the interaction type."""
        return self.value


class PairInteractionType(IntEnum):
    """Pair-wise interaction types formed by primitive type pairs."""

    HYDROPHOBIC = 0  # HI - HI
    HYDROGEN_BOND = 1  # HBD - HBA (bidirectional)
    SALT_BRIDGE = 2  # SBC - SBA (bidirectional)
    PI_PI = 3  # PP - PP
    PI_CATION = 4  # PP - PC (bidirectional)

    @property
    def idx(self) -> int:
        """Get the index of the pair interaction type."""
        return self.value


NUM_INTERACTION_TYPES = len(InteractionType)
NUM_PAIR_INTERACTION_TYPES = len(PairInteractionType)

RESIDUE_INTERACTION_FEATURES: dict[ResidueName, tuple[InteractionType, ...]] = {
    # Hydrophobic residues
    ResidueName.ALA: (InteractionType.HI,),
    ResidueName.VAL: (InteractionType.HI,),
    ResidueName.LEU: (InteractionType.HI,),
    ResidueName.ILE: (InteractionType.HI,),
    ResidueName.MET: (InteractionType.HI,),
    ResidueName.PRO: (InteractionType.HI,),
    # Aromatic residues (hydrophobic + π-stacking capable)
    ResidueName.PHE: (
        InteractionType.HI,
        InteractionType.PP,
    ),
    ResidueName.TRP: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.PP,
    ),
    ResidueName.TYR: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.PP,
    ),
    # Polar residues - H-bond donors/acceptors
    ResidueName.SER: (InteractionType.HBD, InteractionType.HBA),
    ResidueName.THR: (InteractionType.HBD, InteractionType.HBA),
    ResidueName.CYS: (InteractionType.HI, InteractionType.HBD, InteractionType.HBA),
    ResidueName.ASN: (InteractionType.HBD, InteractionType.HBA),
    ResidueName.GLN: (InteractionType.HBD, InteractionType.HBA),
    # Charged residues - Salt bridge capable
    ResidueName.ASP: (InteractionType.HBA, InteractionType.SBA),
    ResidueName.GLU: (InteractionType.HBA, InteractionType.SBA),
    ResidueName.LYS: (
        InteractionType.HBD,
        InteractionType.SBC,
        InteractionType.PC,
    ),
    ResidueName.ARG: (
        InteractionType.HBD,
        InteractionType.SBC,
        InteractionType.PC,
    ),
    ResidueName.HIS: (
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.PP,
    ),
    # Glycine - minimal interactions
    ResidueName.GLY: (),
    # Unknown protein residue (modified amino acids, etc.)
    ResidueName.UNK: (),
    # RNA nucleotides
    ResidueName.A: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
    ResidueName.G: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
    ResidueName.C: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
    ResidueName.U: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
    ResidueName.N: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
    # DNA nucleotides
    ResidueName.DA: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
    ResidueName.DG: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
    ResidueName.DC: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
    ResidueName.DT: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
    ResidueName.DN: (
        InteractionType.HI,
        InteractionType.HBD,
        InteractionType.HBA,
        InteractionType.SBA,
        InteractionType.PP,
    ),
}


@lru_cache(maxsize=128)
def get_residue_interaction_type(
    res_name: ResidueName | str,
    chain_type: ChainType,
) -> tuple[int, ...]:
    """Get multi-hot interaction type indices for a residue."""
    if chain_type.is_ligand:
        return tuple(range(1, NUM_INTERACTION_TYPES))

    if isinstance(res_name, str):
        try:
            res_name = ResidueName[res_name]
        except KeyError:
            pass

    interactions = RESIDUE_INTERACTION_FEATURES.get(res_name, ())  # type: ignore
    return tuple(interaction.idx for interaction in interactions)
