import enum
from collections.abc import Sequence
from functools import lru_cache

from . import residue
from .residue import ResidueName

# === Constants lists === #

protein_atom37: tuple[str, ...] = (
    "N",   "CA",  "C",   "CB",  "O",   "CG",  "CG1", "CG2", "OG",  "OG1",
    "SG",  "CD",  "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",  "CE",
    "CE1", "CE2", "CE3", "NE",  "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH",  "CZ",  "CZ2", "CZ3", "NZ",  "OXT"
)  # fmt: skip

nucleic_acid_atom29: tuple[str, ...] = (
    "C1'", "C2",  "C2'", "C3'", "C4",  "C4'", "C5",  "C5'", "C6",  "C7",
    "C8",  "N1",  "N2",  "N3",  "N4",  "N6",  "N7",  "N9",  "OP3", "O2",
    "O2'", "O3'", "O4",  "O4'", "O5'", "O6",  "OP1", "OP2", "P"
)  # fmt: skip

protein_atom37_order: dict[str, int] = {
    name: idx for idx, name in enumerate(protein_atom37)
}
nucleic_acid_atom29_order: dict[str, int] = {
    name: idx for idx, name in enumerate(nucleic_acid_atom29)
}


class AtomName(enum.StrEnum):
    """This is also used for indexing atom types"""

    # protein
    N = "N"
    CA = "CA"
    C = "C"
    CB = "CB"
    O = "O"  # noqa
    CG = "CG"
    CG1 = "CG1"
    CG2 = "CG2"
    OG = "OG"
    OG1 = "OG1"
    SG = "SG"
    CD = "CD"
    CD1 = "CD1"
    CD2 = "CD2"
    ND1 = "ND1"
    ND2 = "ND2"
    OD1 = "OD1"
    OD2 = "OD2"
    SD = "SD"
    CE = "CE"
    CE1 = "CE1"
    CE2 = "CE2"
    CE3 = "CE3"
    NE = "NE"
    NE1 = "NE1"
    NE2 = "NE2"
    OE1 = "OE1"
    OE2 = "OE2"
    CH2 = "CH2"
    NH1 = "NH1"
    NH2 = "NH2"
    OH = "OH"
    CZ = "CZ"
    CZ2 = "CZ2"
    CZ3 = "CZ3"
    NZ = "NZ"
    OXT = "OXT"

    # rna/dna (using prime notation)
    C1_PRIME = "C1'"
    C2 = "C2"
    C2_PRIME = "C2'"
    C3_PRIME = "C3'"
    C4 = "C4"
    C4_PRIME = "C4'"
    C5 = "C5"
    C5_PRIME = "C5'"
    C6 = "C6"
    C7 = "C7"
    C8 = "C8"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    N4 = "N4"
    N6 = "N6"
    N7 = "N7"
    N9 = "N9"
    O2 = "O2"
    O2_PRIME = "O2'"
    O3_PRIME = "O3'"
    O4 = "O4"
    O4_PRIME = "O4'"
    O5_PRIME = "O5'"
    O6 = "O6"
    OP1 = "OP1"
    OP2 = "OP2"
    OP3 = "OP3"
    P = "P"

    @property
    def index(self) -> int:
        """Get the index of the atom in the combined atom list."""
        return atom_name_to_index[self]


atom_name_to_index: dict[AtomName, int] = {atom: idx for idx, atom in enumerate(AtomName)}


residue_atoms: dict[str, tuple[str, ...]] = {
    # protein
    "ALA": ("N", "CA", "C", "O", "CB"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "NE",  "CZ",  "NH1", "NH2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG",  "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG",  "OD1", "OD2"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "OE2"),
    "GLY": ("N", "CA", "C", "O"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG",  "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "CE",  "NZ"),
    "MET": ("N", "CA", "C", "O", "CB", "CG",  "SD",  "CE"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG",  "CD"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),  # noqa
    "TYR": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ",  "OH"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    "UNK": ("N", "CA", "C", "O", "CB"),

    # rna
    "A": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",  # noqa
          "N9", "C8",  "N7",  "C5",  "C6",  "N6",  "N1",  "C2",  "N3",  "C4"),
    "G": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",  # noqa
          "N9", "C8",  "N7",  "C5",  "C6",  "O6",  "N1",  "C2",  "N2",  "N3",  "C4"),
    "C": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",  # noqa
          "N1", "C2",  "O2",  "N3",  "C4",  "N4",  "C5",  "C6"),
    "U": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",  # noqa
          "N1", "C2",  "O2",  "N3",  "C4",  "O4",  "C5",  "C6"),
    "N": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"),  # noqa

    # dna
    "DA": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
           "N9", "C8",  "N7",  "C5",  "C6",  "N6",  "N1",  "C2",  "N3",  "C4"),
    "DG": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
           "N9", "C8",  "N7",  "C5",  "C6",  "O6",  "N1",  "C2",  "N2",  "N3",  "C4"),
    "DC": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
           "N1", "C2",  "O2",  "N3",  "C4",  "N4",  "C5",  "C6"),
    "DT": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
           "N1", "C2",  "O2",  "N3",  "C4",  "O4",  "C5",  "C7",  "C6"),
    "DN": ("P",  "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
}  # fmt: skip

RESIDUE_FRAME_ATOMS: dict[ResidueName, tuple[AtomName, AtomName, AtomName]] = {
    res: (AtomName.N, AtomName.CA, AtomName.C) for res in residue.PROTEIN_RESIDUES
} | {
    res: (AtomName.C1_PRIME, AtomName.C3_PRIME, AtomName.C4_PRIME)
    for res in residue.RNA_RESIDUES + residue.DNA_RESIDUES
}


# atoms for each residue
RESIDUE_ATOMS: dict[ResidueName, tuple[AtomName, ...]] = {
    ResidueName[res]: tuple(AtomName(atom) for atom in atoms)
    for res, atoms in residue_atoms.items()
}


# reference atom for each residue
REF_ATOM: dict[ResidueName, AtomName] = {
    # protein
    ResidueName.ALA: AtomName.CA,
    ResidueName.ARG: AtomName.CA,
    ResidueName.ASN: AtomName.CA,
    ResidueName.ASP: AtomName.CA,
    ResidueName.CYS: AtomName.CA,
    ResidueName.GLN: AtomName.CA,
    ResidueName.GLU: AtomName.CA,
    ResidueName.GLY: AtomName.CA,
    ResidueName.HIS: AtomName.CA,
    ResidueName.ILE: AtomName.CA,
    ResidueName.LEU: AtomName.CA,
    ResidueName.LYS: AtomName.CA,
    ResidueName.MET: AtomName.CA,
    ResidueName.PHE: AtomName.CA,
    ResidueName.PRO: AtomName.CA,
    ResidueName.SER: AtomName.CA,
    ResidueName.THR: AtomName.CA,
    ResidueName.TRP: AtomName.CA,
    ResidueName.TYR: AtomName.CA,
    ResidueName.VAL: AtomName.CA,
    ResidueName.UNK: AtomName.CA,
    # rna
    ResidueName.A: AtomName.C1_PRIME,
    ResidueName.G: AtomName.C1_PRIME,
    ResidueName.C: AtomName.C1_PRIME,
    ResidueName.U: AtomName.C1_PRIME,
    ResidueName.N: AtomName.C1_PRIME,
    # dna
    ResidueName.DA: AtomName.C1_PRIME,
    ResidueName.DG: AtomName.C1_PRIME,
    ResidueName.DC: AtomName.C1_PRIME,
    ResidueName.DT: AtomName.C1_PRIME,
    ResidueName.DN: AtomName.C1_PRIME,
}
REF_ATOM_INDEX: dict[ResidueName, int] = {
    res: RESIDUE_ATOMS[res].index(atom) for res, atom in REF_ATOM.items()
}

# pseudo-beta atom for each residue to compute distogram
PSEUDO_BETA_ATOM: dict[ResidueName, AtomName] = {
    # protein
    ResidueName.ALA: AtomName.CB,
    ResidueName.ARG: AtomName.CB,
    ResidueName.ASN: AtomName.CB,
    ResidueName.ASP: AtomName.CB,
    ResidueName.CYS: AtomName.CB,
    ResidueName.GLN: AtomName.CB,
    ResidueName.GLU: AtomName.CB,
    ResidueName.GLY: AtomName.CA,  # use CA for glycine
    ResidueName.HIS: AtomName.CB,
    ResidueName.ILE: AtomName.CB,
    ResidueName.LEU: AtomName.CB,
    ResidueName.LYS: AtomName.CB,
    ResidueName.MET: AtomName.CB,
    ResidueName.PHE: AtomName.CB,
    ResidueName.PRO: AtomName.CB,
    ResidueName.SER: AtomName.CB,
    ResidueName.THR: AtomName.CB,
    ResidueName.TRP: AtomName.CB,
    ResidueName.TYR: AtomName.CB,
    ResidueName.VAL: AtomName.CB,
    ResidueName.UNK: AtomName.CB,
    # rna
    ResidueName.A: AtomName.C4,
    ResidueName.G: AtomName.C4,
    ResidueName.C: AtomName.C2,
    ResidueName.U: AtomName.C2,
    ResidueName.N: AtomName.C1_PRIME,
    # dna
    ResidueName.DA: AtomName.C4,
    ResidueName.DG: AtomName.C4,
    ResidueName.DC: AtomName.C2,
    ResidueName.DT: AtomName.C2,
    ResidueName.DN: AtomName.C1_PRIME,
}
PSEUDO_BETA_ATOM_INDEX: dict[ResidueName, int] = {
    res: RESIDUE_ATOMS[res].index(atom) for res, atom in PSEUDO_BETA_ATOM.items()
}

# NOTE: There are no name-swap for RNA/DNA in AlphaFold3 paper and its implementation.
# Therefore, I have just implemented name-swap for protein residues here.
RESIDUE_AMBIGUOUS_ATOMS: dict[
    ResidueName, tuple[tuple[AtomName, ...], tuple[AtomName, ...]]
] = {
    ResidueName.ASP: ((AtomName.OD1,), (AtomName.OD2,)),
    ResidueName.GLU: ((AtomName.OE1,), (AtomName.OE2,)),
    ResidueName.PHE: ((AtomName.CD1, AtomName.CE1), (AtomName.CD2, AtomName.CE2)),
    ResidueName.TYR: ((AtomName.CD1, AtomName.CE1), (AtomName.CD2, AtomName.CE2)),
}

# Extended version including more residues for optimal transport
# TODO:(SeonghwanSeo) Do we have to add DNA/RNA ambiguous atoms?
RESIDUE_AMBIGUOUS_ATOMS_EXTENDED: dict[
    ResidueName, tuple[tuple[AtomName, ...], tuple[AtomName, ...]]
] = {
    ResidueName.ASP: ((AtomName.OD1,), (AtomName.OD2,)),
    ResidueName.GLU: ((AtomName.OE1,), (AtomName.OE2,)),
    ResidueName.PHE: ((AtomName.CD1, AtomName.CE1), (AtomName.CD2, AtomName.CE2)),
    ResidueName.TYR: ((AtomName.CD1, AtomName.CE1), (AtomName.CD2, AtomName.CE2)),
    ResidueName.ARG: ((AtomName.NH1,), (AtomName.NH2,)),
    ResidueName.LEU: ((AtomName.CD1,), (AtomName.CD2,)),
    ResidueName.VAL: ((AtomName.CG1,), (AtomName.CG2,)),
}


# === Function lists === #
@lru_cache(1000)
def encode_atom_name(atom_name: str) -> tuple[int, int, int, int]:
    """Encode atom name to 4-character integer tuple."""
    name = atom_name.strip().upper()
    name_int = [ord(c) - 32 for c in name]
    name_int = name_int + [0] * (4 - len(name_int))  # pad to 4 characters
    return tuple(name_int)  # type: ignore


def decode_atom_name(atom_name_chars: Sequence[int]) -> str:
    if len(atom_name_chars) != 4:
        raise ValueError("Atom name must be a sequence of 4 integers.")
    name = "".join(chr(c + 32) for c in atom_name_chars if c != 0).strip()
    return name


@lru_cache(2000)
def get_residue_atom_index(
    res_name: ResidueName,
    atom_name: AtomName,
) -> int:
    """Get the index of the atom in the residue atom list."""
    atoms = RESIDUE_ATOMS.get(res_name)
    if atoms is None:
        raise ValueError(f"Unknown residue name: {res_name}")
    try:
        return atoms.index(atom_name)
    except ValueError as e:
        raise ValueError(f"Atom {atom_name} not found in residue {res_name}") from e
