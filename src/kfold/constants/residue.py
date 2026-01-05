import enum
from functools import lru_cache
from typing import Self

from .ccd import CCD_NAME_TO_ONE_LETTER
from .chain import ChainType


class ResidueName(enum.IntEnum):
    # pad
    PAD = 0  # We use padding token instead of gap ("-") for MSA.

    # protein
    ALA = 1
    ARG = 2
    ASN = 3
    ASP = 4
    CYS = 5
    GLN = 6
    GLU = 7
    GLY = 8
    HIS = 9
    ILE = 10
    LEU = 11
    LYS = 12
    MET = 13
    PHE = 14
    PRO = 15
    SER = 16
    THR = 17
    TRP = 18
    TYR = 19
    VAL = 20
    UNK = 21

    # rna
    A = 22
    G = 23
    C = 24
    U = 25
    N = 26

    # dna
    DA = 27
    DG = 28
    DC = 29
    DT = 30
    DN = 31

    @classmethod
    def from_string(cls, name: str) -> Self:
        return cls[name]

    @classmethod
    def from_string_with_unk(cls, name: str) -> Self:
        try:
            return cls[name]
        except KeyError:
            return cls["UNK"]

    @property
    def one_letter(self) -> str:
        return RESIDUE_NAME_TO_ONE_LETTER[self]


RESIDUE_NAME_TO_ONE_LETTER: dict[ResidueName, str] = {
    ResidueName.ALA: "A",
    ResidueName.ARG: "R",
    ResidueName.ASN: "N",
    ResidueName.ASP: "D",
    ResidueName.CYS: "C",
    ResidueName.GLN: "Q",
    ResidueName.GLU: "E",
    ResidueName.GLY: "G",
    ResidueName.HIS: "H",
    ResidueName.ILE: "I",
    ResidueName.LEU: "L",
    ResidueName.LYS: "K",
    ResidueName.MET: "M",
    ResidueName.PHE: "F",
    ResidueName.PRO: "P",
    ResidueName.SER: "S",
    ResidueName.THR: "T",
    ResidueName.TRP: "W",
    ResidueName.TYR: "Y",
    ResidueName.VAL: "V",
    ResidueName.UNK: "X",
    ResidueName.A: "A",
    ResidueName.G: "G",
    ResidueName.C: "C",
    ResidueName.U: "U",
    ResidueName.N: "N",
    ResidueName.DA: "A",
    ResidueName.DG: "G",
    ResidueName.DC: "C",
    ResidueName.DT: "T",
    ResidueName.DN: "N",
}


residue_name_to_id: dict[ResidueName, int] = {res: res.value for res in ResidueName}
residue_id_to_name: dict[int, ResidueName] = {res.value: res for res in ResidueName}

# one-letter codes for standard amino acids and nucleic acid bases
PROTEIN_AMINO_ACIDS: tuple[str, ...] = (
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "X"
)  # fmt: skip
PROTEIN_RESIDUES_STR: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK"
)  # fmt: skip

# AA mapping for apo-structure prediction / sequence embedding
PROTEIN_AMINO_ACID_MAPPING: dict[str, str] = {
    "B": "D",  # Aspartic acid or Asparagine
    "Z": "E",  # Glutamic acid or Glutamine
    "J": "X",  # Leucine or Isoleucine mapped to Unknown
    "U": "C",  # Selenocysteine mapped to Cysteine
    "O": "X",  # Pyrrolysine mapped to Unknown
}

DNA_BASES: tuple[str, ...] = ("A", "G", "C", "T", "N")
DNA_RESIDUES_STR: tuple[str, ...] = ("DA", "DG", "DC", "DT", "DN")

RNA_BASES: tuple[str, ...] = ("A", "G", "C", "U", "N")
RNA_RESIDUES_STR: tuple[str, ...] = ("A", "G", "C", "U", "N")

STANDARD_RESIDUES_STR: tuple[str, ...] = (
    *PROTEIN_RESIDUES_STR,
    *RNA_RESIDUES_STR,
    *DNA_RESIDUES_STR,
)

PROTEIN_RESIDUES: tuple[ResidueName, ...] = tuple(
    ResidueName[name] for name in PROTEIN_RESIDUES_STR
)
RNA_RESIDUES: tuple[ResidueName, ...] = tuple(
    ResidueName[name] for name in RNA_RESIDUES_STR
)
DNA_RESIDUES: tuple[ResidueName, ...] = tuple(
    ResidueName[name] for name in DNA_RESIDUES_STR
)

protein_one_letter_to_residue_name: dict[str, ResidueName] = {
    "A": ResidueName.ALA,
    "R": ResidueName.ARG,
    "N": ResidueName.ASN,
    "D": ResidueName.ASP,
    "C": ResidueName.CYS,
    "Q": ResidueName.GLN,
    "E": ResidueName.GLU,
    "G": ResidueName.GLY,
    "H": ResidueName.HIS,
    "I": ResidueName.ILE,
    "L": ResidueName.LEU,
    "K": ResidueName.LYS,
    "M": ResidueName.MET,
    "F": ResidueName.PHE,
    "P": ResidueName.PRO,
    "S": ResidueName.SER,
    "T": ResidueName.THR,
    "W": ResidueName.TRP,
    "Y": ResidueName.TYR,
    "V": ResidueName.VAL,
    "X": ResidueName.UNK,
}
rna_one_letter_to_residue_name: dict[str, ResidueName] = {
    "A": ResidueName.A,
    "G": ResidueName.G,
    "C": ResidueName.C,
    "U": ResidueName.U,
    "N": ResidueName.N,
}
dna_one_letter_to_residue_name: dict[str, ResidueName] = {
    "A": ResidueName.DA,
    "G": ResidueName.DG,
    "C": ResidueName.DC,
    "T": ResidueName.DT,
    "N": ResidueName.DN,
}

unknown_residue_name: dict[ChainType, ResidueName] = {
    ChainType.PROTEIN: ResidueName.UNK,
    ChainType.RNA: ResidueName.N,
    ChainType.DNA: ResidueName.DN,
}


@lru_cache(32)
def get_residue_name_with_unk(
    residue_name: str,
    ctype: ChainType,
    map_ambiguous_to_standard: bool = False,
) -> ResidueName:
    """Get the ResidueName enum, defaulting to UNK if not found.
    NOTE: According to AF3, all non-standard residues are mapped to UNK.
    if map_ambiguous_to_standard is True, ambiguous residues (e.g., ASX, GLX)
        are mapped to their standard counterparts (e.g., ASP, GLU).
    """
    if ctype.is_nonpolymer:
        return ResidueName.UNK

    unk = unknown_residue_name[ctype]
    one_letter = convert_ccd_name_to_one_letter(residue_name, unk=unk.one_letter)

    if ctype is ChainType.PROTEIN and map_ambiguous_to_standard:
        if one_letter in PROTEIN_AMINO_ACID_MAPPING:
            one_letter = PROTEIN_AMINO_ACID_MAPPING[one_letter]

    return map_one_letter_to_residue_name(one_letter, ctype)


@lru_cache(32)
def map_one_letter_to_residue_name(one_letter: str, ctype: ChainType) -> ResidueName:
    """Map one-letter code to ResidueName enum based on chain type."""
    match ctype:
        case ChainType.PROTEIN:
            return protein_one_letter_to_residue_name.get(one_letter, ResidueName.UNK)
        case ChainType.RNA:
            return rna_one_letter_to_residue_name.get(one_letter, ResidueName.N)
        case ChainType.DNA:
            return dna_one_letter_to_residue_name.get(one_letter, ResidueName.DN)
        case _:
            raise ValueError(f"Unsupported chain type: {ctype}")


@lru_cache(32)
def convert_ccd_name_to_one_letter(ccd_name: str, unk: str) -> str:
    """Map a CCD three-letter code to a one-letter amino acid code.

    Args:
        ccd_name (str): The three-letter CCD code.

    Returns:
        str: The corresponding one-letter amino acid code.
    """
    return CCD_NAME_TO_ONE_LETTER.get(ccd_name, unk)


@lru_cache(32)
def is_modified_residue(ccd_name: str) -> bool:
    """Check if the residue is a modified residue."""
    # Modified residues are only defined for polymer types
    # Unknown residue, consider as modified
    if ccd_name not in CCD_NAME_TO_ONE_LETTER:
        return True

    if ccd_name == "MSE":
        # Selenomethionine is considered standard residue
        return False
    if ccd_name in {"ASX", "GLX"}:
        # Ambiguous residues are considered standard here
        return False

    return ccd_name not in PROTEIN_RESIDUES_STR + RNA_RESIDUES_STR + DNA_RESIDUES_STR


@lru_cache(32)
def is_ambiguous_residue(ccd_name: str, ctype: ChainType) -> bool:
    """Check if the residue is a modified residue.
    if include_ambiguous is True, ambiguous residues (e.g., ASX, GLX)
        are also considered modified.
    """
    if ctype is not ChainType.PROTEIN:
        return False
    return ccd_name in {"ASX", "GLX"}
