from typing import Self
import enum
from functools import lru_cache

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


residue_name_to_id: dict[ResidueName, int] = {res: res.value for res in ResidueName}
residue_id_to_name: dict[int, ResidueName] = {res.value: res for res in ResidueName}

# one-letter codes for standard amino acids and nucleic acid bases
PROTEIN_AMINO_ACIDS: tuple[str, ...] = (
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "X"
)  # fmt: skip
PROTEIN_AMINO_ACIDS_EXTENDED: tuple[str, ...] = (
    *PROTEIN_AMINO_ACIDS, "B", "Z", "J", "U", "O"
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
protein_residue_name_to_one_letter: dict[ResidueName, str] = {
    v: k for k, v in protein_one_letter_to_residue_name.items()
}
rna_residue_name_to_one_letter: dict[ResidueName, str] = {
    v: k for k, v in rna_one_letter_to_residue_name.items()
}
dna_residue_name_to_one_letter: dict[ResidueName, str] = {
    v: k for k, v in dna_one_letter_to_residue_name.items()
}


@lru_cache
def get_residue_name_with_unk(residue_name: str, ctype: ChainType) -> ResidueName:
    """Get the ResidueName enum, defaulting to UNK if not found.
    NOTE: According to AF3, all non-standard residues are mapped to UNK.
    """
    match ctype:
        case ChainType.PROTEIN:
            if residue_name == "MSE":
                residue_name = "MET"
            if residue_name not in PROTEIN_RESIDUES_STR:
                return ResidueName.UNK
            return ResidueName[residue_name]
        case ChainType.RNA:
            if residue_name not in RNA_RESIDUES_STR:
                return ResidueName.N
            return ResidueName[residue_name]
        case ChainType.DNA:
            if residue_name not in DNA_RESIDUES_STR:
                return ResidueName.DN
            return ResidueName[residue_name]
        case _:
            # default to UNK for ligand
            return ResidueName.UNK


@lru_cache
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


@lru_cache
def get_one_letter(residue_name: str | ResidueName, unk: str) -> str:
    """Get the one-letter code for a given residue name."""
    assert residue_name != ResidueName.PAD, (
        "Padding residue does not have a one-letter code."
    )
    if isinstance(residue_name, ResidueName):
        residue_name = residue_name.name

    return convert_ccd_name_to_one_letter(residue_name, unk=unk)


@lru_cache(maxsize=128)
def convert_ccd_name_to_one_letter(ccd_code: str, unk: str) -> str:
    """Map a CCD three-letter code to a one-letter amino acid code.

    Args:
        ccd_code (str): The three-letter CCD code.

    Returns:
        str: The corresponding one-letter amino acid code.
    """
    return CCD_NAME_TO_ONE_LETTER.get(ccd_code, unk)
