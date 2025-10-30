import enum
from functools import lru_cache

from .chain import ChainType

# one-letter codes for standard amino acids and nucleic acid bases
PROTEIN_AMINO_ACIDS: tuple[str, ...] = (
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "X"
)  # fmt: skip
RNA_BASES: tuple[str, ...] = ("A", "G", "C", "U", "N")
DNA_BASES: tuple[str, ...] = ("A", "G", "C", "T", "N")

PROTEIN_RESIDUES: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK"
)  # fmt: skip
RNA_RESIDUES: tuple[str, ...] = ("A", "G", "C", "U", "N")
DNA_RESIDUES: tuple[str, ...] = ("DA", "DG", "DC", "DT", "DN")


class ResidueName(enum.IntEnum):
    # protein
    ALA = enum.auto()
    ARG = enum.auto()
    ASN = enum.auto()
    ASP = enum.auto()
    CYS = enum.auto()
    GLN = enum.auto()
    GLU = enum.auto()
    GLY = enum.auto()
    HIS = enum.auto()
    ILE = enum.auto()
    LEU = enum.auto()
    LYS = enum.auto()
    MET = enum.auto()
    PHE = enum.auto()
    PRO = enum.auto()
    SER = enum.auto()
    THR = enum.auto()
    TRP = enum.auto()
    TYR = enum.auto()
    VAL = enum.auto()
    UNK = enum.auto()

    # rna
    A = enum.auto()
    G = enum.auto()
    C = enum.auto()
    U = enum.auto()
    N = enum.auto()

    # dna
    DA = enum.auto()
    DG = enum.auto()
    DC = enum.auto()
    DT = enum.auto()
    DN = enum.auto()


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


@lru_cache(100)
def map_one_letter_to_residue_name(chain_type: ChainType, one_letter: str) -> ResidueName:
    """Map one-letter code to ResidueName enum based on chain type."""
    if chain_type == ChainType.Protein:
        return protein_one_letter_to_residue_name.get(one_letter, ResidueName.UNK)
    elif chain_type == ChainType.RNA:
        return rna_one_letter_to_residue_name.get(one_letter, ResidueName.N)
    elif chain_type == ChainType.DNA:
        return dna_one_letter_to_residue_name.get(one_letter, ResidueName.DN)
    else:
        raise ValueError(f"Unsupported chain type: {chain_type}")
