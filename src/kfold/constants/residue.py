import enum
from functools import lru_cache

from .chain import ChainType

# one-letter codes for standard amino acids and nucleic acid bases
PROTEIN_AMINO_ACIDS: tuple[str, ...] = (
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "X"
)  # fmt: skip
PROTEIN_RESIDUES: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK"
)  # fmt: skip

RNA_BASES: tuple[str, ...] = ("A", "G", "C", "U", "N")
RNA_RESIDUES: tuple[str, ...] = ("A", "G", "C", "U", "N")

DNA_BASES: tuple[str, ...] = ("A", "G", "C", "T", "N")
DNA_RESIDUES: tuple[str, ...] = ("DA", "DG", "DC", "DT", "DN")


class ResidueName(enum.StrEnum):
    # pad
    PAD = "[PAD]"  # We use padding token instead of gap ("-") for MSA.

    # protein
    ALA = "ALA"
    ARG = "ARG"
    ASN = "ASN"
    ASP = "ASP"
    CYS = "CYS"
    GLN = "GLN"
    GLU = "GLU"
    GLY = "GLY"
    HIS = "HIS"
    ILE = "ILE"
    LEU = "LEU"
    LYS = "LYS"
    MET = "MET"
    PHE = "PHE"
    PRO = "PRO"
    SER = "SER"
    THR = "THR"
    TRP = "TRP"
    TYR = "TYR"
    VAL = "VAL"
    UNK = "UNK"

    # rna
    A = "A"
    G = "G"
    C = "C"
    U = "U"
    N = "N"

    # dna
    DA = "DA"
    DG = "DG"
    DC = "DC"
    DT = "DT"
    DN = "DN"

    @property
    def index(self) -> int:
        """Get the index of the residue in the enum."""
        return residue_name_to_index[self]


residue_name_to_index: dict[ResidueName, int] = {
    atom: idx for idx, atom in enumerate(ResidueName)
}
residue_index_to_name: dict[int, ResidueName] = {
    idx: atom for idx, atom in enumerate(ResidueName)
}


def get_residue_name_with_unk(residue_name: str) -> ResidueName:
    """Get the ResidueName enum, defaulting to UNK if not found."""
    try:
        return ResidueName[residue_name]
    except KeyError:
        return ResidueName.UNK


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
    if chain_type == ChainType.PROTEIN:
        return protein_one_letter_to_residue_name.get(one_letter, ResidueName.UNK)
    elif chain_type == ChainType.RNA:
        return rna_one_letter_to_residue_name.get(one_letter, ResidueName.N)
    elif chain_type == ChainType.DNA:
        return dna_one_letter_to_residue_name.get(one_letter, ResidueName.DN)
    else:
        raise ValueError(f"Unsupported chain type: {chain_type}")


@lru_cache(100)
def get_one_letter(residue_name: str | ResidueName) -> str:
    """Get the one-letter code for a given residue name."""
    if isinstance(residue_name, ResidueName):
        residue_name = residue_name.name

    assert residue_name != ResidueName.PAD, (
        "Padding residue does not have a one-letter code."
    )

    if residue_name in PROTEIN_RESIDUES:
        idx = PROTEIN_RESIDUES.index(residue_name)
        return PROTEIN_AMINO_ACIDS[idx]
    elif residue_name in RNA_RESIDUES:
        idx = RNA_RESIDUES.index(residue_name)
        return RNA_BASES[idx]
    elif residue_name in DNA_RESIDUES:
        idx = DNA_RESIDUES.index(residue_name)
        return DNA_BASES[idx]
    else:
        return "X"  # Unknown residue
