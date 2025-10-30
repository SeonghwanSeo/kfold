import enum

from .residue import ResidueName

protein_atom37: tuple[str, ...] = (
    "N",   "CA",  "C",   "CB",  "O",   "CG",  "CG1", "CG2", "OG",  "OG1",
    "SG",  "CD",  "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",  "CE",
    "CE1", "CE2", "CE3", "NE",  "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH",  "CZ",  "CZ2", "CZ3", "NZ",  "OXT"
)  # fmt: skip

nucleic_atom29: tuple[str, ...] = (
    "C1'", "C2",  "C2'", "C3'", "C4",  "C4'", "C5",  "C5'", "C6",  "C7",
    "C8",  "N1",  "N2",  "N3",  "N4",  "N6",  "N7",  "N9",  "OP3", "O2",
    "O2'", "O3'", "O4",  "O4'", "O5'", "O6",  "OP1", "OP2", "P"
)  # fmt: skip


class AtomName(enum.IntEnum):
    """This is also used for indexing atom types"""

    # protein
    N = enum.auto()
    CA = enum.auto()
    C = enum.auto()
    CB = enum.auto()
    O = enum.auto()  # noqa
    CG = enum.auto()
    CG1 = enum.auto()
    CG2 = enum.auto()
    OG = enum.auto()
    OG1 = enum.auto()
    SG = enum.auto()
    CD = enum.auto()
    CD1 = enum.auto()
    CD2 = enum.auto()
    ND1 = enum.auto()
    ND2 = enum.auto()
    OD1 = enum.auto()
    OD2 = enum.auto()
    SD = enum.auto()
    CE = enum.auto()
    CE1 = enum.auto()
    CE2 = enum.auto()
    CE3 = enum.auto()
    NE = enum.auto()
    NE1 = enum.auto()
    NE2 = enum.auto()
    OE1 = enum.auto()
    OE2 = enum.auto()
    CH2 = enum.auto()
    NH1 = enum.auto()
    NH2 = enum.auto()
    OH = enum.auto()
    CZ = enum.auto()
    CZ2 = enum.auto()
    CZ3 = enum.auto()
    NZ = enum.auto()
    OXT = enum.auto()

    # rna/dna
    C1P = enum.auto()
    C2 = enum.auto()
    C2P = enum.auto()
    C3P = enum.auto()
    C4 = enum.auto()
    C4P = enum.auto()
    C5 = enum.auto()
    C5P = enum.auto()
    C6 = enum.auto()
    C7 = enum.auto()
    C8 = enum.auto()
    N1 = enum.auto()
    N2 = enum.auto()
    N3 = enum.auto()
    N4 = enum.auto()
    N6 = enum.auto()
    N7 = enum.auto()
    N9 = enum.auto()
    O2 = enum.auto()
    O2P = enum.auto()
    O3P = enum.auto()
    O4 = enum.auto()
    O4P = enum.auto()
    O5P = enum.auto()
    O6 = enum.auto()
    OP1 = enum.auto()
    OP2 = enum.auto()
    OP3 = enum.auto()
    P = enum.auto()


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

# atoms for each residue
RESIDUE_ATOMS: dict[ResidueName, tuple[AtomName, ...]] = {
    ResidueName[res]: tuple(AtomName[atom] for atom in atoms)
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
    ResidueName.A: AtomName.C1P,
    ResidueName.G: AtomName.C1P,
    ResidueName.C: AtomName.C1P,
    ResidueName.U: AtomName.C1P,
    ResidueName.N: AtomName.C1P,
    # dna
    ResidueName.DA: AtomName.C1P,
    ResidueName.DG: AtomName.C1P,
    ResidueName.DC: AtomName.C1P,
    ResidueName.DT: AtomName.C1P,
    ResidueName.DN: AtomName.C1P,
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
    ResidueName.N: AtomName.C1P,
    # dna
    ResidueName.DA: AtomName.C4,
    ResidueName.DG: AtomName.C4,
    ResidueName.DC: AtomName.C2,
    ResidueName.DT: AtomName.C2,
    ResidueName.DN: AtomName.C1P,
}
