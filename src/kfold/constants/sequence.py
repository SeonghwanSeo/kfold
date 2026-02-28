"""Constants for sequence embeddings."""
# HACK: this is hard-coded but it is fine since we do not expect the vocab to change.

from . import residue

# ============================================================
# Vocabulary for sequence embedding
# ============================================================

# NOTE: The vocabulary order follows the ESM-2 and ESM-3 vocabulary.
# Compared to the ESM-2 vocabulary, an additional '|' (chainbreak) token is
# included to maintain compatibility with both ESM-2 and ESM-3 for common tokens.

# fmt: off
_AMINO_ACIDS: list[str] = [
    'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
    'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
    'X', 'B', 'U', 'Z', 'O', '.', '-', '|',
]
_DNA_BASE: list[str] = [
    'A', 'G', 'C', 'T', 'N',
]
_RNA_BASE: list[str] = [
    'A', 'G', 'C', 'U', 'N',
]
_VOCAB: list[str] = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    *_AMINO_ACIDS,
    "<mask>",
    *_DNA_BASE,
    *_RNA_BASE,
]
BOS_TOKEN_INDEX = _VOCAB.index("<cls>")
EOS_TOKEN_INDEX = _VOCAB.index("<eos>")
PAD_TOKEN_INDEX = _VOCAB.index("<pad>")
UNK_TOKEN_INDEX = _VOCAB.index("<unk>")
MASK_TOKEN_INDEX = _VOCAB.index("<mask>")
# fmt: on


# prepare residue name to index mapping for one-hot encoding
RESIDUE_NAME_TO_SEQ_INDEX: dict[residue.ResidueName, int] = {}
RESIDUE_NAME_TO_SEQ_INDEX[residue.ResidueName.PAD] = _VOCAB.index("<pad>")
for res in residue.PROTEIN_RESIDUES:
    seq_token_i = 4 + _AMINO_ACIDS.index(res.one_letter)
    RESIDUE_NAME_TO_SEQ_INDEX[res] = seq_token_i
for res in residue.DNA_RESIDUES:
    seq_token_i = 4 + len(_AMINO_ACIDS) + 1 + _DNA_BASE.index(res.one_letter)
    RESIDUE_NAME_TO_SEQ_INDEX[res] = seq_token_i
for res in residue.RNA_RESIDUES:
    seq_token_i = (
        4 + len(_AMINO_ACIDS) + 1 + len(_DNA_BASE) + _RNA_BASE.index(res.one_letter)
    )
    RESIDUE_NAME_TO_SEQ_INDEX[res] = seq_token_i


def encode_protein_amino_acid(aa: str) -> int:
    """Encode a single amino acid into a token index."""
    return 4 + (_AMINO_ACIDS.index(aa) if aa in _AMINO_ACIDS else _AMINO_ACIDS.index("X"))


def encode_dna_base(base: str) -> int:
    """Encode a single DNA base into a token index."""
    return (
        4
        + len(_AMINO_ACIDS)
        + 1
        + (_DNA_BASE.index(base) if base in _DNA_BASE else _DNA_BASE.index("N"))
    )


def encode_rna_base(base: str) -> int:
    """Encode a single RNA base into a token index."""
    return (
        4
        + len(_AMINO_ACIDS)
        + 1
        + len(_DNA_BASE)
        + (_RNA_BASE.index(base) if base in _RNA_BASE else _RNA_BASE.index("N"))
    )


_PROTEIN_AMINO_ACID_TO_INDEX: dict[str, int] = {
    aa: encode_protein_amino_acid(aa) for aa in _AMINO_ACIDS
}
_PROTEIN_UNK_INDEX: int = _PROTEIN_AMINO_ACID_TO_INDEX["X"]
_DNA_BASE_TO_INDEX: dict[str, int] = {base: encode_dna_base(base) for base in _DNA_BASE}
_DNA_UNK_INDEX: int = _DNA_BASE_TO_INDEX["N"]
_RNA_BASE_TO_INDEX: dict[str, int] = {base: encode_rna_base(base) for base in _RNA_BASE}
_RNA_UNK_INDEX: int = _RNA_BASE_TO_INDEX["N"]


def encode_protein_sequence(seq: str) -> list[int]:
    """Encode a protein sequence into a list of token indices."""
    return [_PROTEIN_AMINO_ACID_TO_INDEX.get(aa, _PROTEIN_UNK_INDEX) for aa in seq]


def encode_dna_sequence(seq: str) -> list[int]:
    """Encode a DNA sequence into a list of token indices."""
    return [_DNA_BASE_TO_INDEX.get(base, _DNA_UNK_INDEX) for base in seq]


def encode_rna_sequence(seq: str) -> list[int]:
    """Encode a RNA sequence into a list of token indices."""
    return [_RNA_BASE_TO_INDEX.get(base, _RNA_UNK_INDEX) for base in seq]
