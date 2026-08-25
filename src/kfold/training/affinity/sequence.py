"""One canonical protein-sequence contract for affinity artifacts and inputs."""

from __future__ import annotations

import kfold.constants as C

_MODEL_PROTEIN_RESIDUES = frozenset(C.residue.PROTEIN_AMINO_ACIDS_SET)


def normalize_protein_sequence(sequence: str) -> str:
    """Normalize source text to K-Fold's one-letter protein alphabet.

    Whitespace is FASTA formatting and is removed.  K-Fold's approved
    ambiguous-residue mapping is applied first (including selenocysteine
    $U\rightarrow C$); every remaining non-model code stays positionally as
    ``X`` so cache keys, apo coordinates, and tokenizer inputs agree.
    """
    letters = "".join(sequence.split()).upper()
    normalized = "".join(
        C.residue.PROTEIN_AMINO_ACID_MAPPING.get(residue, residue)
        for residue in letters
    )
    normalized = "".join(
        residue if residue in _MODEL_PROTEIN_RESIDUES else "X"
        for residue in normalized
    )
    if not normalized:
        raise ValueError("Affinity input requires a non-empty protein sequence.")
    return normalized
