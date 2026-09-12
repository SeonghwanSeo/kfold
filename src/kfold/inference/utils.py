"""Coordinate conversion and sequence alignment for apo/prior structures."""

import gemmi
import numpy as np
from atlasfold.common import residue_constants
from numpy.typing import NDArray

from kfold.constants.atom import protein_atom37

# One row per AtlasFold residue type, in KFold atom37 order. Absent atoms use -1.
_ATOM37_TO_ATOM14 = np.array(
    [
        [
            residue_constants.restype_atom14_order[
                residue_constants.restype_1to3[aa]
            ].get(atom, -1)
            for atom in protein_atom37
        ]
        for aa in residue_constants.restypes
    ],
    dtype=np.intp,
)


def atom14_to_atom37(
    sequence: str, coordinates: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Convert one AtlasFold (L, 14, 3) array to KFold (L, 37, 3).

    Returns FP32 coordinates with NaN for absent atoms. The lookup table is
    built once; NumPy copies all residue/atom coordinates in one indexed gather.
    """
    if coordinates.shape != (len(sequence), 14, 3):
        raise ValueError("Expected atom14 coordinates with shape (len(sequence), 14, 3).")
    residue_types = np.array(
        [residue_constants.restype_orders[aa] for aa in sequence], dtype=np.intp
    )
    atom_indices = _ATOM37_TO_ATOM14[residue_types]
    atom37 = coordinates[np.arange(len(sequence))[:, None], atom_indices].astype(
        np.float32, copy=False
    )
    atom37[atom_indices == -1] = np.nan
    return atom37


def align_sequences(
    target_sequence: str,
    source_sequence: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Map source residues onto a query sequence using gapped alignment.

    Returns
    -------
    target_indices : ndarray
        Zero-based query positions aligned to observed source residues.
    source_indices : ndarray
        Corresponding source positions. Gaps in either sequence are excluded.
    """
    if not target_sequence or not source_sequence:
        raise ValueError("Cannot align an empty query or source sequence.")
    if target_sequence == source_sequence:
        indices = np.arange(len(target_sequence), dtype=np.int64)
        return indices, indices

    scoring = gemmi.AlignmentScoring()
    # Missing loops may span many residues; charge per gap, not per missing residue.
    scoring.gape = 0
    alignment = gemmi.align_string_sequences(
        list(target_sequence), list(source_sequence), [], scoring
    )
    if alignment.match_count == 0:
        raise ValueError("No matching residues between the query and source structure.")

    aligned_target = np.array(list(alignment.add_gaps(target_sequence, 1)))
    aligned_source = np.array(list(alignment.add_gaps(source_sequence, 2)))
    target_present = aligned_target != "-"
    source_present = aligned_source != "-"
    paired = target_present & source_present
    target_indices = np.cumsum(target_present, dtype=np.int64) - 1
    source_indices = np.cumsum(source_present, dtype=np.int64) - 1
    return target_indices[paired], source_indices[paired]
