"""Coordinate conversion and sequence alignment for apo/prior structures."""

from collections.abc import Iterable
from pathlib import Path

import gemmi
import numpy as np
from atlasfold.common import residue_constants
from numpy.typing import NDArray

from kfold.constants.atom import protein_atom37
from kfold.constants.residue import protein_one_letter_to_residue_name

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
    # Gather coordinates in atom37 order and mask atoms absent from atom14.
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

    # Align nonidentical sequences while allowing missing residue stretches.
    scoring = gemmi.AlignmentScoring()
    # Missing loops may span many residues; charge per gap, not per missing residue.
    scoring.gape = 0
    alignment = gemmi.align_string_sequences(
        list(target_sequence), list(source_sequence), [], scoring
    )
    if alignment.match_count == 0:
        raise ValueError("No matching residues between the query and source structure.")

    # Convert paired alignment positions back to ungapped sequence indices.
    aligned_target = np.array(list(alignment.add_gaps(target_sequence, 1)))
    aligned_source = np.array(list(alignment.add_gaps(source_sequence, 2)))
    target_present = aligned_target != "-"
    source_present = aligned_source != "-"
    paired = target_present & source_present
    target_indices = np.cumsum(target_present, dtype=np.int64) - 1
    source_indices = np.cumsum(source_present, dtype=np.int64) - 1
    return target_indices[paired], source_indices[paired]


def write_protein_pdbs(
    models: Iterable[list[tuple[str, NDArray[np.float32]]]], path: Path
) -> None:
    """Write protein candidates from atom37 coordinates as a PDB ensemble.

    Parameters
    ----------
    models : iterable of list of tuple
        Models in candidate order, each containing (sequence, coordinates) chains.
    path : Path
        Destination PDB file. Its parent directory must exist.
    """
    # Build one PDB model per candidate, retaining only atoms with finite coordinates.
    structure = gemmi.Structure()
    for model_number, chains in enumerate(models, start=1):
        model = gemmi.Model(model_number)
        for chain_index, (sequence, coordinates) in enumerate(chains):
            chain = gemmi.Chain(chr(ord("A") + chain_index))
            for position, (aa, atom_coords) in enumerate(
                zip(sequence, coordinates, strict=True), start=1
            ):
                residue = gemmi.Residue()
                residue.name = protein_one_letter_to_residue_name[aa].name
                residue.seqid = gemmi.SeqId(position, " ")
                for name, xyz in zip(protein_atom37, atom_coords, strict=True):
                    if not np.isfinite(xyz).all():
                        continue
                    atom = gemmi.Atom()
                    atom.name = name
                    atom.element = gemmi.Element(name[0])
                    atom.pos = gemmi.Position(*xyz.tolist())
                    atom.b_iso = 0.0
                    residue.add_atom(atom)
                if len(residue):
                    chain.add_residue(residue)
            model.add_chain(chain)
        structure.add_model(model)
    structure.write_pdb(str(path))
