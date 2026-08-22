"""Exact-SMILES reuse of existing ligand conformers from the K-Fold CCD."""

from __future__ import annotations

import hashlib

import numpy as np
from rdkit import Chem

from kfold.data.types.ccd import Component

from .ligand import LigandApo, ligand_etkdg_seed

CCD_CONFORMER_PRIORITY = ("ideal", "model", "etkdg-cached")


def canonical_component_smiles(component: Component) -> str:
    """Return the exact RDKit canonical identity used for source ligand lookup."""
    return Chem.MolToSmiles(component.mol, canonical=True, isomericSmiles=True)


def existing_ccd_conformer(
    component: Component, canonical_smiles: str
) -> tuple[np.ndarray, str] | None:
    """Select a finite existing CCD conformer without generating ETKDG."""
    rng = np.random.default_rng(ligand_etkdg_seed(canonical_smiles))
    for conformer_type in CCD_CONFORMER_PRIORITY:
        coords = component.get_conformer(conformer_type, rng)
        if coords is not None and np.isfinite(coords).all():
            return np.asarray(coords, dtype=np.float32), conformer_type
    return None


def ccd_ligand_apo(
    *,
    canonical_smiles: str,
    ccd_code: str,
    component: Component,
) -> LigandApo | None:
    """Map an existing CCD conformer into source-SMILES atom order.

    A canonical-SMILES match is necessary but not enough because the PDB CCD
    atom order may differ from the source SMILES atom order used by K-Fold's
    ligand chain. This function requires exact graph identity and reorders the
    CCD coordinates into that source order before caching.
    """
    selected = existing_ccd_conformer(component, canonical_smiles)
    if selected is None:
        return None
    ccd_coords, conformer_type = selected
    source_component = Component.from_smiles("LIG", canonical_smiles)
    source_mol = source_component.mol
    ccd_mol = component.mol
    if source_mol.GetNumAtoms() != ccd_mol.GetNumAtoms():
        return None
    matches = source_mol.GetSubstructMatches(
        ccd_mol,
        uniquify=True,
        useChirality=True,
    )
    if not matches:
        return None
    match = min(matches)
    if len(match) != len(ccd_coords):
        return None
    source_coords = np.empty((source_component.num_atoms, 3), dtype=np.float32)
    source_coords[np.asarray(match, dtype=np.int64)] = ccd_coords
    if not np.isfinite(source_coords).all():
        return None
    return LigandApo(
        canonical_smiles=canonical_smiles,
        coords=source_coords,
        structure_sha256=hashlib.sha256(source_coords.tobytes()).hexdigest(),
        source=f"ccd_{conformer_type.replace('-', '_')}",
        source_id=ccd_code,
    )
