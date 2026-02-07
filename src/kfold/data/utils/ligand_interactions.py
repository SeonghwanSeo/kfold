"""Ligand interaction type annotation based on PLIP-style rules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures

import kfold.constants as C

RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=1)
def _get_feature_factory() -> ChemicalFeatures.MolChemicalFeatureFactory:
    """Build and cache RDKit's default feature factory."""
    fdef_path = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return ChemicalFeatures.BuildFeatureFactory(str(fdef_path))


def _ring_info_initialized(ring_info: Chem.rdchem.RingInfo) -> bool:
    """Return True when RDKit ring information is ready to query."""
    if hasattr(ring_info, "IsInitialized"):
        return ring_info.IsInitialized()
    try:
        ring_info.NumRings()
    except Exception:
        return False
    return True


def _ensure_ring_info(mol: Chem.Mol) -> bool:
    """Ensure ring information is available, with robust fallbacks."""
    ring_info = mol.GetRingInfo()
    if ring_info is not None and _ring_info_initialized(ring_info):
        return True
    try:
        Chem.FastFindRings(mol)
    except Exception:
        try:
            Chem.GetSymmSSSR(mol)
        except Exception:
            return False
    ring_info = mol.GetRingInfo()
    return ring_info is not None and _ring_info_initialized(ring_info)


def _try_sanitize_for_features(mol: Chem.Mol) -> bool:
    """Best-effort sanitization that tolerates RDKit version differences."""
    try:
        result = Chem.SanitizeMol(mol, catchErrors=True)
    except TypeError:
        try:
            Chem.SanitizeMol(mol)
            return True
        except Exception:
            return False
    except Exception:
        return False
    if result is None:
        return True
    try:
        return int(result) == 0
    except Exception:
        return False


def compute_ligand_interaction_types(mol: Chem.Mol) -> np.ndarray:
    """Compute per-atom interaction types for ligand atoms.

    The rules are adapted from PLIP's ligand preparation logic:
    - hydrophobic carbons: carbon atoms with only carbon/hydrogen neighbors
    - HBD/HBA: RDKit feature factory (SMARTS-based) donors and acceptors
    - pi-system: aromatic atoms
    - charged groups: PLIP functional group heuristics + formal charges
    """
    num_atoms = mol.GetNumAtoms()
    interaction_type = np.zeros((num_atoms, C.NUM_INTERACTION_TYPES), dtype=np.int8)
    if num_atoms == 0:
        return interaction_type
    if num_atoms == 1:
        # For single-atom ligands (e.g., ions), formal charge is the most
        # reliable interaction signal.
        _add_formal_charges(mol, interaction_type)
        return interaction_type

    # Ring info is required for aromatic/pi-system detection. Some molecules
    # arrive partially sanitized, so we try lightweight ring detection first
    # and fall back to sanitization when needed.
    if not _ensure_ring_info(mol):
        if not _try_sanitize_for_features(mol):
            _add_formal_charges(mol, interaction_type)
            return interaction_type
        if not _ensure_ring_info(mol):
            _add_formal_charges(mol, interaction_type)
            return interaction_type

    _add_hydrophobic_atoms(mol, interaction_type)
    _add_hbond_features(mol, interaction_type)
    _add_aromatic_atoms(mol, interaction_type)
    _add_plip_charged_groups(mol, interaction_type)
    _add_formal_charges(mol, interaction_type)

    return interaction_type


def _add_hydrophobic_atoms(mol: Chem.Mol, interaction_type: np.ndarray) -> None:
    """Mark hydrophobic carbons following PLIP-style heuristics."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        neighbor_nums = [nbr.GetAtomicNum() for nbr in atom.GetNeighbors()]
        if all(num in (1, 6) for num in neighbor_nums):
            interaction_type[atom.GetIdx(), C.InteractionType.HI] = 1


def _add_hbond_features(mol: Chem.Mol, interaction_type: np.ndarray) -> None:
    """Annotate hydrogen bond donors/acceptors via RDKit features."""
    factory = _get_feature_factory()
    for feature in factory.GetFeaturesForMol(mol):
        family = feature.GetFamily()
        if family == "Donor":
            for atom_idx in feature.GetAtomIds():
                interaction_type[atom_idx, C.InteractionType.HBD] = 1
        elif family == "Acceptor":
            for atom_idx in feature.GetAtomIds():
                interaction_type[atom_idx, C.InteractionType.HBA] = 1


def _add_aromatic_atoms(mol: Chem.Mol, interaction_type: np.ndarray) -> None:
    """Mark aromatic atoms as pi-systems."""
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            interaction_type[atom.GetIdx(), C.InteractionType.PP] = 1


def _add_plip_charged_groups(mol: Chem.Mol, interaction_type: np.ndarray) -> None:
    """Detect charged groups using PLIP-inspired functional heuristics."""
    pos_atoms: set[int] = set()
    neg_atoms: set[int] = set()

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        neighbors = list(atom.GetNeighbors())
        neighbor_nums = [nbr.GetAtomicNum() for nbr in neighbors]
        atomic_num = atom.GetAtomicNum()

        if atomic_num == 7:
            if atom.GetDegree() == 4:
                pos_atoms.add(atom_idx)
            elif (
                atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3
                and atom.GetDegree() >= 3
            ):
                pos_atoms.add(atom_idx)
        elif atomic_num == 16:
            if atom.GetDegree() == 3:
                pos_atoms.add(atom_idx)
            if neighbor_nums.count(8) == 3:
                neg_atoms.add(atom_idx)
                neg_atoms.update(_neighbor_indices(neighbors, 8))
            elif neighbor_nums.count(8) == 4:
                neg_atoms.add(atom_idx)
                neg_atoms.update(_neighbor_indices(neighbors, 8))
        elif atomic_num == 15:
            if neighbor_nums and set(neighbor_nums) == {8}:
                neg_atoms.add(atom_idx)
                neg_atoms.update(_neighbor_indices(neighbors, 8))
        elif atomic_num == 6:
            if neighbor_nums.count(8) == 2 and neighbor_nums.count(6) == 1:
                neg_atoms.update(_neighbor_indices(neighbors, 8))
            elif neighbor_nums.count(7) == 3 and len(neighbor_nums) == 3:
                n_neighbors = [nbr for nbr in neighbors if nbr.GetAtomicNum() == 7]
                n_degrees = [nbr.GetDegree() for nbr in n_neighbors]
                if n_degrees and min(n_degrees) == 1:
                    pos_atoms.update(nbr.GetIdx() for nbr in n_neighbors)

    _mark_positive(interaction_type, pos_atoms)
    _mark_negative(interaction_type, neg_atoms)


def _add_formal_charges(mol: Chem.Mol, interaction_type: np.ndarray) -> None:
    """Apply formal charges as a final, conservative signal."""
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            _mark_positive(interaction_type, [atom.GetIdx()])
        elif charge < 0:
            _mark_negative(interaction_type, [atom.GetIdx()])


def _mark_positive(
    interaction_type: np.ndarray, atom_indices: list[int] | set[int]
) -> None:
    """Mark atoms as salt-bridge cations and pi-cation partners."""
    for atom_idx in atom_indices:
        interaction_type[atom_idx, C.InteractionType.SBC] = 1
        interaction_type[atom_idx, C.InteractionType.PC] = 1


def _mark_negative(
    interaction_type: np.ndarray, atom_indices: list[int] | set[int]
) -> None:
    """Mark atoms as salt-bridge anions."""
    for atom_idx in atom_indices:
        interaction_type[atom_idx, C.InteractionType.SBA] = 1


def _neighbor_indices(neighbors: list[Chem.Atom], atomic_num: int) -> list[int]:
    """Return neighbor atom indices matching a given atomic number."""
    return [nbr.GetIdx() for nbr in neighbors if nbr.GetAtomicNum() == atomic_num]
