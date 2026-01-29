import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs, ETKDGv3


def sanitize_molecule(mol: Chem.Mol, allow_fail: bool = False) -> bool:
    """Sanitize the given molecule."""
    try:
        Chem.SanitizeMol(mol)
        return True
    except ValueError as e:
        if allow_fail:
            return False
        else:
            raise e


@lru_cache(maxsize=1)
def get_periodic_table() -> Chem.PeriodicTable:
    """Get the RDKit periodic table."""
    return Chem.GetPeriodicTable()


def assign_atom_names(mol: Chem.Mol, max_name_length: int = 4) -> None:
    element_counts: dict[str, int] = defaultdict(int)
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        count = element_counts.get(elem, 0) + 1
        element_counts[elem] = count
        atom_name = f"{elem}{count}"
        if len(atom_name) > max_name_length:
            raise ValueError(
                f"Atom name '{atom_name}' exceeds max length of {max_name_length}"
            )
        atom.SetProp("name", f"{elem}{count}")


def run_etkdg(
    mol: Chem.Mol,
    num_confs: int = 1,
    seed: int = 42,
    timeout: int = 30,
) -> Chem.Mol:
    params = ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1  # To ensure reproducibility
    params.timeout = timeout  # Timeout per conformer in seconds
    try:
        if num_confs == 1:
            EmbedMolecule(mol, params=params)
        else:
            EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    except (ValueError, RuntimeError) as e:
        raise e
    return mol


def compute_rdkit_conformer(
    mol: Chem.Mol,
    num_confs: int = 1,
    add_hydrogens: bool = False,
    seed: int = 42,
    timeout: int = 30,
) -> Chem.Mol:
    """Generate conformers for the given molecule using ETKDGv3 algorithm."""
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.Mol(mol)  # Create a copy to avoid modifying the original

    try:
        st = time.time()
        mol = run_etkdg(mol, num_confs, seed, int(timeout))
    except Exception as e:
        print(f"ETKDG conformer generation failed with error: {e}")
    end = time.time()
    if end - st > timeout:
        print(
            f"Warning: ETKDG conformer generation took longer than {timeout} seconds"
            f" ({end - st:.2f} seconds)"
        )
    return mol


def compute_molecule_symmetry(
    mol: Chem.Mol,
    is_leaving_atom: list[bool] | None = None,
) -> tuple[list[int], ...]:
    """Compute permutational symmetries for the given molecule."""
    if is_leaving_atom is None:
        # Assume no leaving atoms
        is_leaving_atom = [False] * mol.GetNumAtoms()
    leaving_atom_indices = {idx for idx, val in enumerate(is_leaving_atom) if val}

    # Calculate self permutations
    permutations: list[list[int]] = []
    for perm in mol.GetSubstructMatches(mol, uniquify=False):
        # Filter out permutations relating to leaving atoms
        for i in leaving_atom_indices:
            if perm[i] != i:
                break
        else:
            permutations.append(list(perm))
    if len(permutations) <= 1:
        return ()  # No symmetry found
    return tuple(permutations)


def get_conformer(mol: Chem.Mol, conf_id: int = 0) -> Chem.Conformer | None:
    """Get the specified conformer of the molecule."""
    for conf in mol.GetConformers():
        if conf.GetId() == conf_id:
            return conf
    return None


def add_conformer(
    mol: Chem.Mol, conformer: Chem.Conformer, set_chirality: bool = False
) -> int:
    """Add a conformer to the molecule."""
    id = mol.AddConformer(conformer, assignId=True)
    if set_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return id


def get_conformer_coordinates(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray | None:
    """Get the specified conformer of the molecule."""
    conf = get_conformer(mol, conf_id)
    if conf is None:
        return None
    return np.array(conf.GetPositions(), dtype=np.float32)


def add_conformer_with_coordinates(
    mol: Chem.Mol,
    coordinates: np.ndarray,
    set_chirality: bool = False,
) -> int:
    """Set the coordinates of the specified conformer of the molecule."""
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetPositions(coordinates.astype(np.float64))
    return add_conformer(mol, conf, set_chirality=set_chirality)
