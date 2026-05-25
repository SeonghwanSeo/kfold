import logging
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.structure import Chain, RefStructure

ChainGroup = tuple[int, ...]
ChainSymmetry = list[ChainGroup]
ResidueSymmetry = np.ndarray | None  # [num_permutations, num_atoms]

__all__ = ["get_symmetries"]

logger = logging.getLogger(__name__)


def get_symmetries(ref_struct: RefStructure, ccd: CCD) -> dict[str, Any]:
    """Get the symmetries of the complex structure.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure.
    ccd : CCD
        The CCD database.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the symmetries of amino acids and ligands.

    """
    return {
        "chain": get_chain_symmetries(ref_struct),
        "residue": get_residue_symmetries(ref_struct, ccd),
    }


def get_chain_symmetries(ref_struct: RefStructure) -> dict[str, ChainSymmetry]:
    """Get possible groups of chain asym_ids that can be permuted among each other.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure.

    Returns
    -------
    dict[str, list[tuple[int, ...]]]
        A dictionary of groups of chain asym_ids that can be permuted among each other.
        The first entry of each group is the representative, and later entries are
        covalently linked ligands to the representative polymer chain.

        e.g.,
        Input:
          - 1, 5: same entity polymers
          - 2, 6: same entity ligands, covalently linked to 1, 5, respectively
          - 3, 4: same entity polymers
          - 7: an unique polymer
          - 8: an unique ligand
        Output:
        {
          symid1: [(1, 2), (5, 6)],
          symid2: [(3,), (4,)],
          symid3: [(7,)]
          symid4: [(8,)]
        }
    """
    # === Get available chain permutations === #
    groups: dict[int, list[Chain]] = {}  # Covalently connected chain groups
    visited: set[int] = set()
    # First, add polymer chains
    for c in ref_struct.chains:
        if c.is_polymer:
            groups[c.asym_id] = [c]
            visited.add(c.asym_id)

    # Next, add non-polymer chains to existing groups if connected
    chain_linkages: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for conn in ref_struct.connections:
        asym_id1, asym_id2 = conn.asym_id
        res_idx1, res_idx2 = conn.residue_index
        if asym_id1 == asym_id2:
            # Skip intra-bonds in multi-residue ligands (glycan)
            continue
        chain_linkages[asym_id1].append((res_idx1, asym_id2))
        chain_linkages[asym_id2].append((res_idx2, asym_id1))

    # Recursive function to add bonded chains
    # NOTE: non-polymer chains are sorted by residue index
    def add_bonded_to_group(asym_id: int, polymer_group: list):
        linkages: list[tuple[int, int]] = chain_linkages[asym_id]
        for res_idx, bonded_asym_id in sorted(linkages):  # noqa
            if bonded_asym_id not in visited:
                visited.add(bonded_asym_id)
                polymer_group.append(ref_struct.get_chain_by_asym_id(bonded_asym_id))
                add_bonded_to_group(bonded_asym_id, polymer_group)

    for poly_asym_id, poly_group in groups.items():
        add_bonded_to_group(poly_asym_id, poly_group)

    # Add remaining non-polymer chains (non-covalent) as individual groups
    for c in ref_struct.chains:
        if c.asym_id not in visited:
            groups[c.asym_id] = [c]
            visited.add(c.asym_id)

    # Collect swappable chain groups
    swappable_groups: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for _, chain_list in groups.items():
        sym_id: str = ";".join(f"{c.entity_id}({c.num_atoms})" for c in chain_list)
        group_asym_ids = tuple(c.asym_id for c in chain_list)
        swappable_groups[sym_id].append(group_asym_ids)

    return dict(swappable_groups)


@lru_cache(32)
def get_standard_residue_permutations(res_name: str) -> ResidueSymmetry:
    """Get the indices of ambiguous atoms for a given residue type."""
    res_name: C.ResidueName = C.ResidueName[res_name]
    if res_name not in C.atom.RESIDUE_AMBIGUOUS_ATOMS:
        # If there is no ambiguous atoms, return empty list
        return None
    residue_atoms = C.atom.RESIDUE_ATOMS[res_name]
    src_atoms, dst_atoms = C.atom.RESIDUE_AMBIGUOUS_ATOMS[res_name]
    src_indices = [residue_atoms.index(atom) for atom in src_atoms]
    dst_indices = [residue_atoms.index(atom) for atom in dst_atoms]
    assert set(src_indices).isdisjoint(set(dst_indices)), (
        "Overlapping ambiguous atom indices."
    )
    # original_perm = list(range(len(residue_atoms)))
    original_perm = np.arange(len(residue_atoms), dtype=np.int32)
    swap_perm = original_perm.copy()
    for s_idx, d_idx in zip(src_indices, dst_indices, strict=True):
        swap_perm[s_idx] = d_idx
        swap_perm[d_idx] = s_idx
    # Hack: there is only up to 2 symmetries per standard residue
    out = np.stack([original_perm, swap_perm], axis=0)
    return out


def get_component_permutations(
    comp: Component,
    valid_atoms: list[str] | None = None,
    max_permutations: int = 1000,
) -> ResidueSymmetry:
    """Get molecule's symmetries from ccd."""
    permutations: Sequence[list[int]] | None = comp.symmetries
    if permutations is None or len(permutations) <= 1:
        # No symmetries
        return None

    if valid_atoms is None:
        valid_atoms = list(comp.names)

    if len(valid_atoms) <= 1:
        return None

    # Limit the number of permutations
    permutations = list(permutations[:max_permutations])
    org_perm = list(range(comp.num_atoms))
    if org_perm != permutations[0]:
        # Ensure the original order is the first permutation
        if org_perm in permutations:
            permutations.remove(org_perm)
        permutations.insert(0, org_perm)
        # Limit the number of permutations again after adding original order
        permutations = permutations[:max_permutations]

    if len(valid_atoms) == comp.num_atoms:
        # All atoms are present, no need to filter
        assert list(valid_atoms) == list(comp.names), (
            "Valid atoms do not match reference molecule atoms:\n"
            f"input: {valid_atoms}\n"
            f"comp:  {comp.names}"
        )
        return np.array(permutations, dtype=np.int32)

    # Some residues (e.g., non-standard amino acids) have non-leaving atoms only.
    # Create ref-index to mol-index mapping
    name_to_index: dict[str, int] = comp.get_atom_index_map()
    ref_to_mol_map: dict[int, int] = {
        name_to_index[name]: i for i, name in enumerate(valid_atoms)
    }
    valid_atom_indices: set[int] = set(ref_to_mol_map.keys())
    all_perms: list[np.ndarray] = []
    for perm in permutations:
        # Example perm for 4-atom molecules: [0, 2, 1, 3] (swapping atom 1 and 2)
        swapped_atoms = set(i for i, j in enumerate(perm) if i != j)
        if len(swapped_atoms - valid_atom_indices) > 0:
            # There are swapped atoms not in the molecule
            continue
        sym_dict: dict[int, int] = {}
        for ref_i, ref_j in enumerate(perm):
            if ref_i != ref_j:
                i = ref_to_mol_map[ref_i]
                j = ref_to_mol_map[ref_j]
                sym_dict[i] = j
        perm = [sym_dict.get(i, i) for i in range(len(valid_atoms))]
        all_perms.append(np.array(perm, dtype=np.int32))
    if len(all_perms) <= 1:
        # No symmetries found
        return None
    return np.stack(all_perms, axis=0)


def get_residue_symmetries(
    ref_struct: RefStructure,
    ccd: CCD,
    max_permutations: int = 1000,
) -> dict[int, list[ResidueSymmetry]]:
    """Return the residue symmetries

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure.
    ccd : CCD
        The CCD database.

    Returns
    -------
    dict[list[ResidueSymmetry]]
        A dictionary of residue symmetries, where each key is a chain asym_id.
        Each symmetry is represented as a list of atom index permutations,
        or None if there is no symmetry for that residue.
    """
    _ref_comp_cache: dict[str, Component] = {}
    _ref_comp_smi_cache: dict[str, Component] = {}
    permutations: dict[int, list[ResidueSymmetry]] = {}
    for chain in ref_struct.chains:
        chain_perms: list[ResidueSymmetry] = []
        ccd_sequences: list[str] = chain.get_ccd_sequence()
        for res_i in range(chain.num_residues):
            res_idx = res_i + 1  # 1-based index
            res_name = ccd_sequences[res_i]

            # Compute residue symmetries.
            if chain.residue.is_standard[res_i]:
                # Standard residues have predefined symmetries.
                res_perms = get_standard_residue_permutations(res_name)
                res_perms = res_perms.copy() if res_perms is not None else None
            else:
                # For ligands, use the CCD component or SMILES to compute symmetries.
                smiles = chain.smiles
                if smiles is not None:
                    ref_comp = _ref_comp_smi_cache.setdefault(
                        smiles,
                        Component.from_smiles(res_name, smiles, compute_symmetry=True),
                    )
                else:
                    ref_comp = _ref_comp_cache.setdefault(res_name, ccd[res_name])

                # Get permutations between valid atoms
                atom_slice = chain.residue.get_atom_slice(res_idx)
                valid_atoms = chain.atom.name[atom_slice].tolist()
                res_perms = get_component_permutations(
                    ref_comp, valid_atoms, max_permutations
                )

            chain_perms.append(res_perms)
        permutations[chain.asym_id] = chain_perms
    return permutations
