"""Symmetry-related utilities for validation metrics."""

import itertools
import math
from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.structure import Chain, RefStructure

AtomPermutation = list[int]
ChainPermutation = list[int]
ResidueSymmetry = list[AtomPermutation] | None


def get_symmetries(
    ref_struct: RefStructure,
    ccd: CCD,
    max_chain_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Get the symmetries of the complex structure.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure.
    ccd : CCD
        The CCD database.
    max_chain_permutations : int, optional
        The maximum number of chain permutations to consider, by default 1000.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the symmetries of amino acids and ligands.

    """
    rng = rng or np.random.default_rng()
    return {
        "chain": get_chain_permutations(ref_struct, rng, max_chain_permutations),
        "residue": get_residue_symmetries(ref_struct, ccd),
    }


def get_chain_permutations(
    ref_struct: RefStructure,
    rng: np.random.Generator,
    max_permutations: int = 1000,
) -> list[list[int]]:
    """Get possible alternative atom coordinates based on chain symmetries.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure.
    rng : np.random.Generator
        The random number generator.
    max_permutations : int, optional
        The maximum number of chain permutations to consider, by default 1000.

    Returns
    -------
    list[list[int]]
        A list of chain asym_ids.
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

    # Collect swappable chains
    swappable_bucket: dict[str, list[int]] = defaultdict(list)  # bucket_id to asym_ids
    for asym_id, g in groups.items():
        bucket_id: str = ":".join(f"{c.entity_id}-{c.num_atoms}" for c in g)
        swappable_bucket[bucket_id].append(asym_id)

    # Filter out non-swappable groups
    swappable_bucket = {k: v for k, v in swappable_bucket.items() if len(v) > 1}
    if len(swappable_bucket) == 0:
        # No symmetries
        return [[c.asym_id for c in ref_struct.chains]]  # Identity only

    # === Create Permutations === #
    # Prepare data for permutation generation
    sorted_bucket_keys = sorted(swappable_bucket.keys())
    bucket_anchors_list = [swappable_bucket[k] for k in sorted_bucket_keys]

    # Calculate total complexity
    total_perms = 1
    for anchors in bucket_anchors_list:
        total_perms *= math.factorial(len(anchors))

    anchor_mappings: list[dict[int, int]] = []

    if total_perms <= max_permutations:
        # Exhaustive Search
        # Generate all permutations for each bucket
        per_bucket_perms = [
            list(itertools.permutations(anchors)) for anchors in bucket_anchors_list
        ]
        # Combine them
        for combination in itertools.product(*per_bucket_perms):
            mapping: dict[int, int] = {}
            for original_anchors, permuted_anchors in zip(
                bucket_anchors_list, combination, strict=True
            ):
                for old_anchor, new_anchor in zip(
                    original_anchors, permuted_anchors, strict=True
                ):
                    mapping[old_anchor] = new_anchor
            anchor_mappings.append(mapping)
    else:
        # Random Sampling
        # 1. Always include identity
        identity_map = {}
        for anchors in bucket_anchors_list:
            for a in anchors:
                identity_map[a] = a
        anchor_mappings.append(identity_map)

        # 2. Track unique permutations via signature
        seen_sigs = set()
        # Signature is tuple of values sorted by keys
        all_swappable_keys = sorted(identity_map.keys())
        sig = tuple(identity_map[k] for k in all_swappable_keys)
        seen_sigs.add(sig)

        # 3. Sample
        for _ in range(max_permutations * 10):
            mapping = {}
            for anchors in bucket_anchors_list:
                perm = list(anchors)
                rng.shuffle(perm)
                for old_anchor, new_anchor in zip(anchors, perm, strict=True):
                    mapping[old_anchor] = new_anchor

            sig = tuple(mapping[k] for k in all_swappable_keys)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                anchor_mappings.append(mapping)
            if len(anchor_mappings) >= max_permutations:
                break

    # === Expand to full chain mappings === #
    final_results: list[dict[int, int]] = []

    for anchor_map in anchor_mappings:
        full_mapping: dict[int, int] = {}
        # Iterate over all defined groups (anchors)
        for anchor_id, chain_list in groups.items():
            # If anchor is swappable, get its target; otherwise it maps to itself
            target_anchor_id = anchor_map.get(anchor_id, anchor_id)
            target_chain_list = groups[target_anchor_id]
            # Map each chain in the group to the corresponding chain in the target group
            for old_c, new_c in zip(chain_list, target_chain_list, strict=True):
                old_id, new_id = old_c.asym_id, new_c.asym_id
                if old_id != new_id:
                    full_mapping[old_id] = new_id
        if len(full_mapping) > 0:
            final_results.append(full_mapping)
    final_results.sort(key=lambda x: sorted(x.items()))
    final_results = [{}] + final_results  # Always include identity mapping
    return [
        [mapping.get(c.asym_id, c.asym_id) for c in ref_struct.chains]
        for mapping in final_results
    ]


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
    original_perm = list(range(len(residue_atoms)))
    swap_perm = original_perm.copy()
    for s_idx, d_idx in zip(src_indices, dst_indices, strict=True):
        swap_perm[s_idx] = d_idx
        swap_perm[d_idx] = s_idx
    # Hack: there is only up to 2 symmetries per standard residue
    return [original_perm, swap_perm]


def get_component_permutations(
    comp: Component, valid_atoms: list[str]
) -> ResidueSymmetry:
    """Get molecule's symmetries from ccd."""
    symmetries = comp.symmetries
    if symmetries is None or len(symmetries) <= 1:
        # No symmetries
        return None
    if len(valid_atoms) == comp.num_atoms:
        # All atoms are present, no need to filter
        assert list(valid_atoms) == list(comp.names), (
            "Valid atoms do not match reference molecule atoms:\n"
            f"input: {valid_atoms}\n"
            f"comp:  {comp.names}"
        )
        return list(symmetries)
    # Some residues (e.g., non-standard amino acids) have non-leaving atoms only.
    # Create ref-index to mol-index mapping
    name_to_index: dict[str, int] = comp.get_atom_index_map()
    ref_to_mol_map: dict[int, int] = {
        name_to_index[name]: i for i, name in enumerate(valid_atoms)
    }
    valid_atom_indices: set[int] = set(ref_to_mol_map.keys())
    all_perms: list[AtomPermutation] = []
    for perm in symmetries:
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
        all_perms.append([sym_dict.get(i, i) for i in range(len(valid_atoms))])
    if len(all_perms) <= 1:
        # No symmetries found
        return None
    return all_perms


def get_residue_symmetries(
    ref_struct: RefStructure,
    ccd: CCD,
) -> list[ResidueSymmetry]:
    """Return the residue symmetries

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure.
    ccd : CCD
        The CCD database.

    Returns
    -------
    list[ResidueSymmetry]
        A list of residue symmetries in the complex.
    """
    component_cache: dict[str, Component] = {}
    permutations: list[ResidueSymmetry] = []
    for chain in ref_struct.chains:
        ccd_sequences: list[str] = chain.get_ccd_sequence()
        for res_i in range(chain.num_residues):
            res_idx = res_i + 1  # 1-based index
            res_name = ccd_sequences[res_i]
            if chain.residue.is_standard[res_i]:
                res_perms = get_standard_residue_permutations(res_name)
            else:
                # Get valid atoms in the residue
                atom_slice = chain.residue.get_atom_slice(res_idx)
                valid_atoms = chain.atom.name[atom_slice].tolist()
                if len(valid_atoms) <= 1:
                    # No symmetry for single-atom residues (e.g., ions)
                    res_perms = None
                else:
                    ref_mol = component_cache.setdefault(res_name, ccd[res_name])
                    res_perms = get_component_permutations(ref_mol, valid_atoms)
            permutations.append(res_perms)
    return permutations
