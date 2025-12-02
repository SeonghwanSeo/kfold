import pickle
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

import kfold.constants as C
from kfold.data.model_input import FoldingInput
from kfold.data.structure import TokenizedStructure

# TODO(SeonghwanSeo):
# - Consider partial crop of ligand molecules.
#   This can be addressed by revising cropping algorithm to include whole ligands
# - Implement the option which compute symmetry in cropped structure only for
#   geometric OT computation

AtomSwaps = tuple[list[int], list[int]]  # (src_indices, dst_indices)
ResidueSymmetry = list[AtomSwaps]  # List of atom swaps for a residue or molecule


def load_ligand_symmetry_dict(path: str) -> dict:
    """Create a dictionary for the ligand symmetries.

    Parameters
    ----------
    path : str
        The path to the ligand symmetries.

    Returns
    -------
    dict
        The ligand symmetries.

    """
    with Path(path).open("rb") as f:
        data: dict = pickle.load(f)  # noqa: S301

    symmetries = {}
    for key, mol in data.items():
        try:
            serialized_sym = bytes.fromhex(mol.GetProp("symmetries"))
            sym = pickle.loads(serialized_sym)  # noqa: S301
            atom_names: list[str] = [
                atom.GetProp("name").strip() for atom in mol.GetAtoms()
            ]
            symmetries[key] = (sym, atom_names)
        except Exception:  # noqa: BLE001, PERF203, S110
            pass
    return symmetries


def get_symmetries(
    f_input: FoldingInput,
    cropped_struct: TokenizedStructure,
    all_struct: TokenizedStructure,
    ligand_symmetry_dict: dict,
    max_chain_symmetries: int = 100,
) -> dict[str, Any]:
    """Get the symmetries of the complex structure.

    Parameters
    ----------
    cropped_struct : TokenizedStructure
        The cropped structure, raw data of model_input.
    all_struct : TokenizedStructure
        The full structure before cropping.
    ligand_symmetry_dict : dict
        The ligand symmetries.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the symmetries of amino acids and ligands.

    """
    features = {}

    # === Chain symmetries === #
    alt_coords_dict = get_alt_coordinates(
        cropped_struct, all_struct, max_chain_symmetries
    )
    alt_coordinates = alt_coords_dict["alt_coordinates"]
    alt_resolved_mask = alt_coords_dict["alt_resolved_mask"]
    num_chain_symmetries = len(alt_coordinates)
    # Make sparse array to dense
    dense_alt_coords = np.zeros(
        (num_chain_symmetries, f_input.num_atoms, 3), dtype=np.float32
    )
    dense_alt_mask = np.zeros((num_chain_symmetries, f_input.num_atoms), dtype=np.float32)
    pad_mask = f_input.atom.pad_mask
    num_atoms = pad_mask.sum().item()
    for i in range(num_chain_symmetries):
        dense_alt_coords[i, :num_atoms, :] = alt_coordinates[i][pad_mask]
        dense_alt_mask[i, :num_atoms] = alt_resolved_mask[i][pad_mask]
    features["alt_coordinates"] = torch.from_numpy(dense_alt_coords)  # [Nsym, Natoms, 3]
    features["alt_resolved_mask"] = torch.from_numpy(dense_alt_mask)  # [Nsym, Natoms]

    # === Residue symmetries === #
    res_syms = get_residue_symmetries(cropped_struct)
    # convert local atom index to global atom index
    res_syms_global: list[ResidueSymmetry] = []
    atom_st = 0
    for token_i in range(cropped_struct.num_tokens):
        swaps = res_syms[token_i]
        if len(swaps) == 0:
            # No symmetry for this residue
            continue
        atom_swaps_shifted = []
        for src_indices, dst_indices in swaps:
            src_global = [atom_st + idx for idx in src_indices]
            dst_global = [atom_st + idx for idx in dst_indices]
            atom_swaps_shifted.append((src_global, dst_global))
        res_syms_global.append(atom_swaps_shifted)
        atom_st += cropped_struct.token.num_atoms[token_i]
    del atom_st, res_syms
    features["residue_symmetries"] = res_syms_global

    # === Ligand and non-standard amino-acid symmetries === #
    mol_syms = get_molecule_symmetries(cropped_struct, all_struct, ligand_symmetry_dict)
    # Convert token index to global atom index
    # NOTE: Each token corresponds to one atom for ligands and non-standard amino-acids
    mol_syms_global: list[ResidueSymmetry] = []
    for swaps in mol_syms:
        if len(swaps) == 0:
            continue
        atom_swaps_shifted = []
        # compute atom offset
        token_i = swaps[0][0][0]  # Get the first source index
        atom_st = int(cropped_struct.token.num_atoms[:token_i].sum())
        atom_offset = atom_st - token_i  # Since each token has one atom
        for src_indices, dst_indices in swaps:
            src_global = [atom_offset + idx for idx in src_indices]
            dst_global = [atom_offset + idx for idx in dst_indices]
            atom_swaps_shifted.append((src_global, dst_global))
        mol_syms_global.append(atom_swaps_shifted)
    del mol_syms
    features["molecule_symmetries"] = mol_syms_global

    return features


def get_alt_coordinates(
    cropped_struct: TokenizedStructure,
    all_struct: TokenizedStructure,
    max_symmetries: int = 100,
) -> dict[str, list[np.ndarray]]:
    """Get possible alternative atom coordinates based on chain symmetries.

    Parameters
    ----------
    cropped_struct : TokenizedStructure
        The cropped structure, raw data of model_input.
    all_struct : TokenizedStructure
        The full structure before cropping.
    max_symmetries : int, optional
        The maximum number of symmetries to consider, by default 100.

    Returns
    -------
    dict[str, list[np.ndarray]]
        A dictionary containing alternative coordinates and masks.
    """

    # Get original coordinates
    original_coords = cropped_struct.atom.coords[:, :, 0, :]  # [Ntoken, 24, 3]
    original_resolved_mask = cropped_struct.atom.resolved_mask  # [Ntoken, 24]

    # Get entire coordinates as the source of symmetries
    all_coords = all_struct.atom.coords[:, :, 0, :]  # [Ntoken, 24, 3]
    all_resolved_mask = all_struct.atom.resolved_mask  # [Ntoken, 24]

    # Lists to store alternative coordinates and masks
    alter_coords_list: list[np.ndarray] = [original_coords]
    alter_resolved_mask_list: list[np.ndarray] = [original_resolved_mask]

    # === Get chain symmetries === #
    entity_chains: dict[int, list[int]] = defaultdict(list)
    for i in range(all_struct.num_chains):
        asym_id = all_struct.chain.asym_id[i]
        entity_id = all_struct.chain.entity_id[i]
        entity_chains[entity_id].append(asym_id)

    # Get symmetries of chains in the cropped structure
    # NOTE: Chain can be mapped to any other chain with the same entity_id
    # even if that chain is not in the cropped structure
    chain_symmetries: dict[int, list[int]] = {}
    # NOTE: Ensure the same number of tokens and atoms for symmetry
    chain_num_tokens: dict[int, int] = {
        all_struct.chain.asym_id[i]: all_struct.chain.num_tokens[i]
        for i in range(all_struct.num_chains)
    }
    chain_num_atoms: dict[int, int] = {
        all_struct.chain.asym_id[i]: all_struct.chain.num_atoms[i]
        for i in range(all_struct.num_chains)
    }
    for i in range(cropped_struct.num_chains):
        asym_id = cropped_struct.chain.asym_id[i]
        entity_id = cropped_struct.chain.entity_id[i]
        symmetry_chains = entity_chains[entity_id]
        # Get exactly symmetric chains
        symmetry_chains = [
            alt_asym_id
            for alt_asym_id in symmetry_chains
            if chain_num_tokens[alt_asym_id] == chain_num_tokens[asym_id]
            and chain_num_atoms[alt_asym_id] == chain_num_atoms[asym_id]
        ]

    # If there is no symmetry, return itself
    if len(chain_symmetries) == 0:
        return {
            "alt_coordinates": alter_coords_list,
            "alt_resolved_mask": alter_resolved_mask_list,
        }

    # Generate all possible swaps
    swappable_chains = sorted(chain_symmetries.keys())
    all_swaps: list[dict[int, int]] = []
    limit_count = max_symmetries * 10

    def backtrack(idx: int, current_mapping: dict[int, int], used_values: set[int]):
        if len(all_swaps) >= limit_count:
            # If we've reached the limit, stop further exploration
            return

        if idx == len(swappable_chains):
            # Complete mapping found
            mapping = {k: v for k, v in current_mapping.items() if k != v}
            all_swaps.append(mapping)
            return

        src_asym_id = swappable_chains[idx]
        possible_targets = list(chain_symmetries[src_asym_id])
        random.shuffle(possible_targets)

        for tgt_asym_id in possible_targets:
            if tgt_asym_id in used_values:
                # Ensure one-to-one mapping
                continue
            current_mapping[src_asym_id] = tgt_asym_id
            used_values.add(tgt_asym_id)

            backtrack(idx + 1, current_mapping, used_values)

            if len(all_swaps) >= limit_count:
                break

            used_values.remove(tgt_asym_id)
            del current_mapping[src_asym_id]

    backtrack(0, {}, set())

    # Exclude identity mapping (no swap), which is always included
    all_swaps = [swap for swap in all_swaps if len(swap) > 0]
    # Limit the number of symmetries to max_symmetries
    if len(all_swaps) > max_symmetries - 1:
        swaps = random.sample(all_swaps, max_symmetries - 1)
    else:
        swaps = all_swaps

    # === Get alternative atom coordinates === #
    global_token_st = all_struct.chain.token_starts
    global_token_offset: dict[int, tuple[int, int]] = {
        all_struct.chain.asym_id[i]: (global_token_st[i])
        for i in range(all_struct.num_chains)
    }

    chain_token_idcs = {}  # {asym_id: (crop_indices, global_indices)}
    crop_token_st = cropped_struct.chain.token_starts
    crop_token_end = crop_token_st + cropped_struct.chain.num_tokens
    for i in range(cropped_struct.num_chains):
        asym_id = cropped_struct.chain.asym_id[i]
        token_st, token_end = crop_token_st[i], crop_token_end[i]
        idcs_in_crop = slice(token_st, token_end)
        idcs_in_all = cropped_struct.token.token_index[idcs_in_crop]
        chain_token_idcs[asym_id] = (idcs_in_crop, idcs_in_all)

    for swap in swaps:
        alter_coords = original_coords.copy()  # [Ntoken, 24, 3]
        alter_mask = original_resolved_mask.copy()  # [Ntoken, 24]
        for src_asym_id, tgt_asym_id in swap.items():
            crop_idcs, global_idcs = chain_token_idcs[src_asym_id]

            # Map to swapped chain
            offset_src = global_token_offset[src_asym_id]
            offset_tgt = global_token_offset[tgt_asym_id]
            global_idcs_swap = global_idcs - offset_src + offset_tgt

            # In-place swap
            alter_coords[crop_idcs] = all_coords[global_idcs_swap]
            alter_mask[crop_idcs] = all_resolved_mask[global_idcs_swap]
        if alter_mask.sum() <= 4:
            # Skip if too few resolved atoms
            continue
        alter_coords_list.append(alter_coords)
        alter_resolved_mask_list.append(alter_mask)

    return {
        "alt_coordinates": alter_coords_list,
        "alt_resolved_mask": alter_resolved_mask_list,
    }


@lru_cache(100)
def get_ambigous_atom_indices(res_type: int) -> ResidueSymmetry:
    """Get the indices of ambigous atoms for a given residue type.

    Parameters
    ----------
    res_type : int
        The residue type.

    Returns
    -------
    ResidueSymmetry(=list[AtomSwaps])
        A list of atom swaps for the residue.
    """
    res_name = C.residue.residue_index_to_name[res_type]
    if res_name not in C.atom.RESIDUE_AMBIGUOUS_ATOMS:
        # If there is no ambiguous atoms, return empty list
        return []

    src_atoms, dst_atoms = C.atom.RESIDUE_AMBIGUOUS_ATOMS[res_name]
    residue_atoms = C.atom.RESIDUE_ATOMS[res_name]
    src_indices = [residue_atoms.index(atom) for atom in src_atoms]
    dst_indices = [residue_atoms.index(atom) for atom in dst_atoms]
    swap: AtomSwaps = (src_indices, dst_indices)
    return [swap]


def get_residue_symmetries(
    cropped_struct: TokenizedStructure,
) -> list[ResidueSymmetry]:
    """Get the residue symmetries for the cropped structure.

    Parameters
    ----------
    cropped : TokenizedStructure
        The cropped structure.

    Returns
    -------
    list[ResidueSymmetry]
        A list of atom swaps for each residue.
    """
    res_atom_swaps: list[ResidueSymmetry] = []
    for i in range(cropped_struct.num_tokens):
        restype = cropped_struct.token.res_type[i]
        num_atoms = cropped_struct.token.num_atoms[i]
        if num_atoms == 1:
            # NOTE (SeonghwanSeo): Currently, we set residue types of PTMs as UNK, so that
            # they are not considered for residue symmetries right now. However, if we
            # want to use its original residue type (before modification) in the future,
            # some PTMs can be assigned to residue types with ambiguous atoms (e.g., CYS),
            # even though they are already considered by `get_ligand_symmetries`. To avoid
            # this issue, we filter out non standard amino acids with `num_atoms == 1`.
            res_atom_swaps.append([])
        else:
            res_atom_swaps.append(get_ambigous_atom_indices(restype))
    return res_atom_swaps


# Type alias for residue unique identifier
ResUID = tuple[int, int]  # (asym_id, residue_index)


def get_molecule_symmetries(
    cropped_struct: TokenizedStructure,
    all_struct: TokenizedStructure,
    symmetries: dict,
) -> list[ResidueSymmetry]:
    # Compute ligand and non-standard amino-acids symmetries

    visited: set[ResUID] = set()
    mol_uids: list[ResUID] = []
    residue_mol: dict[ResUID, tuple[str, list[str]]] = {}
    mol_token_st: dict[ResUID, int] = {}

    for token_i in range(cropped_struct.num_tokens):
        if all_struct.token.is_standard[token_i]:
            # Skip standard amino acids and nucleotides
            continue

        # Get unique residue id (chain id, residue index)
        asym_id = cropped_struct.token.asym_id[token_i]
        res_idx = cropped_struct.token.residue_index[token_i]
        res_uid = (asym_id, res_idx)

        if res_uid in visited:
            # Already processed
            continue

        # Mark as visited
        visited.add(res_uid)

        # get the molecule type and indices
        res_i_global = all_struct.residue.get_global_residue_idx(asym_id, res_idx)

        # Get molecule name and atom names
        ccd_id = all_struct.residue.name[res_i_global]
        num_atoms = all_struct.residue.num_atoms[res_i_global]
        num_tokens = all_struct.residue.num_tokens[res_i_global]
        assert num_atoms == num_tokens, "Molecule token and atom number mismatch"

        # Check all tokens are belong to the cropped structure
        assert np.all(
            cropped_struct.token.asym_id[token_i : token_i + num_atoms] == asym_id
        ), "All molecule tokens should belong to the same chain after cropping."
        assert np.all(
            cropped_struct.token.residue_index[token_i : token_i + num_atoms] == res_idx
        ), "All molecule tokens should belong to the same residue after cropping."
        assert np.all(
            cropped_struct.token.num_atoms[token_i : token_i + num_atoms] == 1
        ), "Molecule tokens should have one atom per token."

        mol_atom_names: list[str] = [
            C.atom.decode_atom_name(cropped_struct.atom.ref_atom_name_chars[i, 0])
            for i in range(token_i, token_i + num_atoms)
        ]
        residue_mol[res_uid] = (ccd_id, mol_atom_names)
        mol_token_st[res_uid] = token_i

        # Record molecule uid to process
        mol_uids.append(res_uid)

    # for each molecule, get the symmetries
    mol_atom_swaps: list[list[tuple[list[int], list[int]]]] = []
    for mol_uid in mol_uids:
        ccd_id, mol_atom_names = residue_mol[mol_uid]
        if ccd_id not in symmetries:
            continue

        ccd_syms, ccd_atom_names = symmetries[ccd_id]
        atom_id_in_ccd: dict[int, int] = {
            ccd_atom_names.index(name): i for i, name in enumerate(mol_atom_names)
        }

        all_syms: list[list[int]] = []
        # Get symmetries
        for sym in ccd_syms:
            sym_dict: dict[int, int] = {}
            for i, j in enumerate(sym):
                if not (i in atom_id_in_ccd and j in atom_id_in_ccd):
                    # Some atoms are not in the cropped structure
                    raise ValueError("Some atoms are not in the cropped structure.")
                i_true = atom_id_in_ccd[i]
                j_true = atom_id_in_ccd[j]
                sym_dict[i_true] = j_true
            # Completed without break
            all_syms.append([sym_dict[i] for i in range(len(atom_id_in_ccd))])

        swaps = []
        token_st = mol_token_st[mol_uid]
        for sym in all_syms:
            src_indices = []
            dst_indices = []
            for i, j in enumerate(sym):
                if i != j:
                    src_indices.append(token_st + i)
                    dst_indices.append(token_st + j)
            if len(src_indices) > 0:
                swaps.append((src_indices, dst_indices))
        if len(swaps) > 0:
            mol_atom_swaps.append(swaps)

    return mol_atom_swaps
