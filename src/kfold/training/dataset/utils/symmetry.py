from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import torch

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.tokenized import TokenizedStructure

# TODO(SeonghwanSeo):
# - Consider partial crop of ligand molecules.
#   This can be addressed by revising cropping algorithm to include whole ligands
# - Implement the option which compute symmetry in cropped structure only for
#   geometric OT computation

AtomSwaps = tuple[list[int], list[int]]  # (src_indices, dst_indices)
ResidueSymmetry = list[AtomSwaps]  # List of atom swaps for a residue or molecule


def get_symmetries(
    f_input: FoldingInput,
    cropped_struct: TokenizedStructure,
    all_struct: TokenizedStructure,
    ccd: CCD,
    max_chain_symmetries: int = 100,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Get the symmetries of the complex structure.

    Parameters
    ----------
    f_input : FoldingInput
        The model input data.
    cropped_struct : TokenizedStructure
        The cropped structure, raw data of model_input.
    all_struct : TokenizedStructure
        The full structure before cropping.
    ccd : CCD
        The CCD database.
    max_chain_symmetries : int, optional
        The maximum number of chain symmetries to consider, by default 100.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the symmetries of amino acids and ligands.

    """
    rng = rng or np.random.default_rng()

    symmetry_info = {}

    # === Chain symmetries === #
    alt_coords_dict = get_alt_coordinates(
        cropped_struct, all_struct, rng, max_chain_symmetries
    )
    alt_coordinates = alt_coords_dict["alt_coordinates"]
    alt_resolved_mask = alt_coords_dict["alt_resolved_mask"]
    num_chain_symmetries = len(alt_coordinates)

    # Make sparse array to dense
    num_atoms = f_input.num_atoms
    pad_mask = cropped_struct.atom.pad_mask
    dense_alt_coords = np.zeros((num_chain_symmetries, num_atoms, 3), dtype=np.float32)
    dense_alt_mask = np.zeros((num_chain_symmetries, num_atoms), dtype=np.bool)
    assert pad_mask.sum() == num_atoms, "Number of atoms mismatch after featurization"

    for i in range(num_chain_symmetries):
        dense_alt_coords[i] = alt_coordinates[i][pad_mask]
        dense_alt_mask[i] = alt_resolved_mask[i][pad_mask]
    symmetry_info["alt_coordinates"] = torch.from_numpy(
        dense_alt_coords
    )  # [Nsym, Natoms, 3]
    symmetry_info["alt_resolved_mask"] = torch.from_numpy(
        dense_alt_mask
    )  # [Nsym, Natoms]

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
        atom_st += int(cropped_struct.token.num_atoms[token_i])
    del atom_st, res_syms
    symmetry_info["residue_symmetries"] = res_syms_global

    # === Ligand and non-standard amino-acid symmetries === #
    mol_syms = get_molecule_symmetries(cropped_struct, ccd)
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
    symmetry_info["molecule_symmetries"] = mol_syms_global

    return symmetry_info


def get_alt_coordinates(
    cropped_struct: TokenizedStructure,
    all_struct: TokenizedStructure,
    rng: np.random.Generator,
    max_symmetries: int = 100,
) -> dict[str, list[np.ndarray]]:
    """Get possible alternative atom coordinates based on chain symmetries.

    Parameters
    ----------
    cropped_struct : TokenizedStructure
        The cropped structure, raw data of model_input.
    all_struct : TokenizedStructure
        The full structure before cropping.
    rng : np.random.Generator
        The random number generator.
    max_symmetries : int, optional
        The maximum number of symmetries to consider, by default 100.

    Returns
    -------
    dict[str, list[np.ndarray]]
        A dictionary containing alternative coordinates and masks.
    """

    # Get original coordinates
    original_coords = cropped_struct.atom.coords  # [Ntoken, 24, 3]
    original_resolved_mask = cropped_struct.atom.resolved_mask  # [Ntoken, 24]

    # Get entire coordinates as the source of symmetries
    all_coords = all_struct.atom.coords  # [Ntoken, 24, 3]
    all_resolved_mask = all_struct.atom.resolved_mask  # [Ntoken, 24]

    # === Get chain symmetries === #
    entity_chains: dict[int, list[int]] = defaultdict(list)
    for i in range(all_struct.num_chains):
        asym_id = int(all_struct.chain.asym_id[i])
        entity_id = int(all_struct.chain.entity_id[i])
        entity_chains[entity_id].append(asym_id)

    # Get symmetries of chains in the cropped structure
    # NOTE: Chain can be mapped to any other chain with the same entity_id
    # even if that chain is not in the cropped structure
    chain_symmetries: dict[int, list[int]] = {}
    # NOTE: Ensure the same number of tokens and atoms for symmetry
    chain_num_tokens: dict[int, int] = {
        int(all_struct.chain.asym_id[i]): int(all_struct.chain.num_tokens[i])
        for i in range(all_struct.num_chains)
    }
    chain_num_atoms: dict[int, int] = {
        int(all_struct.chain.asym_id[i]): int(all_struct.chain.num_atoms[i])
        for i in range(all_struct.num_chains)
    }
    for i in range(cropped_struct.num_chains):
        asym_id = int(cropped_struct.chain.asym_id[i])
        entity_id = int(cropped_struct.chain.entity_id[i])
        symmetry_chains = entity_chains[entity_id]
        # Get exactly symmetric chains
        symmetry_chains = [
            alt_asym_id
            for alt_asym_id in symmetry_chains
            if chain_num_tokens[alt_asym_id] == chain_num_tokens[asym_id]
            and chain_num_atoms[alt_asym_id] == chain_num_atoms[asym_id]
        ]
        if len(symmetry_chains) > 1:
            chain_symmetries[asym_id] = symmetry_chains

    # If there is no symmetry, return itself
    if len(chain_symmetries) == 0:
        return {
            "alt_coordinates": [original_coords],
            "alt_resolved_mask": [original_resolved_mask],
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
        rng.shuffle(possible_targets)

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
        sample_swap_indices = rng.choice(
            len(all_swaps), size=max_symmetries - 1, replace=False
        )
        swaps = [all_swaps[i] for i in sample_swap_indices]
    else:
        swaps = all_swaps

    # === Get alternative atom coordinates === #
    global_token_st = all_struct.chain.token_start
    global_token_offset: dict[int, int] = {
        int(all_struct.chain.asym_id[i]): int(global_token_st[i])
        for i in range(all_struct.num_chains)
    }

    chain_token_idcs = {}  # {asym_id: (crop_indices, global_indices)}
    crop_token_st = cropped_struct.chain.token_start
    crop_token_end = crop_token_st + cropped_struct.chain.num_tokens
    crop_token_index = cropped_struct.token.token_index
    for i in range(cropped_struct.num_chains):
        asym_id = cropped_struct.chain.asym_id[i]
        token_st, token_end = crop_token_st[i], crop_token_end[i]
        # tokens are continuous in cropped structure
        idcs_in_crop = np.arange(token_st, token_end)

        # it is possible that tokens are not continuous in the full structure
        # e.g., 100 102 103 106 ...
        idcs_in_all = crop_token_index[token_st:token_end]

        # Move to chain-based indexing
        # e.g., 100 102 103 ... -> 0 2 3 ...
        idcs_in_all = idcs_in_all - global_token_offset[asym_id]

        chain_token_idcs[asym_id] = (idcs_in_crop, idcs_in_all)

    # Apply each swap to get alternative coordinates
    # NOTE: this tensor requires ~100 MB memory (for 4096 residues with 100 symmetries)

    # Lists to store alternative coordinates and masks
    alt_coords_list: list[np.ndarray] = [original_coords]
    alt_resolved_mask_list: list[np.ndarray] = [original_resolved_mask]

    for swap in swaps:
        alt_coords = original_coords.copy()  # [Ntoken, 24, 3]
        alt_mask = original_resolved_mask.copy()  # [Ntoken, 24]
        for src_asym_id, tgt_asym_id in swap.items():
            crop_idcs, global_idcs = chain_token_idcs[src_asym_id]

            # Get the global indices in all_struct
            # chain-based -> global token indices
            global_idcs_swap = global_idcs + global_token_offset[tgt_asym_id]

            # In-place swap
            alt_coords[crop_idcs] = all_coords[global_idcs_swap]
            alt_mask[crop_idcs] = all_resolved_mask[global_idcs_swap]

        if alt_mask.sum() <= 4:
            # Skip if too few resolved atoms
            continue

        alt_coords_list.append(alt_coords)
        alt_resolved_mask_list.append(alt_mask)

    return {
        "alt_coordinates": alt_coords_list,
        "alt_resolved_mask": alt_resolved_mask_list,
    }


@lru_cache(100)
def get_ambiguous_atom_indices(res_type: int) -> ResidueSymmetry:
    """Get the indices of ambiguous atoms for a given residue type.

    Parameters
    ----------
    res_type : int
        The residue type.

    Returns
    -------
    ResidueSymmetry(=list[AtomSwaps])
        A list of atom swaps for the residue.
    """
    res_name = C.residue.residue_id_to_name[res_type]
    if res_name not in C.atom.RESIDUE_AMBIGUOUS_ATOMS:
        # If there is no ambiguous atoms, return empty list
        return []

    src_atoms, dst_atoms = C.atom.RESIDUE_AMBIGUOUS_ATOMS[res_name]
    residue_atoms = C.atom.RESIDUE_ATOMS[res_name]
    src_indices = [residue_atoms.index(atom) for atom in src_atoms]
    dst_indices = [residue_atoms.index(atom) for atom in dst_atoms]
    # Make bijective mapping
    swap: AtomSwaps = (src_indices + dst_indices, dst_indices + src_indices)
    assert set(src_indices).isdisjoint(set(dst_indices)), (
        "Overlapping ambiguous atom indices."
    )
    return [swap]


def get_residue_symmetries(
    cropped_struct: TokenizedStructure,
) -> list[ResidueSymmetry]:
    """Get the residue symmetries for the cropped structure.

    Parameters
    ----------
    cropped_struct : TokenizedStructure
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
            res_atom_swaps.append(get_ambiguous_atom_indices(restype))
    return res_atom_swaps


# Type alias for residue unique identifier
ResUID = tuple[int, int]  # (asym_id, residue_index)


def get_molecule_symmetries(
    cropped_struct: TokenizedStructure,
    ccd: CCD,
) -> list[ResidueSymmetry]:
    # Compute ligand and non-standard amino-acids symmetries

    residue_mol: OrderedDict[ResUID, tuple[str, list[str]]] = OrderedDict()
    mol_token_st: dict[ResUID, int] = {}

    for res_i in range(cropped_struct.num_residues):
        if cropped_struct.residue.is_standard[res_i]:
            # Skip standard amino acids and nucleotides
            continue

        # Get CCD ID
        ccd_id = str(cropped_struct.residue.name[res_i])
        if ccd_id not in ccd:
            # Skip if there is no symmetry information
            continue

        # Get unique residue id (chain id, residue index)
        asym_id = int(cropped_struct.residue.asym_id[res_i])
        res_idx = int(cropped_struct.residue.residue_index[res_i])
        res_uid = (asym_id, res_idx)

        # Check input valid
        num_atoms = int(cropped_struct.residue.num_atoms[res_i])
        num_tokens = int(cropped_struct.residue.num_tokens[res_i])
        assert num_atoms == num_tokens, (
            f"Molecule token and atom number mismatch, {num_atoms} != {num_tokens}"
        )

        # Get molecule atom names
        token_st = int(cropped_struct.residue.token_start[res_i])
        mol_atom_names: list[str] = [
            C.atom.decode_atom_name(
                cropped_struct.atom.ref_atom_name_chars[i, 0].tolist()
            )
            for i in range(token_st, token_st + num_atoms)
        ]
        residue_mol[res_uid] = (ccd_id, mol_atom_names)
        mol_token_st[res_uid] = token_st

    # for each molecule, get the symmetries
    mol_atom_swaps: list[ResidueSymmetry] = []
    for mol_uid in residue_mol.keys():
        ccd_id, mol_atom_names = residue_mol[mol_uid]
        ref_mol: Component = ccd[ccd_id]

        atom_id_in_ccd: dict[int, int] = {
            ref_mol.atom_names.index(name): i for i, name in enumerate(mol_atom_names)
        }
        valid_atoms: set[int] = set(atom_id_in_ccd.keys())

        symmetries = ccd[ccd_id].symmetries
        if symmetries is None:
            symmetries = [list(range(ref_mol.num_atoms))]
        symmetries: Sequence[list[int]] = symmetries

        all_perms: list[list[int]] = []
        # Get symmetries
        for perms in symmetries:
            # Example perms for 4-atom molecules: [0, 2, 1, 3] (swapping atom 1 and 2)
            sym_dict: dict[int, int] = {}
            for i, j in enumerate(perms):
                if i not in valid_atoms:
                    # atom i is not in the molecule
                    continue
                if j in valid_atoms:
                    # both atoms are in the molecule
                    i_true = atom_id_in_ccd[i]
                    j_true = atom_id_in_ccd[j]
                    sym_dict[i_true] = j_true
                else:
                    # atom j is not in the molecule
                    # skip this symmetry
                    break
            else:
                # Completed without break
                # NOTE: This is bijective mapping within valid atoms (see above)
                all_perms.append([sym_dict[i] for i in range(len(valid_atoms))])

        swaps = []
        token_st = mol_token_st[mol_uid]
        for perm in all_perms:
            src_indices = []
            dst_indices = []
            for i, j in enumerate(perm):
                if i != j:
                    src_indices.append(i + token_st)
                    dst_indices.append(j + token_st)
            if len(src_indices) > 0:
                # NOTE: This is bijective mapping
                swaps.append((src_indices, dst_indices))
        if len(swaps) > 0:
            mol_atom_swaps.append(swaps)

    return mol_atom_swaps
