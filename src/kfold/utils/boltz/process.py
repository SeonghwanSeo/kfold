from typing import TypeVar

import numpy as np
import torch

import kfold.constants as C
from kfold.data import model_input

from .structure import BoltzStructure

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


def compute_ligand_frames_inplace(
    token_layout: model_input.TokenLayout,
    atom_layout: model_input.AtomLayout,
    chain_layout: model_input.ChainLayout,
):
    """Update frames for non-polymer chains."""

    def compute_collinear_mask(v1: torch.Tensor, v2: torch.Tensor):
        norm1 = torch.norm(v1, dim=1, keepdim=True)
        norm2 = torch.norm(v2, dim=1, keepdim=True)
        v1 = v1 / (norm1 + 1e-6)
        v2 = v2 / (norm2 + 1e-6)
        mask_angle = torch.abs(torch.sum(v1 * v2, dim=1)) < 0.9063
        mask_overlap1 = norm1.reshape(-1) > 1e-2
        mask_overlap2 = norm2.reshape(-1) > 1e-2
        return mask_angle & mask_overlap1 & mask_overlap2

    num_chains = len(chain_layout)
    is_ligand = chain_layout.is_ligand
    num_atoms = chain_layout.num_atoms
    num_tokens = chain_layout.num_tokens
    token_starts = chain_layout.num_tokens.cumsum(0) - num_tokens
    atom_starts = chain_layout.num_atoms.cumsum(0) - num_atoms

    frames_index = token_layout.frames_index  # (Nt, 3)
    frames_mask = token_layout.frames_mask  # (Nt,)

    for i in range(num_chains):
        if not is_ligand[i] or num_atoms[i] < 3:
            continue

        atom_st, atom_end = atom_starts[i], atom_starts[i] + num_atoms[i]
        token_st, token_end = token_starts[i], token_starts[i] + num_tokens[i]
        assert num_atoms[i] == num_tokens[i], (
            "Ligand chain should have equal number of tokens and atoms."
        )

        coords = atom_layout.label_coords[atom_st:atom_end].reshape(-1, 3)
        dist_mat = torch.cdist(coords, coords)

        resolved_mask = atom_layout.resolved_mask[atom_st:atom_end]
        resolved_pair = resolved_mask[None, :] & resolved_mask[:, None]
        dist_mat = torch.where(resolved_pair, dist_mat, torch.tensor(1e6))
        indices = dist_mat.argsort(dim=1)
        frames = (
            torch.stack([indices[:, 1], indices[:, 0], indices[:, 2]], dim=1) + atom_st
        )
        frames_index[token_st:token_end] = frames
        frames_mask[token_st:token_end] = atom_layout.resolved_mask[frames].all(dim=1)

    coords = atom_layout.label_coords.reshape(-1, 3)
    frames_expanded = coords[frames_index]  # (Nt, 3, 3)
    mask_collinear = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    )
    frames_mask[~mask_collinear] = False


def centering(coords: ArrayT, mask: ArrayT, mask_to_zero: bool = True) -> ArrayT:
    """Center coordinates based on the masked mean position.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        The coordinates tensor of shape (N, ..., 3).
    mask : np.ndarray | torch.Tensor
        The boolean mask tensor of shape (N,).
    mask_to_zero : bool, optional
        If True, positions where mask is False will be set to zero after centering.
        # NOTE: this is not used in Boltz. (Boltz's masked coords is -center_pos)

    Returns
    -------
    np.ndarray | torch.Tensor
        The centered coordinates tensor of shape (N, ..., 3).

    """
    if not mask.any():
        return coords
    masked_coords = coords[mask]
    if isinstance(masked_coords, np.ndarray):
        center_pos = masked_coords.mean(axis=0, keepdims=True)
    else:
        center_pos = masked_coords.mean(dim=0, keepdim=True)
    centered_coords = coords - center_pos
    if mask_to_zero:
        centered_coords[~mask] = 0.0
    return centered_coords  # type: ignore


def parse_structure(
    chains: np.ndarray,
    structure: BoltzStructure,
) -> model_input.FoldingInput:
    """Extract structure layout from BoltzStructure.

    Parameters
    ----------
    chains : np.ndarray (boltz.types.Chain)
        The chain array.
    structure : BoltzStructure
        The BoltzStructure object.

    Returns
    -------
    FoldingInput
        The parsed structure layout.

    """

    # Define field types
    boolean_fields = [
        "frames_mask",
        "resolved_mask",
        "pad_mask",
        "is_pocket",
    ]
    float_fields = [
        "ref_charge",
        "ref_pos",
        "label_coords",
        "apo_coords",
    ]

    def get_dtype(key: str) -> torch.dtype:
        if key in boolean_fields:
            return torch.bool
        elif key in float_fields:
            return torch.float32
        else:
            return torch.int32

    chain_info = {
        "chain_type": torch.as_tensor(chains["mol_type"].copy(), dtype=torch.int32),
        "entity_id": torch.as_tensor(chains["entity_id"].copy(), dtype=torch.int32) + 1,
        "asym_id": torch.as_tensor(chains["asym_id"].copy(), dtype=torch.int32) + 1,
        "sym_id": torch.as_tensor(chains["sym_id"].copy(), dtype=torch.int32) + 1,
        "num_residues": torch.as_tensor(chains["res_num"].copy(), dtype=torch.int32),
        "num_atoms": torch.as_tensor(chains["atom_num"].copy(), dtype=torch.int32),
        # 'num_tokens' will be computed below
        "num_tokens": torch.zeros(len(chains), dtype=torch.int32),
    }

    token_info = {
        "res_type": [],
        "chain_type": [],
        "entity_id": [],
        "asym_id": [],
        "sym_id": [],
        "residue_index": [],
        "disto_index": [],
        "center_index": [],
        "cyclic_period": [],
        "frames_index": [],
        "resolved_mask": [],
        "frames_mask": [],
    }

    atom_info = {
        "ref_space_uid": [],
        "token_index": [],
    }

    atom_info_concat = {
        "ref_atom_name_chars": [],
        "ref_element": [],
        "ref_charge": [],
        "ref_pos": [],
        "resolved_mask": [],
        "label_coords": [],
        "apo_coords": [],
    }

    bond_info = {
        "asym_id": [],
        "token_index": [],
        "atom_index": [],
        "bond_type": [],
    }

    # Since some chains can be masked,
    # we reindex the chain, residue, and atom indices
    chain_index_map = {}  # starts from 0
    res_index_map = {}  # starts from 0 for each chain
    atom_index_map = {}  # shifted for masked chains

    # === Get token features and some atom features === #
    # global index (offset)
    global_atom_idx = 0
    global_token_idx = 0
    global_res_idx = 0
    for chain_index, chain in enumerate(chains):
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]
        chain_index_map[chain["asym_id"]] = chain_index

        # Chain indices
        chain_type = chain_info["chain_type"][chain_index]
        asym_id = chain_info["asym_id"][chain_index]
        entity_id = chain_info["entity_id"][chain_index]
        sym_id = chain_info["sym_id"][chain_index]

        # === Iterate residues === #
        # Iterate residues in the chain and fill token and some atom info
        # NOTE: res_index is reindexed per chain
        num_tokens_in_chain = 0
        for i, residue in enumerate(structure.residues[res_start:res_end]):
            # AF3 indexing rule: starts from 1
            chain_res_index = i + 1
            # Map original residue index to reindexed residue index
            res_index_map[residue["res_idx"]] = chain_res_index

            atom_start = residue["atom_idx"]
            num_atoms_in_res = residue["atom_num"]
            atom_end = atom_start + num_atoms_in_res
            residue_atoms = structure.atoms[atom_start:atom_end]

            if residue["is_standard"]:
                # Proteins' amino acid and nucleic acids' base
                res_name = C.residue.ResidueName(str(residue["name"]).strip())
                token_info["res_type"].append(res_name.index)

                # === Insert token info === #
                token_info["chain_type"].append(chain_type)
                token_info["asym_id"].append(asym_id)
                token_info["entity_id"].append(entity_id)
                token_info["sym_id"].append(sym_id)

                token_info["residue_index"].append(chain_res_index)
                # shift atom indices to be relative to the entire structure
                token_info["disto_index"].append(
                    global_atom_idx + (residue["atom_disto"] - atom_start)
                )
                token_info["center_index"].append(
                    global_atom_idx + (residue["atom_center"] - atom_start)
                )
                token_info["resolved_mask"].append(residue["is_present"])
                # NOTE: Boltz1 training set does not include cyclic_period info
                token_info["cyclic_period"].append(0)

                # === Insert frame info === #
                if res_name is C.residue.ResidueName.UNK or (num_atoms_in_res < 3):
                    # Unknown residue or insufficient atoms for frame
                    frames_index = [0, 0, 0]
                    is_frames = False
                else:
                    restype_atoms = C.atom.RESIDUE_ATOMS[res_name]
                    n, ca, c = C.atom.RESIDUE_FRAME_ATOMS[res_name]
                    frames_index = [
                        restype_atoms.index(n),
                        restype_atoms.index(ca),
                        restype_atoms.index(c),
                    ]
                    is_frames = residue_atoms[frames_index]["is_present"].all()
                token_info["frames_index"].append(
                    [v + global_atom_idx for v in frames_index]
                )
                token_info["frames_mask"].append(is_frames)

                # === Insert atom info === #
                atom_info["token_index"].extend([global_token_idx] * num_atoms_in_res)
                atom_info["ref_space_uid"].extend([global_res_idx] * num_atoms_in_res)

                # === Update offset === #
                num_tokens_in_chain += 1
                global_token_idx += 1
                global_atom_idx += num_atoms_in_res
            else:
                # Ligands, Modifications, Covalent inhibitors
                res_name = C.residue.ResidueName("UNK")  # use unknown residue name
                res_type = res_name.index

                for atom in residue_atoms:
                    # === Insert token info === #
                    token_info["chain_type"].append(chain_type)
                    token_info["asym_id"].append(asym_id)
                    token_info["entity_id"].append(entity_id)
                    token_info["sym_id"].append(sym_id)

                    token_info["res_type"].append(res_type)
                    token_info["residue_index"].append(chain_res_index)
                    token_info["disto_index"].append(global_atom_idx)
                    token_info["center_index"].append(global_atom_idx)
                    token_info["resolved_mask"].append(atom["is_present"])
                    # NOTE: Boltz1 training set does not include cyclic_period info
                    token_info["cyclic_period"].append(0)

                    # === Insert frame info === #
                    token_info["frames_index"].append(
                        (global_atom_idx, global_atom_idx, global_atom_idx)
                    )
                    token_info["frames_mask"].append(False)

                    # === Insert atom info === #
                    atom_info["token_index"].append(global_token_idx)
                    atom_info["ref_space_uid"].append(global_res_idx)

                    # === Update index === #
                    num_tokens_in_chain += 1
                    global_token_idx += 1
                    global_atom_idx += 1

            # Update global residue index
            global_res_idx += 1

        # Update number of tokens in the chain
        chain_info["num_tokens"][chain_index] = num_tokens_in_chain

    # === Get atom features === #
    global_atom_idx = 0
    for chain in chains:
        atom_start = chain["atom_idx"]
        atom_end = atom_start + chain["atom_num"]
        atoms = structure.atoms[atom_start:atom_end]
        atom_info_concat["ref_atom_name_chars"].append(atoms["name"])
        atom_info_concat["ref_element"].append(atoms["element"])
        atom_info_concat["ref_charge"].append(atoms["charge"])
        atom_info_concat["ref_pos"].append(atoms["conformer"])
        atom_info_concat["label_coords"].append(atoms["coords"][:, None, :])

        # FIXME: we should use actual apo coords when available
        chain_coords = centering(atoms["coords"], atoms["is_present"])[:, None, :]

        mask = atoms["is_present"]
        chain_centers = chain_coords[mask].mean(axis=0, keepdims=True)
        chain_apo_coords = chain_coords - chain_centers
        atom_info_concat["apo_coords"].append(chain_apo_coords)
        atom_info_concat["resolved_mask"].append(atoms["is_present"])

        # Map original atom index to reindexed atom index
        for i, atom_index in enumerate(range(atom_start, atom_end)):
            atom_index_map[atom_index] = global_atom_idx + i

        # Update atom offset
        global_atom_idx += chain["atom_num"]

    # === Get bond features === #
    # First iterate bonds (intra-chain)
    for bond in structure.bonds:
        # Map original atom indices to reindexed atom indices
        if bond["atom_1"] not in atom_index_map or bond["atom_2"] not in atom_index_map:
            continue
        atom_1 = atom_index_map[bond["atom_1"]]
        atom_2 = atom_index_map[bond["atom_2"]]
        bond_type = bond["type"]

        # Get asym_id and token_index from atom_index
        token_index_1 = atom_info["token_index"][atom_1]
        token_index_2 = atom_info["token_index"][atom_2]
        asym_id1 = token_info["asym_id"][token_index_1]
        asym_id2 = token_info["asym_id"][token_index_2]

        assert asym_id1 == asym_id2, "Bonds across chains are not supported."
        bond_info["asym_id"].append((asym_id1, asym_id2))
        bond_info["token_index"].append((token_index_1, token_index_2))
        bond_info["atom_index"].append((atom_1, atom_2))
        bond_info["bond_type"].append(bond_type)

    # Then iterate cross-chain bonds
    for bond in structure.connections:
        if bond["atom_1"] not in atom_index_map or bond["atom_2"] not in atom_index_map:
            continue

        # Map original atom indices to reindexed atom indices
        atom_1 = atom_index_map[bond["atom_1"]]
        atom_2 = atom_index_map[bond["atom_2"]]
        bond_type = C.bond.ConnectType.COVALENT.value

        # Get asym_id and token_index from atom_index
        token_index_1 = atom_info["token_index"][atom_1]
        token_index_2 = atom_info["token_index"][atom_2]
        asym_id1 = token_info["asym_id"][token_index_1]
        asym_id2 = token_info["asym_id"][token_index_2]

        bond_info["asym_id"].append((asym_id1, asym_id2))
        bond_info["token_index"].append((token_index_1, token_index_2))
        bond_info["atom_index"].append((atom_1, atom_2))
        bond_info["bond_type"].append(bond_type)

    # === Convert lists to tensors === #
    token_info = {
        key: torch.as_tensor(value, dtype=get_dtype(key))
        for key, value in token_info.items()
    }

    atom_info = {
        key: torch.as_tensor(value, dtype=get_dtype(key))
        for key, value in atom_info.items()
    } | {
        key: torch.as_tensor(np.concatenate(value), dtype=get_dtype(key))
        for key, value in atom_info_concat.items()
    }
    # Centering the ground truth coords
    atom_info["label_coords"] = centering(
        atom_info["label_coords"], atom_info["resolved_mask"]
    )

    bond_info = {
        key: torch.as_tensor(value, dtype=get_dtype(key))
        for key, value in bond_info.items()
    }

    # Add additional fields
    token_info["token_index"] = torch.arange(1, len(token_info["res_type"]) + 1)
    token_info["is_pocket"] = torch.zeros_like(token_info["resolved_mask"])

    # Add mask fields
    chain_info["pad_mask"] = torch.ones_like(chain_info["chain_type"], dtype=torch.bool)
    token_info["pad_mask"] = torch.ones_like(token_info["resolved_mask"])
    atom_info["pad_mask"] = torch.ones_like(atom_info["resolved_mask"])
    bond_info["pad_mask"] = torch.ones_like(bond_info["bond_type"], dtype=torch.bool)

    Nc = chain_info["chain_type"].shape[0]
    Nt = token_info["res_type"].shape[0]
    Na = atom_info["ref_atom_name_chars"].shape[0]
    Nb = bond_info["bond_type"].shape[0]
    chain_layout = model_input.ChainLayout(
        chain_type=chain_info["chain_type"].view(Nc),
        entity_id=chain_info["entity_id"].view(Nc),
        asym_id=chain_info["asym_id"].view(Nc),
        sym_id=chain_info["sym_id"].view(Nc),
        num_residues=chain_info["num_residues"].view(Nc),
        num_atoms=chain_info["num_atoms"].view(Nc),
        num_tokens=chain_info["num_tokens"].view(Nc),
        pad_mask=chain_info["pad_mask"].view(Nc),
    )
    token_layout = model_input.TokenLayout(
        token_index=token_info["token_index"].view(Nt),
        res_type=token_info["res_type"].view(Nt),
        chain_type=token_info["chain_type"].view(Nt),
        entity_id=token_info["entity_id"].view(Nt),
        asym_id=token_info["asym_id"].view(Nt),
        sym_id=token_info["sym_id"].view(Nt),
        residue_index=token_info["residue_index"].view(Nt),
        disto_index=token_info["disto_index"].view(Nt),
        center_index=token_info["center_index"].view(Nt),
        frames_index=token_info["frames_index"].view(Nt, 3),
        resolved_mask=token_info["resolved_mask"].view(Nt),
        frames_mask=token_info["frames_mask"].view(Nt),
        pad_mask=token_info["pad_mask"].view(Nt),
        is_pocket=token_info["is_pocket"].view(Nt),
        cyclic_period=token_info["cyclic_period"].view(Nt),
    )
    atom_layout = model_input.AtomLayout(
        ref_atom_name_chars=atom_info["ref_atom_name_chars"].view(Na, 4),
        ref_element=atom_info["ref_element"].view(Na),
        ref_charge=atom_info["ref_charge"].view(Na),
        ref_pos=atom_info["ref_pos"].view(Na, 3),
        ref_space_uid=atom_info["ref_space_uid"].view(Na),
        token_index=atom_info["token_index"].view(Na),
        resolved_mask=atom_info["resolved_mask"].view(Na),
        label_coords=atom_info["label_coords"].view(Na, -1, 3),
        apo_coords=atom_info["apo_coords"].view(Na, -1, 3),
        pad_mask=atom_info["pad_mask"].view(Na),
    )
    bond_layout = model_input.BondLayout(
        asym_id=bond_info["asym_id"].view(Nb, 2),
        token_index=bond_info["token_index"].view(Nb, 2),
        atom_index=bond_info["atom_index"].view(Nb, 2),
        bond_type=bond_info["bond_type"].view(Nb),
        pad_mask=bond_info["pad_mask"].view(Nb),
    )
    # Update ligand frames
    compute_ligand_frames_inplace(token_layout, atom_layout, chain_layout)

    # === Construct FoldingInput === #
    folding_input = model_input.FoldingInput(
        chain=chain_layout,
        token=token_layout,
        atom=atom_layout,
        bond=bond_layout,
    )
    return folding_input
