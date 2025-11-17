from collections import defaultdict

import numpy as np
import torch

import kfold.constants as C
from kfold.constants.residue import residue_index_to_name

from . import model_input, tokenized
from .utils import augmentation, frame_utils

# TODO list:
# 1. Random augmentation for each ref-pos
# 2. Replace apo_coords with real apo structure
# 3. Add symmetry.


def featurize_structure(
    structure: tokenized.TokenizedStructure,
    synchronize_ref_pos_augmentation: bool = False,
    rng: np.random.Generator | None = None,
) -> model_input.FoldingInput:
    """Featurize a tokenized structure into model input features.

    Parameters
    ----------
    structure : tokenized.TokenizedStructure
        The tokenized structure to featurize.
    synchronize_ref_pos_augmentation : bool, optional
        Whether to synchronize the random augmentation for ref_pos across all atoms,

    Returns:
        FoldingInput: The featurized model input.
    """

    def cast(data: np.ndarray) -> np.ndarray:
        if np.issubdtype(data.dtype, np.floating):
            return data.astype(np.float32, copy=False)
        elif np.issubdtype(data.dtype, np.integer):
            return data.astype(np.long, copy=False)
        elif np.issubdtype(data.dtype, np.bool_):
            return data
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")

    # =========================================== #
    # ====== Extract raw features and cast ====== #
    # =========================================== #

    chain_data = structure.chain
    token_data = structure.token
    atom_data = structure.atom
    bond_data = structure.bond

    # === Chain-level features ===
    num_chains = chain_data.length
    chain_dict: dict[str, np.ndarray] = {
        k: cast(v) for k, v in chain_data.to_dict().items()
    }
    chain_dict["pad_mask"] = np.ones((num_chains,), dtype=np.bool_)

    # === Token-level features ===
    num_tokens = token_data.length
    token_dict: dict[str, np.ndarray] = {
        k: cast(v) for k, v in token_data.to_dict().items()
    }
    token_dict["pad_mask"] = np.ones((num_tokens,), dtype=np.bool_)

    # === Atom-level features ===
    # Make sparse atom features [Ntoken, 24, ...] into dense [Nallatom, ...]
    atom_to_token = np.repeat(
        np.arange(num_tokens), token_dict["num_atoms"]
    )  # [Nallatom,]
    num_total_atoms = atom_to_token.shape[0]
    # Atom index within each token (0 to num_atoms_in_token-1)
    atom_in_token_idx = np.arange(num_total_atoms) - np.repeat(
        np.cumsum(np.concatenate([[0], token_dict["num_atoms"][:-1]])),
        token_dict["num_atoms"],
    )  # [Nallatom,]
    atom_dict = {
        k: cast(v)[atom_to_token, atom_in_token_idx]  # Fancy indexing - no loop!
        for k, v in atom_data.to_dict().items()
    }
    atom_dict["label_coords"] = atom_dict.pop("coords")  # Rename for clarity
    atom_dict["token_index"] = atom_to_token
    atom_dict["pad_mask"] = np.ones((num_total_atoms,), dtype=np.bool_)

    # Centering the ground truth coords
    atom_dict["label_coords"] = augmentation.do_centering(
        atom_dict["label_coords"], atom_dict["resolved_mask"]
    )
    # TODO: if we use multiple apo structures, randomly sample one apo structure here.
    # TODO: If we use CCD, use random ETKDG conformers here.
    atom_dict["apo_coords"] = augmentation.do_centering(
        atom_dict["apo_coords"], atom_dict["apo_mask"]
    )

    # === Bond-level features ===
    num_bonds = bond_data.length
    bond_dict: dict[str, np.ndarray] = {
        k: cast(v) for k, v in bond_data.to_dict().items()
    }
    bond_dict["pad_mask"] = np.ones((num_bonds,), dtype=np.bool_)

    # ============================================
    # ======= Compute additional features ========
    # ============================================

    # === Token-level features ===
    # Make one-hot vector for residue types
    residue_one_hot = np.eye(32, dtype=np.float32)
    token_dict["res_type"] = residue_one_hot[token_dict["res_type"]]

    # NOTE: Token index should be remapped due cropping
    token_dict["org_token_index"] = token_dict["token_index"]
    token_dict["token_index"] = np.arange(num_tokens, dtype=np.long)
    token_dict["pocket_contact_type"] = np.full(
        (num_tokens,), C.pocket.ContactType.UNSPECIFIED, dtype=np.long
    )

    # Add frame information
    token_dict["frames_index"] = np.zeros((num_tokens, 3), dtype=np.long)
    token_dict["frames_mask"] = np.zeros((num_tokens,), dtype=np.bool_)
    for tidx in range(num_tokens):
        is_standard = token_dict["is_standard"][tidx]
        if is_standard:
            res_name = residue_index_to_name[token_data.res_type[tidx]]
            num_atoms_in_res = token_dict["num_atoms"][tidx]
            if res_name is not C.residue.ResidueName.UNK and (num_atoms_in_res >= 3):
                restype_atoms = C.atom.RESIDUE_ATOMS[res_name]
                n, ca, c = C.atom.RESIDUE_FRAME_ATOMS[res_name]
                frame_atom_index = [restype_atoms.index(a) for a in (n, ca, c)]
                is_frame = atom_data.resolved_mask[tidx, frame_atom_index].all()
                token_dict["frames_index"][tidx] = frame_atom_index
                token_dict["frames_mask"][tidx] = is_frame

    # Map atom indices to global atom indices
    atom_offset = (
        np.cumsum(token_dict["num_atoms"]) - token_dict["num_atoms"]
    )  # [Ntoken,]
    token_dict["center_index"] = token_dict["center_index"] + atom_offset
    token_dict["disto_index"] = token_dict["disto_index"] + atom_offset
    token_dict["frames_index"] = token_dict["frames_index"] + atom_offset[:, np.newaxis]

    # add disto/center coords
    # HACK: we assume there is only one holo coordinate set.
    token_dict["disto_coords"] = atom_dict["label_coords"][:, 0][
        token_dict["disto_index"]
    ]
    token_dict["center_coords"] = atom_dict["label_coords"][:, 0][
        token_dict["center_index"]
    ]

    # Masks indicating whether the center/disto atoms are resolved
    token_dict["resolved_mask"] = (
        token_data.resolved_mask & atom_dict["resolved_mask"][token_dict["center_index"]]
    )
    token_dict["disto_mask"] = (
        token_data.resolved_mask & atom_dict["resolved_mask"][token_dict["disto_index"]]
    )

    # =========================== #
    # === Atom-level features === #
    # =========================== #

    # Make one-hot vector for atom types
    ref_element_one_hot = np.eye(128, dtype=np.float32)
    atom_dict["ref_element"] = ref_element_one_hot[atom_dict["ref_element"]]
    ref_atom_name_one_hot = np.eye(64, dtype=np.float32)
    atom_dict["ref_atom_name_chars"] = ref_atom_name_one_hot[
        atom_dict["ref_atom_name_chars"]
    ]

    # Add ref space uid info
    # See section 2.8 Table 5 of AlphaFold3 paper
    ref_space_uid: list[int] = []
    ref_space_dict: dict[tuple[int, int], int] = {}
    ref_space_natoms: defaultdict[int, int] = defaultdict(int)
    for tidx, ref_key in enumerate(
        zip(token_dict["asym_id"], token_dict["residue_index"], strict=True)
    ):
        if ref_key not in ref_space_dict:
            ref_space_dict[ref_key] = len(ref_space_dict)
        v = ref_space_dict[ref_key]
        ref_space_uid.extend([v] * token_dict["num_atoms"][tidx])
        ref_space_natoms[v] += token_dict["num_atoms"][tidx]
    atom_dict["ref_space_uid"] = np.array(ref_space_uid, dtype=np.long)

    # TODO: rotate conformers for each residue.
    # TODO: random sample from multiple ETKDG conformers.
    if synchronize_ref_pos_augmentation:
        atom_dict["ref_pos"] = augmentation.center_random_augmentation(
            atom_dict["ref_pos"], atom_dict["pad_mask"], rng=rng
        )
    else:
        new_ref_pos_list = np.zeros_like(atom_dict["ref_pos"])
        start_idx = 0
        # UID is incremental
        for uid in sorted(ref_space_natoms.keys()):
            end_idx = start_idx + ref_space_natoms[uid]
            ref_pos_residue = atom_dict["ref_pos"][start_idx:end_idx]
            # Apply random augmentation per residue
            ref_pos_residue = augmentation.center_random_augmentation(
                ref_pos_residue,
                atom_dict["pad_mask"][start_idx:end_idx],
                rng=rng,
            )
            # Store back
            new_ref_pos_list[start_idx:end_idx] = ref_pos_residue
            start_idx = end_idx
        atom_dict["ref_pos"] = new_ref_pos_list

    # === Bond-level features ===
    # TODO: Remap token indices to cropped tokens
    # e.g., [0, 3, 4, 5, 8] -> [0, 1, 2, 3, 4]
    original_token_index = token_dict["org_token_index"]
    token_index = token_dict["token_index"]
    token_map = np.zeros(original_token_index.max() + 1, dtype=np.long) - 1
    token_map[original_token_index] = token_index
    bond_dict["token_index"] = token_map[bond_dict["token_index"]]

    # Indicate whether the bond atoms belong to polymer or ligand
    token1, token2 = bond_dict["token_index"][:, 0], bond_dict["token_index"][:, 1]

    is_ligand1 = token_dict["chain_type"][token1] == C.chain.ChainType.LIGAND
    is_ligand2 = token_dict["chain_type"][token2] == C.chain.ChainType.LIGAND
    bond_dict["is_polymer_ligand"] = ((~is_ligand1) & is_ligand2) | (
        is_ligand1 & (~is_ligand2)
    )
    bond_dict["is_ligand_ligand"] = is_ligand1 & is_ligand2

    # Remove unused feature before converting to tensors
    token_dict.pop("is_standard")
    token_dict.pop("num_atoms")

    # === Convert to tensors ===
    chain_layout = model_input.ChainLayout(
        **{k: torch.from_numpy(v) for k, v in chain_dict.items()}
    )

    token_layout = model_input.TokenLayout(
        **{k: torch.from_numpy(v) for k, v in token_dict.items()}
    )

    atom_layout = model_input.AtomLayout(
        **{k: torch.from_numpy(v) for k, v in atom_dict.items()}
    )

    bond_layout = model_input.BondLayout(
        **{k: torch.from_numpy(v) for k, v in bond_dict.items()}
    )

    # === Before returning, compute ligand frames inplace === #
    frame_utils.compute_ligand_frames_inplace(token_layout, atom_layout, chain_layout)

    folding_input = model_input.FoldingInput(
        chain=chain_layout,
        token=token_layout,
        atom=atom_layout,
        bond=bond_layout,
    )
    return folding_input
