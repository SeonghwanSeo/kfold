from collections import defaultdict

import numpy as np
import torch

import kfold.constants as C
from kfold.constants.residue import residue_index_to_name
from kfold.utils.geometry.random_augment import center_random_augmentation, do_centering

from . import model_input, structure
from .utils import frame_utils

# TODO list:
# - Add symmetry.


# === Helper functions === #
def do_augment_ref_pos(
    ref_pos: np.ndarray,
    mask: np.ndarray,
    conformer_sizes: list[int] | None = None,
    rng: np.random.Generator | None = None,
    synchronize: bool = False,
) -> np.ndarray:
    """Augment reference positions with random translation and rotation.

    Parameters
    ----------
    ref_pos : np.ndarray
        Reference positions of shape [Natom, 3].
    mask : np.ndarray
        Mask indicating valid atoms of shape [Natom,].
    conformer_sizes : list[int] | None
        List of number of atoms per each conformer.
    rng : np.random.Generator | None
        Random number generator for augmentation.
    synchronize : bool
        Whether to synchronize the random augmentation across all atoms.

    Returns
    -------
    augmented_ref_pos : np.ndarray
        Augmented reference positions of shape [Natom, 3].

    """

    if synchronize:
        # NOTE: Just for comparison with Boltz. This flag should be False.
        return center_random_augmentation(ref_pos, mask, rng=rng)
    else:
        assert conformer_sizes is not None

        new_ref_pos = np.zeros_like(ref_pos)
        start_idx = 0
        # Apply random augmentation per residue(conformer)
        for natom in conformer_sizes:
            end_idx = start_idx + natom
            new_ref_pos[start_idx:end_idx] = center_random_augmentation(
                ref_pos[start_idx:end_idx],  # =residue_conf
                mask[start_idx:end_idx],  # =residue_mask
                rng=rng,
            )
            start_idx = end_idx
        return new_ref_pos


def do_augment_apo_structure(
    apo_coords: np.ndarray,
    mask: np.ndarray,
    chain_sizes: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Augment apo structure coordinates with random rotation.
    Parameters
    ----------
    apo_coords : np.ndarray
        Apo structure coordinates of shape [Napo, Natom, 3].
    mask : np.ndarray
        Mask indicating valid atoms of shape [Napo, Natom].
    chain_sizes : np.ndarray
        Array of number of atoms per each chain.
    rng : np.random.Generator | None
        Random number generator for augmentation.

    Returns
    -------
    augmented_apo_coords : np.ndarray
        Augmented apo structure coordinates of shape [Napo, Natom, 3].
    """
    # Apply random rotation per apo coords
    # NOTE: Unlike holo structure, apo structure is input so that
    # random translation is not applied.

    new_coords = np.zeros_like(apo_coords)
    start_idx = 0
    for natom in chain_sizes:
        end_idx = start_idx + natom
        new_coords[:, start_idx:end_idx] = center_random_augmentation(
            apo_coords[:, start_idx:end_idx],  # =apo_chain_coords
            mask[:, start_idx:end_idx],  # =apo_chain_mask
            rng=rng,
        )
        start_idx = end_idx
    return new_coords


def featurize_structure(
    struct: structure.TokenizedStructure,
    augment_ref_pos: bool = True,
    augment_apo: bool = True,
    synchronize_ref_pos_augmentation: bool = False,
    rng: np.random.Generator | None = None,
) -> model_input.FoldingInput:
    """Featurize a tokenized structure into model input features.

    Parameters
    ----------
    struct : structure.TokenizedStructure
        The tokenized structure to featurize.
    augment_ref_pos : bool, optional
        Whether to apply random augmentation to ref_pos,
    augment_apo : bool, optional
        Whether to apply random augmentation to apo_coords,
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

    chain_data = struct.chain
    token_data = struct.token
    atom_data = struct.atom
    bond_data = struct.bond

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
    atom_dict["pad_mask"] = np.ones((num_total_atoms,), dtype=np.bool_)  # Remove padding

    # Centering the ground truth coords
    # [Natom, Nholo, 3] -> [Nholo, Natom, 3] -> [Natom, Nholo, 3]
    atom_dict["label_coords"] = do_centering(
        atom_dict["label_coords"].transpose(1, 0, 2),
        atom_dict["resolved_mask"].reshape(1, -1),
        mask_to_zero=True,
    ).transpose(1, 0, 2)

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
        (num_tokens,), C.constraint.ConstraintType.UNSPECIFIED, dtype=np.long
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

    # Random augmentation (stochasticity)
    # TODO: random sample from multiple ETKDG conformers.
    if augment_ref_pos:
        # Compute the number of atoms per each conformer
        num_atoms_per_conformer = [
            ref_space_natoms[uid] for uid in sorted(ref_space_natoms.keys())
        ]
        atom_dict["ref_pos"] = do_augment_ref_pos(
            ref_pos=atom_dict["ref_pos"],
            mask=atom_dict["pad_mask"],  # Same to np.ones(...)
            conformer_sizes=num_atoms_per_conformer,
            synchronize=synchronize_ref_pos_augmentation,
            rng=rng,
        )

    # TODO: if we use multiple apo structures, randomly sample one apo structure here.
    # TODO: If we use CCD, use random ETKDG conformers here.
    if augment_apo:
        num_atoms_per_chains = chain_dict["num_atoms"]
        # [Natom, Napo, 3] -> [Napo, Natom, 3] -> [Natom, Napo, 3]
        atom_dict["apo_coords"] = do_augment_apo_structure(
            apo_coords=atom_dict["apo_coords"].transpose(1, 0, 2),
            mask=atom_dict["apo_mask"].transpose(1, 0),
            chain_sizes=num_atoms_per_chains,
            rng=rng,
        ).transpose(1, 0, 2)
    else:
        # [Natom, Napo, 3] -> [Napo, Natom, 3] -> [Natom, Napo, 3]
        atom_dict["apo_coords"] = do_centering(
            atom_dict["apo_coords"].transpose(1, 0, 2),
            atom_dict["apo_mask"].transpose(1, 0),
        ).transpose(1, 0, 2)

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
