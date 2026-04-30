from collections import defaultdict

import numpy as np
import torch

import kfold.constants as C
from kfold.data.types.model_input import (
    AtomTensor,
    BondTensor,
    ChainTensor,
    ConstraintTensor,
    FoldingInput,
    SequenceTensor,
    TokenTensor,
)
from kfold.data.types.tokenized import TokenizedStructure


class InputFeaturizer:
    """A class for featurizing tokenized structures into model input features."""

    def __call__(self, struct: TokenizedStructure) -> FoldingInput:
        return self.run(struct)

    def run(self, struct: TokenizedStructure) -> FoldingInput:
        """Featurize a tokenized structure into model input features.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure to featurize.

        Returns
        -------
        f_input_upd: FoldingInput
            The featurized model input
        """
        return to_folding_input(struct)


def to_folding_input(struct: TokenizedStructure) -> FoldingInput:
    """Featurize a tokenized structure into model input features.

    Parameters
    ----------
    struct : TokenizedStructure
        The tokenized structure to featurize.

    Returns
    -------
    f_input: FoldingInput
        The featurized model input
    """

    # =========================================== #
    # ========== Extract raw features =========== #
    # =========================================== #
    chain_data = struct.chain
    token_data = struct.token
    atom_data = struct.atom
    bond_data = struct.bond
    constraint_data = struct.constraint

    # === Chain-level features ===
    num_chains = chain_data.length
    chain_dict: dict[str, np.ndarray] = chain_data.to_dict()
    chain_dict["pad_mask"] = np.ones((num_chains,), dtype=np.bool_)

    # === Token-level features ===
    num_tokens = token_data.length
    token_dict: dict[str, np.ndarray] = token_data.to_dict()
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
        k: v[atom_to_token, atom_in_token_idx] for k, v in atom_data.to_dict().items()
    }
    atom_dict["label_coords"] = np.nan_to_num(atom_dict["label_coords"], nan=0.0)
    atom_dict["apo_coords"] = np.nan_to_num(atom_dict["apo_coords"], nan=0.0)
    atom_dict["ref_pos"] = np.nan_to_num(atom_dict["ref_pos"], nan=0.0)
    atom_dict["token_index"] = atom_to_token
    atom_dict["pad_mask"] = np.ones((num_total_atoms,), dtype=np.bool_)  # Remove padding

    # === Bond-level features ===
    num_bonds = bond_data.length
    bond_dict: dict[str, np.ndarray] = bond_data.to_dict()
    bond_dict["pad_mask"] = np.ones((num_bonds,), dtype=np.bool_)

    # === Constant-level features ===
    num_constraints = constraint_data.length
    constraint_dict: dict[str, np.ndarray] = constraint_data.to_dict()
    constraint_dict["pad_mask"] = np.ones((num_constraints,), dtype=np.bool_)

    # ============================================
    # ======= Compute additional features ========
    # ============================================

    # === Token-level features ===
    # Make one-hot vector for residue types
    residue_one_hot = np.eye(32, dtype=np.float32)
    token_dict["res_type"] = residue_one_hot[token_dict["res_type"]]

    # NOTE: Token index should be remapped due cropping
    org_token_index = token_dict["token_index"]
    max_token_index = org_token_index.max()
    token_index = np.arange(num_tokens, dtype=np.int64)
    token_dict["org_token_index"] = org_token_index
    token_dict["token_index"] = token_index
    token_map = np.full((max_token_index + 2), -1, dtype=np.int64)
    token_map[org_token_index] = token_index

    # Map atom indices to global atom indices
    atom_offset = np.cumsum(token_dict["num_atoms"]) - token_dict["num_atoms"]
    token_dict["center_index"] = token_dict["center_index"] + atom_offset
    token_dict["repr_index"] = token_dict["repr_index"] + atom_offset

    # Add center/representative atom coordinates
    token_dict["repr_coords"] = atom_dict["label_coords"][token_dict["repr_index"]]
    token_dict["center_coords"] = atom_dict["label_coords"][token_dict["center_index"]]

    # Masks indicating whether the center/repr atoms are resolved
    token_dict["center_mask"] = atom_dict["resolved_mask"][token_dict["center_index"]]
    token_dict["repr_mask"] = atom_dict["resolved_mask"][token_dict["repr_index"]]

    # Map frame indices to global atom indices
    # NOTE: If token_index=-1, atom_index is also set to -1.
    raw_frame_token_index = token_dict.pop("frame_token_index")
    raw_frame_token_index[raw_frame_token_index > max_token_index] = -1
    frame_token_index = token_map[raw_frame_token_index]
    frame_token_index[raw_frame_token_index == -1] = -1
    frame_atom_index = token_dict.pop("frame_atom_index")
    frame_mask = (frame_token_index != -1).all(-1) & (frame_atom_index != -1).all(-1)
    # Map to global atom indices, set to 0 for invalid indices
    frame_index = atom_offset[frame_token_index] + frame_atom_index
    frame_index[~frame_mask] = 0
    token_dict["frame_index"] = frame_index
    token_dict["frame_mask"] = frame_mask

    # === Atom-level features ===
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
    atom_dict["ref_space_uid"] = np.array(ref_space_uid, dtype=np.int64)

    # === Bond-level features ===
    # TODO: Remap token indices to cropped tokens
    # e.g., [0, 3, 4, 5, 8] -> [0, 1, 2, 3, 4]
    bond_token_index = token_map[bond_dict["token_index"]]
    assert (bond_token_index != -1).all(), (
        "Bond token indices contain invalid values after mapping."
    )
    bond_dict["token_index"] = bond_token_index

    # Indicate whether the bond atoms belong to polymer or ligand
    token1, token2 = bond_dict["token_index"][:, 0], bond_dict["token_index"][:, 1]
    is_ligand1 = token_dict["chain_type"][token1] == C.chain.ChainType.LIGAND.value
    is_ligand2 = token_dict["chain_type"][token2] == C.chain.ChainType.LIGAND.value
    bond_dict["is_polymer_ligand"] = ((~is_ligand1) & is_ligand2) | (
        is_ligand1 & (~is_ligand2)
    )
    bond_dict["is_ligand_ligand"] = is_ligand1 & is_ligand2

    # === Sequence-level features ===
    seq_dict = struct.sequence.to_dict()
    seq_dict["pad_mask"] = np.ones(len(struct.sequence), dtype=np.bool_)

    # === Convert to tensors ===
    chain_layout = ChainTensor(**{k: torch.from_numpy(v) for k, v in chain_dict.items()})
    token_layout = TokenTensor(**{k: torch.from_numpy(v) for k, v in token_dict.items()})
    atom_layout = AtomTensor(**{k: torch.from_numpy(v) for k, v in atom_dict.items()})
    bond_layout = BondTensor(**{k: torch.from_numpy(v) for k, v in bond_dict.items()})
    sequence_layout = SequenceTensor(
        **{k: torch.from_numpy(v) for k, v in seq_dict.items()}
    )

    # === Constraint-level features ===
    # TODO: Remap token indices to cropped tokens
    # e.g., [0, 3, 4, 5, 8] -> [0, 1, 2, 3, 4]
    constraint_token_index = token_map[constraint_dict["token_index"]]
    assert (constraint_token_index != -1).all(), (
        "Constraint token indices contain invalid values after mapping."
    )
    constraint_dict["token_index"] = constraint_token_index
    constraint_layout = ConstraintTensor(
        **{k: torch.from_numpy(v) for k, v in constraint_dict.items()}
    )

    folding_input = FoldingInput(
        chain=chain_layout,
        token=token_layout,
        atom=atom_layout,
        bond=bond_layout,
        sequence=sequence_layout,
        constraint=constraint_layout,
    )
    return folding_input
