from collections import defaultdict

import numpy as np
import torch

import kfold.constants as C
from kfold.data.types.model_input import (
    AtomTensor,
    BondTensor,
    ChainTensor,
    FoldingInput,
    SequenceTensor,
    TokenTensor,
)
from kfold.data.types.tokenized import TokenizedStructure
from kfold.data.utils import frame_utils


class InputFeaturizer:
    """A class for featurizing tokenized structures into model input features."""

    def __call__(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator | None = None,
    ) -> FoldingInput:
        return self.run(struct, rng)

    def run(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator | None = None,
    ) -> FoldingInput:
        """Featurize a tokenized structure into model input features.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure to featurize.
        rng : np.random.Generator
            Random number generator for augmentation.

        Returns
        -------
        f_input_upd: FoldingInput
            The featurized model input
        """
        rng = rng or np.random.default_rng()
        return self.to_folding_input(struct, rng)

    def to_folding_input(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator,
    ) -> FoldingInput:
        """Featurize a tokenized structure into model input features.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure to featurize.
        rng : np.random.Generator
            Random number generator for augmentation.

        Returns
        -------
        f_input: FoldingInput
            The featurized model input
        """
        rng = rng or np.random.default_rng()

        # =========================================== #
        # ========== Extract raw features =========== #
        # =========================================== #

        chain_data = struct.chain
        token_data = struct.token
        atom_data = struct.atom
        bond_data = struct.bond

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
            k: v[atom_to_token, atom_in_token_idx]  # Fancy indexing - no loop!
            for k, v in atom_data.to_dict().items()
        }
        atom_dict["label_coords"] = np.nan_to_num(atom_dict["label_coords"], nan=0.0)
        atom_dict["apo_coords"] = np.nan_to_num(atom_dict["apo_coords"], nan=0.0)
        atom_dict["ref_pos"] = np.nan_to_num(atom_dict["ref_pos"], nan=0.0)
        atom_dict["token_index"] = atom_to_token
        atom_dict["pad_mask"] = np.ones(
            (num_total_atoms,), dtype=np.bool_
        )  # Remove padding

        # === Bond-level features ===
        num_bonds = bond_data.length
        bond_dict: dict[str, np.ndarray] = bond_data.to_dict()
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
        token_dict["token_index"] = np.arange(num_tokens, dtype=np.int64)
        token_dict["pocket_contact_type"] = np.full(
            (num_tokens,), C.constraint.ConstraintType.UNSPECIFIED, dtype=np.int64
        )

        # Add frame information
        token_dict["frames_index"] = np.zeros((num_tokens, 3), dtype=np.int64)
        token_dict["frames_mask"] = np.zeros((num_tokens,), dtype=np.bool_)
        for tidx in range(num_tokens):
            if not token_dict["is_standard"][tidx]:
                # Skip non-standard residues
                continue
            res_name = C.residue.residue_id_to_name[int(token_data.res_type[tidx])]
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
        token_dict["frames_index"] = (
            token_dict["frames_index"] + atom_offset[:, np.newaxis]
        )

        # add disto/center coords
        # HACK: we assume there is only one holo coordinate set.
        token_dict["disto_coords"] = atom_dict["label_coords"][token_dict["disto_index"]]
        token_dict["center_coords"] = atom_dict["label_coords"][
            token_dict["center_index"]
        ]

        # Masks indicating whether the center/disto atoms are resolved
        token_dict["center_mask"] = atom_dict["resolved_mask"][token_dict["center_index"]]
        token_dict["disto_mask"] = atom_dict["resolved_mask"][token_dict["disto_index"]]

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
        original_token_index = token_dict["org_token_index"]
        token_index = token_dict["token_index"]
        token_map = np.zeros(original_token_index.max() + 1, dtype=np.int64) - 1
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

        # === Sequence-level features ===
        seq_dict = struct.sequence.to_dict()
        seq_dict["pad_mask"] = np.ones_like(seq_dict["input_id"], dtype=np.bool_)

        # === Convert to tensors ===
        chain_layout = ChainTensor(
            **{k: torch.from_numpy(v) for k, v in chain_dict.items()}
        )
        token_layout = TokenTensor(
            **{k: torch.from_numpy(v) for k, v in token_dict.items()}
        )
        atom_layout = AtomTensor(**{k: torch.from_numpy(v) for k, v in atom_dict.items()})
        bond_layout = BondTensor(**{k: torch.from_numpy(v) for k, v in bond_dict.items()})
        sequence_layout = SequenceTensor(
            **{k: torch.from_numpy(v) for k, v in seq_dict.items()}
        )

        # === Before returning, compute ligand frames inplace === #
        frame_utils.compute_ligand_frames_inplace(token_layout, atom_layout, chain_layout)

        folding_input = FoldingInput(
            chain=chain_layout,
            token=token_layout,
            atom=atom_layout,
            bond=bond_layout,
            sequence=sequence_layout,
        )
        return folding_input
