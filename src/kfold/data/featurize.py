import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import kfold.constants as C
from kfold.utils.geometry.random_augment import center_random_augmentation, do_centering

from . import model_input, structure
from .utils import frame_utils


class InputFeaturizer:
    """A class for featurizing tokenized structures into model input features."""

    def __init__(
        self,
        augment_ref_pos: bool = True,
        synchronize_ref_pos_augmentation: bool = False,
        seq_embedding_dim: int | None = None,
        struct_embedding_dim: int | None = None,
    ) -> None:
        """Initialize the InputFeaturizer.

        Parameters
        ----------
        augment_ref_pos : bool, optional
            Whether to apply random augmentation to ref_pos,
        synchronize_ref_pos_augmentation : bool
            Whether to synchronize the random augmentation for ref_pos across all atoms,
            default: False.
            NOTE: This flag is just for running Boltz1 in this repository. (Boltz1: True)
        seq_embedding_dim : int | None, optional
            Dimension of the sequence embedding.
        struct_embedding_dim : int | None, optional
            Dimension of the structure embedding.
        """
        # Featurization arguments
        self.augment_ref_pos: bool = augment_ref_pos
        self.synchronize_ref_pos_augmentation: bool = synchronize_ref_pos_augmentation

        # Precomputed embeddings
        self.seq_embedding_dim: int | None = seq_embedding_dim
        self.struct_embedding_dim: int | None = struct_embedding_dim

    def __call__(
        self,
        struct: structure.TokenizedStructure,
        seq_embedding_paths: dict[int, Path] | None = None,
        struct_embedding_paths: dict[int, Path] | None = None,
        rng: np.random.Generator | None = None,
    ) -> model_input.FoldingInput:
        return self.run(struct, seq_embedding_paths, struct_embedding_paths, rng)

    def run(
        self,
        struct: structure.TokenizedStructure,
        seq_embedding_paths: dict[int, Path] | None = None,
        struct_embedding_paths: dict[int, Path] | None = None,
        rng: np.random.Generator | None = None,
    ) -> model_input.FoldingInput:
        """Featurize a tokenized structure into model input features.

        Parameters
        ----------
        struct : structure.TokenizedStructure
            The tokenized structure to featurize.
        seq_embedding_paths : dict[int, Path] | None
            Mapping from entity_id to file path of the pre-computed sequence embedding.
        struct_embedding_paths : dict[int, Path] | None
            Mapping from entity_id to file path of the pre-computed structure embedding.
        rng : np.random.Generator
            Random number generator for augmentation.

        Returns
        -------
        f_input_upd: FoldingInput
            The featurized model input
        """
        rng = rng or np.random.default_rng()

        # Convert to model input features
        f_input = self.to_folding_input(struct, rng)

        # Add pre-computed embeddings
        f_input_upd = self.add_precomputed_embedding(
            f_input, seq_embedding_paths, struct_embedding_paths
        )

        return f_input_upd

    def to_folding_input(
        self,
        struct: structure.TokenizedStructure,
        rng: np.random.Generator,
    ) -> model_input.FoldingInput:
        """Convert the tokenized structure to model input features."""
        return featurize_structure(
            struct,
            augment_ref_pos=self.augment_ref_pos,
            synchronize_ref_pos_augmentation=self.synchronize_ref_pos_augmentation,
            rng=rng,
        )

    def add_precomputed_embedding(
        self,
        f_input: model_input.FoldingInput,
        seq_embedding_paths: dict[int, Path] | None = None,
        struct_embedding_paths: dict[int, Path] | None = None,
    ) -> model_input.FoldingInput:
        """Add pre-trained embeddings to the model input from pre-computed files.

        Parameters
        ----------
        f_input : model_input.FoldingInput
            The model input to add pretrained features.
        seq_embedding_paths : dict[int, Path] | None
            Mapping from entity_id to file path of the pre-computed sequence embedding.
        struct_embedding_paths : dict[int, Path] | None
            Mapping from entity_id to file path of the pre-computed structure embedding.

        Returns
        -------
        f_input_upd: FoldingInput
            The featurized model input with pretrained features added.
        """

        pretrained_dict: dict[str, torch.Tensor] = {}
        if self.seq_embedding_dim is not None:
            assert self.seq_embedding_dim > 0, "seq_embedding_dim must be positive."
            assert seq_embedding_paths is not None, (
                "seq_embedding_paths must be provided when seq_embedding_dim is set."
            )
            pretrained_dict["sequence_embedding"] = load_pretrained_embedding(
                f_input, seq_embedding_paths, self.seq_embedding_dim
            )
        else:
            assert seq_embedding_paths is None or len(seq_embedding_paths) == 0, (
                "seq_embedding_paths must be None when seq_embedding_dim is not set."
            )

        if self.struct_embedding_dim is not None:
            assert self.struct_embedding_dim > 0, "struct_embedding_dim must be positive."
            assert struct_embedding_paths is not None, (
                "struct_embedding_paths must be provided"
                " when struct_embedding_dim is set."
            )
            pretrained_dict["structure_embedding"] = load_pretrained_embedding(
                f_input, struct_embedding_paths, self.struct_embedding_dim
            )
        else:
            assert struct_embedding_paths is None or len(struct_embedding_paths) == 0, (
                "struct_embedding_paths must be None when "
                "struct_embedding_dim is not set."
            )

        # Replace the pretrained features in FoldingInput
        new_pretrained = f_input.pretrained.copy_with(**pretrained_dict)
        return f_input.copy_with(pretrained=new_pretrained)


# === Helper functions === #
def do_augment_ref_pos(
    ref_pos: np.ndarray,
    mask: np.ndarray,
    conformer_sizes: list[int],
    rng: np.random.Generator,
    synchronize: bool = False,
) -> np.ndarray:
    """Augment reference positions with random translation and rotation.

    Parameters
    ----------
    ref_pos : np.ndarray
        Reference positions of shape [Natom, 3].
    mask : np.ndarray
        Mask indicating valid atoms of shape [Natom,].
    conformer_sizes : list[int]
        List of number of atoms per each conformer.
    rng : np.random.Generator
        Random number generator for augmentation.
    synchronize : bool
        Whether to synchronize the random augmentation across all atoms.

    Returns
    -------
    augmented_ref_pos : np.ndarray
        Augmented reference positions of shape [Natom, 3].

    """

    if synchronize:
        # NOTE: Just for running Boltz. This flag should be False.
        return center_random_augmentation(ref_pos, mask, rng=rng)
    else:
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


def featurize_structure(
    struct: structure.TokenizedStructure,
    augment_ref_pos: bool = True,
    synchronize_ref_pos_augmentation: bool = False,
    rng: np.random.Generator | None = None,
) -> model_input.FoldingInput:
    """Featurize a tokenized structure into model input features.

    Parameters
    ----------
    struct : structure.TokenizedStructure
        The tokenized structure to featurize.
    augment_ref_pos : bool
        Whether to apply random augmentation to ref_pos,
    synchronize_ref_pos_augmentation : bool
        Whether to synchronize the random augmentation for ref_pos across all atoms,
    rng : np.random.Generator
        Random number generator for augmentation.

    Returns
    -------
    f_input: FoldingInput
        The featurized model input
    """
    rng = rng or np.random.default_rng()

    def cast(data: np.ndarray) -> np.ndarray:
        if np.issubdtype(data.dtype, np.floating):
            return data.astype(np.float32, copy=False)
        elif np.issubdtype(data.dtype, np.integer):
            return data.astype(np.int64, copy=False)
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
    atom_dict["apo_coords"] = np.nan_to_num(atom_dict["apo_coords"], nan=0.0)
    atom_dict["ref_pos"] = np.nan_to_num(atom_dict["ref_pos"], nan=0.0)
    atom_dict["token_index"] = atom_to_token
    atom_dict["pad_mask"] = np.ones((num_total_atoms,), dtype=np.bool_)  # Remove padding

    # Random sample the ground truth holo coords if multiple holo coords are given.
    # [Natom, Nholo, 3] -> [Natom, 3]
    label_coords = atom_dict.pop("coords")  # Rename for clarity
    n_holo = label_coords.shape[-2]
    assert n_holo == 1, "Currently only single holo coordinate is supported."
    sampled_idx = rng.integers(low=0, high=n_holo)
    label_coords = label_coords[:, sampled_idx, :]

    # Centering the ground truth coords
    label_coords = do_centering(label_coords, atom_dict["resolved_mask"])
    atom_dict["label_coords"] = label_coords

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
    token_dict["token_index"] = np.arange(num_tokens, dtype=np.int64)
    token_dict["pocket_contact_type"] = np.full(
        (num_tokens,), C.constraint.ConstraintType.UNSPECIFIED, dtype=np.int64
    )

    # Add frame information
    token_dict["frames_index"] = np.zeros((num_tokens, 3), dtype=np.int64)
    token_dict["frames_mask"] = np.zeros((num_tokens,), dtype=np.bool_)
    for tidx in range(num_tokens):
        is_standard = token_dict["is_standard"][tidx]
        if not is_standard:
            # Skip non-standard residues
            continue
        res_name = C.residue.residue_id_to_name[int(token_data.res_type[tidx])]
        if res_name is C.residue.ResidueName.UNK:
            # Skip unknown residues
            continue
        num_atoms_in_res = token_dict["num_atoms"][tidx]
        if num_atoms_in_res >= 3:
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
    token_dict["disto_coords"] = atom_dict["label_coords"][token_dict["disto_index"]]
    token_dict["center_coords"] = atom_dict["label_coords"][token_dict["center_index"]]

    # Masks indicating whether the center/disto atoms are resolved
    token_dict["resolved_mask"] = (
        token_data.resolved_mask & atom_dict["resolved_mask"][token_dict["center_index"]]
    )
    token_dict["disto_mask"] = (
        token_data.resolved_mask & atom_dict["resolved_mask"][token_dict["disto_index"]]
    )

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

    # [Natom, Napo, 3] -> [Natom, 3], Napo must be 1 (sampled beforehand)
    apo_coords = atom_dict.pop("apo_coords")  # [Natom, Napo, 3]
    apo_mask = atom_dict.pop("apo_mask")  # [Natom, Napo]
    n_apo: int = apo_coords.shape[1]
    assert n_apo == 1, "Apo coordinates should be sampled beforehand."
    apo_coords = apo_coords[:, 0, :]
    apo_mask = apo_mask[:, 0]

    apo_coords = do_centering(apo_coords, apo_mask)
    atom_dict["apo_coords"] = apo_coords
    atom_dict["apo_mask"] = apo_mask

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

    # === placeholder for pretrained embeddings === #
    pretrained_dict = {
        "sequence_embedding": np.empty((num_tokens, 0), dtype=np.float32),
        "structure_embedding": np.empty((num_tokens, 0), dtype=np.float32),
        "pad_mask": token_dict["pad_mask"],
    }

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

    pretrained_layout = model_input.PretrainedLayout(
        **{k: torch.from_numpy(v) for k, v in pretrained_dict.items()}
    )

    # === Before returning, compute ligand frames inplace === #
    frame_utils.compute_ligand_frames_inplace(token_layout, atom_layout, chain_layout)

    folding_input = model_input.FoldingInput(
        chain=chain_layout,
        token=token_layout,
        atom=atom_layout,
        bond=bond_layout,
        pretrained=pretrained_layout,
    )
    return folding_input


def load_pretrained_embedding(
    f_input: model_input.FoldingInput,
    paths: dict[int, str | Path],
    embedding_dim: int,
) -> torch.Tensor:
    """Load pre-trained embedding from a file.

    Parameters
    ----------
    f_input : model_input.FoldingInput
        The model input containing chain and token layouts.
    paths : dict[int, str | Path]
        Mapping from entity_id to file path of the pre-computed embedding.
    embedding_dim : int
        Dimension of the embedding.

    Returns
    -------
    embedding : torch.Tensor
        Loaded embedding tensor of shape [Ntoken, Nfeat].
    """
    entity_ids = f_input.chain.entity_id
    cached_embeddings: dict[int, torch.Tensor | None] = {}

    embedding_tensors: list[torch.Tensor] = []

    # NOTE: the tokens are ordered by chains.
    for cidx in range(f_input.num_chains):
        entity_id = int(entity_ids[cidx].item())
        asym_id = f_input.chain.asym_id[cidx].item()
        chain_type = C.ChainType(int(f_input.chain.chain_type[cidx]))

        if entity_id in cached_embeddings:
            # Use cached embedding if already loaded
            embedding_tensor = cached_embeddings[entity_id]
        else:
            # Load embedding from file
            emb_path = paths.get(entity_id, None)
            if emb_path is not None and Path(emb_path).exists():
                embedding_tensor = torch.load(emb_path, "cpu", weights_only=True)
            else:
                embedding_tensor = None
            cached_embeddings[entity_id] = embedding_tensor

            if embedding_tensor is None:
                # HACK: (SeonghwanSeo) Print warning only for protein chains, since other
                # chain types are not prepared yet. In future, we may want to enforce the
                # existence of embedding files for all chain types.
                print_warning: bool = True  # For debugging purpose
                if print_warning and chain_type is C.ChainType.PROTEIN:
                    warnings.warn(
                        f"Precomputed Embedding file not found: {emb_path}."
                        f" Zero tensor is used instead.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Extract embeddings for the tokens in this chain
        chain_token_mask = f_input.token.asym_id == asym_id
        num_tokens_in_chain = int(chain_token_mask.sum())
        if embedding_tensor is not None:
            if chain_type in (C.ChainType.PROTEIN, C.ChainType.DNA, C.ChainType.RNA):
                # For polymer chains, we load embeddings according to the residue indices.
                residue_indices = f_input.token.residue_index[chain_token_mask]
                # NOTE: residue_index is 1-based indexing
                residue_indices = residue_indices - 1
                chain_embeddings = embedding_tensor[residue_indices]
            else:
                # For ligand, we ensure the number of tokens match.
                assert embedding_tensor.shape[0] == num_tokens_in_chain, (
                    f"Number of tokens in chain ({num_tokens_in_chain}) does not match "
                    f"the number of embeddings ({embedding_tensor.shape[0]}) for ligand."
                )
                chain_embeddings = embedding_tensor
        else:
            # If no embedding file found, use zero tensor.
            chain_embeddings = torch.zeros(
                (num_tokens_in_chain, embedding_dim), dtype=torch.float32
            )
        embedding_tensors.append(chain_embeddings)

    return torch.cat(embedding_tensors, dim=0)
