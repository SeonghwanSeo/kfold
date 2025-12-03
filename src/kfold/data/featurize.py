import os
from collections import defaultdict

import numpy as np
import torch

import kfold.constants as C
from kfold.constants.residue import residue_index_to_name
from kfold.utils.geometry.random_augment import center_random_augmentation, do_centering

from . import model_input, structure
from .utils import frame_utils

__all__ = ["featurize_structure", "add_pretrained_embeddings"]

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
        Apo structure coordinates of shape [Natom, 3].
    mask : np.ndarray
        Mask indicating valid atoms of shape [Natom].
    chain_sizes : np.ndarray
        Array of number of atoms per each chain.
    rng : np.random.Generator | None
        Random number generator for augmentation.

    Returns
    -------
    augmented_apo_coords : np.ndarray
        Augmented apo structure coordinates of shape [Natom, 3].
    """
    # Apply random rotation and translation per apo coords
    new_coords = np.zeros_like(apo_coords)
    start_idx = 0
    for natom in chain_sizes:
        end_idx = start_idx + natom
        chain_coords = apo_coords[start_idx:end_idx]
        chain_mask = mask[start_idx:end_idx]
        new_coords[start_idx:end_idx] = center_random_augmentation(
            chain_coords, chain_mask, rng=rng
        )
        start_idx = end_idx
    return new_coords


def featurize_structure(
    struct: structure.TokenizedStructure,
    augment_ref_pos: bool = True,
    augment_apo: bool = True,
    synchronize_ref_pos_augmentation: bool = False,
    rng: np.random.Generator | None = None,
    **kwargs,
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

    Returns
    -------
    f_input: FoldingInput
        The featurized model input
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
    atom_dict["token_index"] = atom_to_token
    atom_dict["pad_mask"] = np.ones((num_total_atoms,), dtype=np.bool_)  # Remove padding

    # Random sample the ground truth holo coords if multiple holo coords are given.
    # [Natom, Nholo, 3] -> [Natom, 3]
    label_coords = atom_dict.pop("coords")  # Rename for clarity
    n_holo = label_coords.shape[-2]
    assert n_holo == 1, "Currently only single holo coordinate is supported."
    sampled_idx = rng and rng.integers(0, n_holo) or np.random.randint(0, n_holo)
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

    # TODO: If we use CCD, use random ETKDG conformers here.
    # [Natom, Napo, 3] -> [Natom, 3]
    apo_coords = atom_dict.pop("apo_coords")
    apo_mask = atom_dict.pop("apo_mask")
    n_apo = apo_coords.shape[-2]
    assert n_apo == 1, "Currently only single apo coordinate is supported."
    sampled_idx = rng and rng.integers(0, n_holo) or np.random.randint(0, n_holo)
    apo_coords = apo_coords[:, sampled_idx, :]
    apo_mask = apo_mask[:, sampled_idx]
    if augment_apo:
        # Augment apo structure (chain-wise)
        num_atoms_per_chains = chain_dict["num_atoms"]
        apo_coords = do_augment_apo_structure(
            apo_coords=apo_coords,
            mask=apo_mask,
            chain_sizes=num_atoms_per_chains,
            rng=rng,
        )
    apo_coords = do_centering(apo_coords, apo_mask)
    atom_dict["apo_coords"] = apo_coords
    atom_dict["apo_mask"] = apo_mask

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
    prefix: str,
    embedding_dim: int,
) -> torch.Tensor:
    """Load pre-trained embedding from a file.

    Parameters
    ----------
    f_input : model_input.FoldingInput
        The model input containing chain and token layouts.
    prefix : str
        Prefix for the path to pre-computed embeddings.
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
        if entity_id not in cached_embeddings:
            # Load from file
            filepath = f"{prefix}{entity_id}_{chain_type.name.lower()}.pt"
            # TODO: In future, we may want to enforce the existence of embedding files
            # for all chain types.
            if not os.path.exists(filepath):
                # HACK: (SeonghwanSeo) Print warning only for protein chains, since other
                # chain types are not prepared yet. In future, we may want to enforce the
                # existence of embedding files for all chain types.
                if chain_type is C.ChainType.PROTEIN:
                    import warnings

                    warnings.warn(
                        "Precomputed Embedding file not found for protein chain: "
                        f"{filepath}. Using zero tensor as placeholder.",
                        UserWarning,
                    )

                embedding_tensor = None
            else:
                embedding_tensor = torch.load(filepath, "cpu", weights_only=True)
            cached_embeddings[entity_id] = embedding_tensor
        else:
            embedding_tensor = cached_embeddings[entity_id]

        # Extract embeddings for the tokens in this chain
        chain_token_mask = f_input.token.asym_id == asym_id
        num_tokens_in_chain = int(chain_token_mask.sum())
        if embedding_tensor is not None:
            if chain_type in (C.ChainType.PROTEIN, C.ChainType.DNA, C.ChainType.RNA):
                # For polymer chains, we load embeddings according to the residue indices.
                residue_indices = f_input.token.residue_index[chain_token_mask]
                # NOTE: residue_index is starting from 1.
                assert (residue_indices >= 1).all(), "Residue indices should be positive."
                chain_embeddings = embedding_tensor[residue_indices - 1]
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


def add_pretrained_embeddings(
    f_input: model_input.FoldingInput,
    seq_embedding_prefix: str | None = None,
    struct_embedding_prefix: str | None = None,
    seq_embedding_dim: int | None = None,
    struct_embedding_dim: int | None = None,
) -> model_input.FoldingInput:
    """Add pre-trained features to the model input.

    Parameters
    ----------
    f_input : model_input.FoldingInput
        The model input to add pretrained features.
    seq_embedding_prefix : str | None, optional
        Prefix for the path to pre-computed sequence embeddings.
        Example of the filename:
            "embeddings/esm/6o/6oim/6oim_2_protein.pt"
            where 2 is the entity index.
        Example of the prefix:
            "embeddings/esm/6o/6oim/6oim_"
    struct_embedding_prefix : str | None, optional
        Prefix for the path to pre-computed structure embeddings.
        Example of the filename:
            "embeddings/struct/10/10gs/10gs_1_protein.pt"
            where 1 is the entity index.
        Example of the prefix:
            "embeddings/struct/10/10gs/10gs_"
    seq_embedding_dim : int | None, optional
        Dimension of the sequence embedding.
    struct_embedding_dim : int | None, optional
        Dimension of the structure embedding.

    Returns
    -------
    f_input_upd: FoldingInput
        The featurized model input with pretrained features added.
    """
    if seq_embedding_prefix is None and struct_embedding_prefix is None:
        return f_input

    pretrained_dict = {}
    if seq_embedding_prefix is not None:
        assert seq_embedding_dim is not None
        seq_embedding = load_pretrained_embedding(
            f_input, seq_embedding_prefix, seq_embedding_dim
        )
        pretrained_dict["sequence_embedding"] = seq_embedding
    else:
        pretrained_dict["sequence_embedding"] = f_input.pretrained.sequence_embedding

    if struct_embedding_prefix is not None:
        assert struct_embedding_dim is not None
        struct_embedding = load_pretrained_embedding(
            f_input, struct_embedding_prefix, struct_embedding_dim
        )
        pretrained_dict["structure_embedding"] = struct_embedding
    else:
        pretrained_dict["structure_embedding"] = f_input.pretrained.structure_embedding

    pretrained_dict["pad_mask"] = f_input.pretrained.pad_mask

    pretrained_layout = model_input.PretrainedLayout(
        **{k: v for k, v in pretrained_dict.items()}
    )

    # Return updated FoldingInput
    f_input_upd = model_input.FoldingInput(
        chain=f_input.chain,
        token=f_input.token,
        atom=f_input.atom,
        bond=f_input.bond,
        pretrained=pretrained_layout,
    )
    return f_input_upd
