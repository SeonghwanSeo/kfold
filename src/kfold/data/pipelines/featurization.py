import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import kfold.constants as C
from kfold.data.types.model_input import (
    AtomTensor,
    BondTensor,
    ChainTensor,
    FoldingInput,
    PretrainedTensor,
    TokenTensor,
)
from kfold.data.types.tokenized import TokenizedStructure
from kfold.data.utils import frame_utils

# Assume pre-computed embeddings are post-norm values.
DEFAULT_EMBEDDING_DTYPE = torch.float16


def parse_residue_map(residue_map: str) -> tuple[int, int, int, int]:
    """Parse residue map string into start and end indices.
    Example:
        "1:100->5:104" -> (0, 100, 4, 104)
    """
    res_range, emb_range = residue_map.split("->")
    res_st, res_end = map(int, res_range.split(":"))
    emb_st, emb_end = map(int, emb_range.split(":"))
    if res_end - res_st != emb_end - emb_st:
        raise ValueError(
            f"Residue range length mismatch: "
            f"residue range {res_st}:{res_end} cannot be mapped to "
            f"embedding range {emb_st}:{emb_end}."
        )
    # Convert to 0-based indexing
    # 1:100 means residues 1 to 100 inclusive -> coords[0:100]
    res_st, emb_st = res_st - 1, emb_st - 1
    return res_st, res_end, emb_st, emb_end


def load_embedding_from_file(emb_path: Path) -> torch.Tensor:
    """Load embedding tensor from a file."""
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    format = emb_path.suffix
    if format == ".npy":
        embedding_array = np.load(emb_path, mmap_mode="r")
        embedding_tensor = torch.from_numpy(embedding_array)
    elif format in (".pt", ".pth"):
        embedding_tensor = torch.load(emb_path, "cpu", weights_only=True)
    else:
        raise ValueError(f"Unsupported embedding file format: {format}")
    return embedding_tensor.to(DEFAULT_EMBEDDING_DTYPE)


class InputFeaturizer:
    """A class for featurizing tokenized structures into model input features."""

    def __init__(
        self,
        seq_embedding_dim: int | None = None,
        struct_embedding_dim: int | None = None,
    ) -> None:
        """Initialize the InputFeaturizer.

        Parameters
        ----------
        seq_embedding_dim : int | None, optional
            Dimension of the sequence embedding.
        struct_embedding_dim : int | None, optional
            Dimension of the structure embedding.
        """
        # Precomputed embeddings
        self.seq_embedding_dim: int | None = seq_embedding_dim
        self.struct_embedding_dim: int | None = struct_embedding_dim

    def __call__(
        self,
        struct: TokenizedStructure,
        seq_embeddings: dict[int, dict] | None = None,
        struct_embeddings: dict[int, dict] | None = None,
        rng: np.random.Generator | None = None,
    ) -> FoldingInput:
        return self.run(struct, seq_embeddings, struct_embeddings, rng)

    def run(
        self,
        struct: TokenizedStructure,
        seq_embeddings: dict[int, dict] | None = None,
        struct_embeddings: dict[int, dict] | None = None,
        rng: np.random.Generator | None = None,
    ) -> FoldingInput:
        """Featurize a tokenized structure into model input features.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure to featurize.
        seq_embeddings : dict[int, dict] | None
            Sequence embedding information per entity_id.
            - path: Path to the embedding file.
            - residue_map: str indicating how to map residues (e.g., "1:24->5:20")
        struct_embeddings : dict[int, dict] | None
            Structure embedding information per entity_id.
            - path: Path to the embedding file.
            - residue_map: str indicating how to map residues (e.g., "1:24->5:20")
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
            f_input, seq_embeddings, struct_embeddings
        )

        return f_input_upd

    def to_folding_input(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator,
    ) -> FoldingInput:
        """Convert the tokenized structure to model input features."""
        return featurize_structure(struct, rng=rng)

    def add_precomputed_embedding(
        self,
        f_input: FoldingInput,
        seq_embeddings: dict[int, dict] | None,
        struct_embeddings: dict[int, dict] | None,
    ) -> FoldingInput:
        """Add pre-trained embeddings to the model input from pre-computed files.

        Parameters
        ----------
        f_input : model_input.FoldingInput
            The model input to add pretrained features.
        seq_embedding_paths : dict[int, Path] | None
            Mapping from entity_id to file path of the pre-computed sequence embedding.
        struct_embedding_paths : dict[int, Path] | None
            Mapping from entity_id to file path of the pre-computed structure embedding.
        rng : np.random.Generator
            Random number generator for augmentation.

        Returns
        -------
        f_input_upd: FoldingInput
            The featurized model input with pretrained features added.
        """

        pretrained_dict: dict[str, torch.Tensor] = {}
        if self.seq_embedding_dim is not None:
            assert self.seq_embedding_dim > 0, "seq_embedding_dim must be positive."
            assert seq_embeddings is not None, (
                "seq_embeddings must be provided when seq_embedding_dim is set."
            )
            pretrained_dict["sequence_embedding"] = load_pretrained_sequence_embedding(
                f_input,
                embedding_info=seq_embeddings,
                embedding_dim=self.seq_embedding_dim,
            )
        else:
            assert seq_embeddings is None or len(seq_embeddings) == 0, (
                "seq_embedding_paths must be None when seq_embedding_dim is not set."
            )

        if self.struct_embedding_dim is not None:
            assert self.struct_embedding_dim > 0, "struct_embedding_dim must be positive."
            assert struct_embeddings is not None, (
                "struct_embeddings must be provided when struct_embedding_dim is set."
            )
            pretrained_dict["structure_embedding"] = load_pretrained_structure_embedding(
                f_input,
                embedding_info=struct_embeddings,
                embedding_dim=self.struct_embedding_dim,
            )
        else:
            assert struct_embeddings is None or len(struct_embeddings) == 0, (
                "struct_embedding_paths must be None when "
                "struct_embedding_dim is not set."
            )

        # Replace the pretrained features in FoldingInput
        new_pretrained = f_input.pretrained.copy_with(**pretrained_dict)
        return f_input.copy_with(pretrained=new_pretrained)


def featurize_structure(
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
    atom_dict["pad_mask"] = np.ones((num_total_atoms,), dtype=np.bool_)  # Remove padding

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

    # === placeholder for pretrained embeddings === #
    pretrained_dict = {
        "sequence_embedding": np.empty((num_tokens, 0), dtype=np.float32),
        "structure_embedding": np.empty((num_tokens, 0), dtype=np.float32),
        "pad_mask": token_dict["pad_mask"],
    }

    # === Convert to tensors ===
    chain_layout = ChainTensor(**{k: torch.from_numpy(v) for k, v in chain_dict.items()})

    token_layout = TokenTensor(**{k: torch.from_numpy(v) for k, v in token_dict.items()})

    atom_layout = AtomTensor(**{k: torch.from_numpy(v) for k, v in atom_dict.items()})

    bond_layout = BondTensor(**{k: torch.from_numpy(v) for k, v in bond_dict.items()})

    pretrained_layout = PretrainedTensor(
        **{k: torch.from_numpy(v) for k, v in pretrained_dict.items()}
    )

    # === Before returning, compute ligand frames inplace === #
    frame_utils.compute_ligand_frames_inplace(token_layout, atom_layout, chain_layout)

    folding_input = FoldingInput(
        chain=chain_layout,
        token=token_layout,
        atom=atom_layout,
        bond=bond_layout,
        pretrained=pretrained_layout,
    )
    return folding_input


def load_pretrained_sequence_embedding(
    f_input: FoldingInput,
    embedding_info: dict[int, dict],
    embedding_dim: int,
) -> torch.Tensor:
    """Load pre-trained sequence embedding from a file.
    NOTE: the sequence embedding is always contains full sequence,
    while structure embedding may only contain partial residues in an apo structure.

    Parameters
    ----------
    f_input : FoldingInput
        The model input containing chain and token layouts.
    embedding_info : dict[int, dict]
        Mapping from entity_id to file path of the pre-computed embedding.
    embedding_dim : int
        Dimension of the embedding.

    Returns
    -------
    embedding : torch.Tensor
        Loaded embedding tensor of shape [Ntoken, Nfeat].
    """
    cached_embeddings: dict[Path, torch.Tensor] = {}

    # Sanity check: token asym_id should match chain asym_id
    assert torch.all(
        f_input.chain.asym_id.repeat_interleave(f_input.chain.num_tokens)
        == f_input.token.asym_id
    ), "Token asym_id does not match chain asym_id."

    entity_ids: list[int] = f_input.chain.entity_id.tolist()
    chain_types: list[C.ChainType] = [
        C.ChainType(ct) for ct in f_input.chain.chain_type.tolist()
    ]

    num_chain_tokens = f_input.chain.num_tokens.tolist()
    token_starts = np.cumsum([0] + num_chain_tokens[:-1]).tolist()  # [num_chains,]

    default_dtype = DEFAULT_EMBEDDING_DTYPE
    out = torch.zeros((f_input.num_tokens, embedding_dim), dtype=default_dtype)

    # NOTE: the tokens are ordered by chains.
    for cidx in range(f_input.num_chains):
        entity_id = entity_ids[cidx]
        ctype = chain_types[cidx]

        token_st = token_starts[cidx]
        token_num = num_chain_tokens[cidx]
        token_end = token_st + token_num
        if token_num == 0:
            continue

        if not ctype.is_polymer:
            # Currently we only support sequence embeddings for polymer chains.
            continue

        # Get embedding info for this entity_id
        assert entity_id in embedding_info, (
            f"No embedding info found for entity_id {entity_id}."
        )
        emb_path = Path(embedding_info[entity_id]["path"])
        if not emb_path.exists():
            warnings.warn(
                f"Precomputed sequence embedding file not found: {emb_path}."
                f" Zero tensor is used instead.",
                UserWarning,
                stacklevel=2,
            )
            continue
        if emb_path not in cached_embeddings:
            cached_embeddings[emb_path] = load_embedding_from_file(emb_path)
        embedding_tensor = cached_embeddings[emb_path]

        # For polymer chains, we load embeddings according to the residue indices.
        residue_indices = f_input.token.residue_index[token_st:token_end]
        # NOTE: residue_index is 1-based indexing
        residue_indices = residue_indices - 1
        # Feed the embedding tensor to the corresponding token positions
        out[token_st:token_end] = embedding_tensor[residue_indices]

    return out


def load_pretrained_structure_embedding(
    f_input: FoldingInput,
    embedding_info: dict[int, dict],
    embedding_dim: int,
) -> torch.Tensor:
    """Load pre-trained structure embedding from a file.

    Parameters
    ----------
    f_input : FoldingInput
        The model input containing chain and token layouts.
    embedding_info : dict[int, dict]
        Mapping from entity_id to file path of the pre-computed embedding.
    embedding_dim : int
        Dimension of the embedding.

    Returns
    -------
    embedding : torch.Tensor
        Loaded embedding tensor of shape [Ntoken, Nfeat].
    """
    cached_embeddings: dict[Path, torch.Tensor] = {}

    # Sanity check: token asym_id should match chain asym_id
    assert torch.all(
        f_input.chain.asym_id.repeat_interleave(f_input.chain.num_tokens)
        == f_input.token.asym_id
    ), "Token asym_id does not match chain asym_id."

    entity_ids: list[int] = f_input.chain.entity_id.tolist()
    chain_types: list[C.ChainType] = [
        C.ChainType(ct) for ct in f_input.chain.chain_type.tolist()
    ]

    num_chain_tokens = f_input.chain.num_tokens.tolist()
    token_starts = np.cumsum([0] + num_chain_tokens[:-1]).tolist()  # [num_chains,]

    default_dtype = DEFAULT_EMBEDDING_DTYPE
    out = torch.zeros((f_input.num_tokens, embedding_dim), dtype=default_dtype)

    # NOTE: the tokens are ordered by chains.
    for cidx in range(f_input.num_chains):
        entity_id = entity_ids[cidx]
        ctype = chain_types[cidx]

        token_st = token_starts[cidx]
        token_num = num_chain_tokens[cidx]
        token_end = token_st + token_num
        if token_num == 0:
            continue

        if not ctype.is_protein:
            # Apo structure embeddings are only supported for protein chains.
            continue

        if entity_id not in embedding_info:
            # If no embedding file found, use zero tensor.
            # This happens when apo structure is not available for a protein chain.
            warnings.warn(
                f"No precomputed structure embedding: {entity_id}."
                f" Zero tensor is used instead.",
                UserWarning,
                stacklevel=2,
            )
            continue

        entity_info = embedding_info[entity_id]

        # Get embedding info for this entity_id
        emb_path = Path(embedding_info[entity_id]["path"])
        if not emb_path.exists():
            warnings.warn(
                f"Precomputed structure embedding file not found: {emb_path}."
                f" Zero tensor is used instead.",
                UserWarning,
                stacklevel=2,
            )
            continue
        if emb_path not in cached_embeddings:
            cached_embeddings[emb_path] = load_embedding_from_file(emb_path)
        embedding_tensor = cached_embeddings[emb_path]

        # Trim embedding to the specified range
        if "residue_map" in entity_info:
            res_st, res_end, emb_st, emb_end = parse_residue_map(
                entity_info["residue_map"]
            )
            embedding_tensor = embedding_tensor[emb_st:emb_end]
        else:
            res_st, res_end = 0, embedding_tensor.shape[0]

        # For polymer chains, we load embeddings according to the residue indices.
        residue_indices = f_input.token.residue_index[token_st:token_end]
        # NOTE: residue_index is 1-based indexing
        residue_indices = residue_indices - 1
        residue_mask = (residue_indices >= res_st) & (residue_indices < res_end)

        if residue_mask.any():
            valid_residue_indices = residue_indices[residue_mask] - res_st
            out_slice = out[token_st:token_end]
            out_slice[residue_mask] = embedding_tensor[valid_residue_indices]
        else:
            # All residues are outside the specified range
            # Here we simply keep zero embeddings.
            pass

    return out
