import dataclasses
from functools import cached_property
from typing import Self

import torch

import kfold.constants as C
from kfold.data.layout import TensorLayout
from kfold.utils.misc import check_tensor

__all__ = ["FoldingInput"]


# === Layout dataclasses (chain-level, token-level, atom-level, bond-level) === #
@dataclasses.dataclass(frozen=True)
class ChainTensor(TensorLayout):
    """Chain-level layout information.

    Shape: [Nchain, ...] or [B, Nchain, ...]

    Attributes
    ----------
    chain_type: torch.Tensor (long)
        Chain types of shape [Nchain,], indicating the type of each chain.
    entity_id: torch.Tensor (long)
        Entity IDs of shape [Nchain,], starting from 1.
    asym_id: torch.Tensor (long)
        Asymmetric unit IDs of shape [Nchain,], starting from 1.
    apo_uid: torch.Tensor (long)
        Apo rigid-group IDs of shape [Nchain,], starting from 1.
    sym_id: torch.Tensor (long)
        Symmetry IDs of shape [Nchain,], starting from 1.
    num_tokens: torch.Tensor (long)
        Number of tokens per chain of shape [Nchain,].
    num_residues: torch.Tensor (long)
        Number of residues per chain of shape [Nchain,].
    num_atoms: torch.Tensor (long)
        Number of atoms per chain of shape [Nchain,].
    pad_mask: torch.Tensor (bool)
        Mask tensor of shape [Nchain,], indicating valid chains.

    # NOTE: this layout might not be used in model.forward(),
    # but it may be useful for data processing and analysis.

    """

    chain_type: torch.Tensor  # [Nchain,], long
    entity_id: torch.Tensor  # [Nchain,], long
    asym_id: torch.Tensor  # [Nchain,], long
    apo_uid: torch.Tensor  # [Nchain,], long
    sym_id: torch.Tensor  # [Nchain,], long
    num_tokens: torch.Tensor  # [Nchain,], long
    num_residues: torch.Tensor  # [Nchain,], long
    num_atoms: torch.Tensor  # [Nchain,], long
    pad_mask: torch.Tensor  # [Nchain,], bool

    # === Batched layout === #
    @property
    def layout_shape(self) -> tuple[int, ...]:
        return self.pad_mask.shape

    @property
    def ndim_unbatched(self) -> int:
        """[ClassVar] The number of dimensions of the layout."""
        return 1

    def __post_init__(self):
        # shape: [Nchain,] or [B, Nchain]
        shape = self.layout_shape
        attributes = [
            ("chain_type", torch.long, shape),
            ("entity_id", torch.long, shape),
            ("asym_id", torch.long, shape),
            ("apo_uid", torch.long, shape),
            ("sym_id", torch.long, shape),
            ("num_tokens", torch.long, shape),
            ("num_residues", torch.long, shape),
            ("num_atoms", torch.long, shape),
            ("pad_mask", torch.bool, shape),
        ]
        for name, dtype, shape in attributes:
            check_tensor(getattr(self, name), name=name, dtype=dtype, shape=shape)

    def pad(self, *pad_shape: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        self._check_pad_input(pad_shape)

        if pad_shape == self.layout_shape:
            return self

        total_length = pad_shape[0]  # single dimension
        L = len(self)

        pad_values = {
            "chain_type": -1,
            "entity_id": -1,
            "asym_id": -1,
            "apo_uid": -1,
            "sym_id": -1,
            "num_tokens": -1,
            "num_residues": -1,
            "num_atoms": -1,
            "pad_mask": False,
        }

        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            to_shape = (total_length,) + tensor.shape[1:]
            padded_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:L] = tensor
            fields[name] = padded_tensor
        return self.from_dict(fields)

    @cached_property
    def is_protein(self) -> torch.Tensor:
        """Boolean tensor indicating whether the chain is protein."""
        return self.chain_type == C.chain.ChainType.PROTEIN.value

    @cached_property
    def is_dna(self) -> torch.Tensor:
        """Boolean tensor indicating whether the chain is dna."""
        return self.chain_type == C.chain.ChainType.DNA.value

    @cached_property
    def is_rna(self) -> torch.Tensor:
        """Boolean tensor indicating whether the chain is rna."""
        return self.chain_type == C.chain.ChainType.RNA.value

    @cached_property
    def is_ligand(self) -> torch.Tensor:
        """Boolean tensor indicating whether the chain is ligand."""
        return self.chain_type == C.chain.ChainType.LIGAND.value


@dataclasses.dataclass(frozen=True)
class TokenTensor(TensorLayout):
    """Token-level layout information.

    Attributes
    ----------
    chain_type: torch.Tensor (long)
        Chain types of shape [Ntoken,], indicating the type of each token.
    res_type: torch.Tensor (float32)
        Sequence tokens of shape [Ntoken, 32] (aatype, atom, ...)
        One-hot vector
    is_standard: torch.Tensor (bool)
        Boolean tensor of shape [Ntoken,], indicating whether the token is standard.
    num_atoms: torch.Tensor (long)
        Number of atoms per token of shape [Ntoken,].
    token_index: torch.Tensor (long)
        Token indices of shape [Ntoken,], mapping each token to its position.
    org_token_index: torch.Tensor (long)
        Original token indices of shape [Ntoken,], before cropping.
    residue_index: torch.Tensor (long)
        Residue indices of shape [Ntoken,], used for residue-level operations.
        Starting from 1 for each chain.
        example)
            4-len polymer(protein/RNA/DNA):
                residue_index: [1, 2, 3, 4]
            6-sized ligand:
                residue_index: [1, 1, 1, 1, 1, 1]
    seq_token_index: torch.Tensor  # [Ntoken,], long
        Sequence token indices of shape [Ntoken,], mapping each token to its position
        in the SequenceTensor.
    center_index: torch.Tensor (long)
        Center indices of shape [Ntoken,], Cα
    repr_index: torch.Tensor (long)
        Representative atom indices of shape [Ntoken,], Cβ
    frame_index: torch.Tensor (long)
        Frame indices of shape [Ntoken, 3].
    frame_mask: torch.Tensor (bool)
        Boolean tensor of shape [Ntoken,], indicating whether token's frame is resolved.
    pad_mask: torch.Tensor (bool)
        Mask tensor of shape [Ntoken,], indicating valid tokens.
    apo_center_coords: torch.Tensor (float32)
        Apo state center coordinates of shape [Ntoken, 3].
    apo_repr_coords: torch.Tensor (float32)
        Apo state representative atom center coordinates of shape [Ntoken, 3].
    apo_frame_coords: torch.Tensor (float32)
        Apo state frame of shape [Ntoken, 3, 3].
    apo_center_mask: torch.Tensor (bool)
        Mask tensor of shape [Ntoken,], indicating apo Cα is resolved.
    apo_repr_mask: torch.Tensor (bool)
        Mask tensor of shape [Ntoken,], indicating apo Cβ is resolved.
    apo_frame_mask: torch.Tensor (bool)
        Mask tensor of shape [Ntoken,], indicating apo frame is resolved.

    # For model training
    center_coords: torch.Tensor (float32)
        Center coordinates of shape [Ntoken, 3].
    repr_coords: torch.Tensor (float32)
        Representative atom center coordinates of shape [Ntoken, 3].
    center_mask: torch.Tensor (bool)
        Mask tensor of shape [Ntoken,], indicating center atom of tokens to be resolved.
    repr_mask: torch.Tensor (bool)
        Mask tensor of shape [Ntoken,], indicating repr atom of tokens to be resolved.
    """

    chain_type: torch.Tensor  # [Ntoken,], long
    entity_id: torch.Tensor  # [Ntoken,], long
    asym_id: torch.Tensor  # [Ntoken,], long, same to sequence_id
    apo_uid: torch.Tensor  # [Ntoken,], long
    sym_id: torch.Tensor  # [Ntoken,], long
    res_type: torch.Tensor  # [Ntoken, 32], float32
    is_standard: torch.Tensor  # [Ntoken,], bool
    num_atoms: torch.Tensor  # [Ntoken,], long
    token_index: torch.Tensor  # [Ntoken,], long
    org_token_index: torch.Tensor  # [Ntoken,], long
    residue_index: torch.Tensor  # [Ntoken,], long
    seq_token_index: torch.Tensor  # [Ntoken,], long
    center_index: torch.Tensor  # [Ntoken,], long
    repr_index: torch.Tensor  # [Ntoken,], long
    frame_index: torch.Tensor  # [Ntoken, 3], long
    frame_mask: torch.Tensor  # [Ntoken,], bool
    pad_mask: torch.Tensor  # [Ntoken,], bool

    # Apo state indices.
    apo_center_coords: torch.Tensor  # [Ntoken,], float32
    apo_repr_coords: torch.Tensor  # [Ntoken,], float32
    apo_frame_coords: torch.Tensor  # [Ntoken, 3, 3], float32
    apo_center_mask: torch.Tensor  # [Ntoken,], bool
    apo_repr_mask: torch.Tensor  # [Ntoken,], bool
    apo_frame_mask: torch.Tensor  # [Ntoken,], bool

    # For model training
    center_coords: torch.Tensor  # [Ntoken, 3], float32
    repr_coords: torch.Tensor  # [Ntoken, 3], float32
    center_mask: torch.Tensor  # [Ntoken,], bool
    repr_mask: torch.Tensor  # [Ntoken,], bool

    @property
    def layout_shape(self) -> tuple[int, ...]:
        return self.pad_mask.shape

    @property
    def ndim_unbatched(self) -> int:
        """[ClassVar] The number of dimensions of the layout."""
        return 1

    def __post_init__(self):
        shape = self.layout_shape
        attributes = [
            ("chain_type", torch.long, shape),
            ("entity_id", torch.long, shape),
            ("asym_id", torch.long, shape),
            ("apo_uid", torch.long, shape),
            ("sym_id", torch.long, shape),
            ("res_type", torch.float32, (*shape, 32)),
            ("is_standard", torch.bool, shape),
            ("num_atoms", torch.long, shape),
            ("token_index", torch.long, shape),
            ("org_token_index", torch.long, shape),
            ("seq_token_index", torch.long, shape),
            ("residue_index", torch.long, shape),
            ("repr_index", torch.long, shape),
            ("center_index", torch.long, shape),
            ("frame_index", torch.long, (*shape, 3)),
            ("pad_mask", torch.bool, shape),
            ("frame_mask", torch.bool, shape),
            # Apo features.
            ("apo_center_coords", torch.float32, (*shape, 3)),
            ("apo_repr_coords", torch.float32, (*shape, 3)),
            ("apo_frame_coords", torch.float32, (*shape, 3, 3)),
            ("apo_center_mask", torch.bool, shape),
            ("apo_repr_mask", torch.bool, shape),
            ("apo_frame_mask", torch.bool, shape),
            # For model training
            ("center_coords", torch.float32, (*shape, 3)),
            ("repr_coords", torch.float32, (*shape, 3)),
            ("center_mask", torch.bool, shape),
            ("repr_mask", torch.bool, shape),
        ]
        for name, dtype, shape in attributes:
            check_tensor(getattr(self, name), name=name, dtype=dtype, shape=shape)

    @cached_property
    def is_protein(self) -> torch.Tensor:
        """Boolean tensor of shape [Ntoken,], indicating whether the token is protein."""
        return self.chain_type == C.chain.ChainType.PROTEIN.value

    @cached_property
    def is_dna(self) -> torch.Tensor:
        """Boolean tensor of shape [Ntoken,], indicating whether the token is dna."""
        return self.chain_type == C.chain.ChainType.DNA.value

    @cached_property
    def is_rna(self) -> torch.Tensor:
        """Boolean tensor of shape [Ntoken,], indicating whether the token is rna."""
        return self.chain_type == C.chain.ChainType.RNA.value

    @cached_property
    def is_ligand(self) -> torch.Tensor:
        """Boolean tensor of shape [Ntoken,], indicating whether the token is ligand."""
        return self.chain_type == C.chain.ChainType.LIGAND.value

    def pad(self, *pad_shape: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        self._check_pad_input(pad_shape)

        if pad_shape == self.layout_shape:
            return self

        total_length = pad_shape[0]  # single dimension
        L = len(self)

        pad_values = {
            "is_standard": False,
            "num_atoms": -1,
            "token_index": -1,
            "org_token_index": -1,
            "seq_token_index": -1,
            "res_type": 0,  # 0 = pad
            "chain_type": -1,
            "entity_id": -1,
            "asym_id": -1,
            "apo_uid": -1,
            "sym_id": -1,
            "residue_index": -1,
            "repr_index": -1,
            "center_index": -1,
            "frame_index": -1,
            "frame_mask": False,
            "pad_mask": False,
            # Apo features.
            "apo_center_coords": 0.0,
            "apo_repr_coords": 0.0,
            "apo_frame_coords": 0.0,
            "apo_center_mask": False,
            "apo_repr_mask": False,
            "apo_frame_mask": False,
            # For model training
            "center_coords": 0.0,
            "repr_coords": 0.0,
            "center_mask": False,
            "repr_mask": False,
        }

        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            to_shape = (total_length,) + tensor.shape[1:]
            padded_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:L] = tensor
            fields[name] = padded_tensor
        return self.from_dict(fields)


@dataclasses.dataclass(frozen=True)
class AtomTensor(TensorLayout):
    """Atom-level layout information for molecular structures.

    Shape: [Natom, ...] or [B, Natom, ...]

    Attributes
    ----------
    atom_type: np.ndarray (int)
        Atom types of shape [Natom,], indicating the type of each atom.
    ref_atom_name_chars: torch.Tensor (float32)
        Encoded atom name of shape [Natom, 4, 64].
        One-hot vector
    ref_element: torch.Tensor (float32)
        One-hot encoded atomic numbers of shape [Natom, 128].
    ref_charge: torch.Tensor (float32)
        Formal charges of shape [Natom,].
    ref_pos: torch.Tensor (float32)
        Reference coordinates of shape [Natom, 3].
        Generated from ETKDG or ccd
        (TODO (seonghwan): I think we can replace this to apo_coords)
    ref_space_uid: torch.Tensor (long)
        Numerical encoding of the chain id and residue index associated with
        this reference conformer.
    token_index: torch.Tensor (long)
        Token indices mapping atoms to their parent tokens of shape [Natom,].
    atom_index: torch.Tensor (long)
        Atom indices of shape [Natom,]. Starting from 0 for each chain.
    apo_coords: torch.Tensor (float32)
        Apo (unbound) state coordinates of shape [Natom, 3],
    prior_coords: torch.Tensor (float32)
        Prior coordinates of shape [Natom, Nprior, 3],
    apo_mask: torch.Tensor (bool)
        Boolean mask of shape [Natom,] indicating atoms with apo coordinates.
    pad_mask: torch.Tensor (bool)
        Boolean mask of shape [Natom,] indicating valid (non-padded) atoms.

    # For model training
    label_coords: torch.Tensor (float32)
        Holo (bound) state coordinates of shape [Natom, 3],
        This is used as the ground truth for training, and may be set to 0
        for inference.
    resolved_mask: torch.Tensor (bool)
        Boolean mask of shape [Natom,] indicating atoms to be resolved.
    """

    atom_type: torch.Tensor  # [Natom,], long
    ref_atom_name_chars: torch.Tensor  # [Natom, 4, 64], float32
    ref_element: torch.Tensor  # [Natom, 128], float32
    ref_charge: torch.Tensor  # [Natom,], float32
    ref_pos: torch.Tensor  # [Natom, 3], float32
    ref_mask: torch.Tensor  # [Natom,], bool
    ref_space_uid: torch.Tensor  # [Natom,], long
    token_index: torch.Tensor  # [Natom,], long
    atom_index: torch.Tensor  # [Natom,], long
    apo_coords: torch.Tensor  # [Natom, 3], float32
    prior_coords: torch.Tensor  # [Natom, Nprior, 3], float32
    apo_mask: torch.Tensor  # [Natom,], bool
    pad_mask: torch.Tensor  # [Natom,], bool

    # For model training
    label_coords: torch.Tensor  # [Natom, 3], float32
    resolved_mask: torch.Tensor  # [Natom,], bool

    @property
    def layout_shape(self) -> tuple[int, ...]:
        return self.pad_mask.shape

    @property
    def ndim_unbatched(self) -> int:
        """[ClassVar] The number of dimensions of the layout."""
        return 1

    def __post_init__(self):
        shape = self.layout_shape
        attributes = [
            ("atom_type", torch.long, shape),
            ("ref_atom_name_chars", torch.float32, (*shape, 4, 64)),
            ("ref_element", torch.float32, (*shape, 128)),
            ("ref_charge", torch.float32, shape),
            ("ref_pos", torch.float32, (*shape, 3)),
            ("ref_mask", torch.bool, shape),
            ("ref_space_uid", torch.long, shape),
            ("token_index", torch.long, shape),
            ("atom_index", torch.long, shape),
            ("apo_coords", torch.float32, (*shape, 3)),
            ("prior_coords", torch.float32, (*shape, -1, 3)),
            ("apo_mask", torch.bool, shape),
            ("pad_mask", torch.bool, shape),
            ("label_coords", torch.float32, (*shape, 3)),
            ("resolved_mask", torch.bool, shape),
        ]
        for name, dtype, shape in attributes:
            check_tensor(getattr(self, name), name=name, dtype=dtype, shape=shape)

    def pad(self, *pad_shape: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        self._check_pad_input(pad_shape)

        if pad_shape == self.layout_shape:
            return self

        total_length = pad_shape[0]  # single dimension
        L = len(self)

        pad_values = {
            "atom_type": 0,
            "ref_atom_name_chars": 0.0,
            "ref_element": 0.0,
            "ref_charge": 0.0,
            "ref_pos": 0.0,
            "ref_mask": False,
            "ref_space_uid": -1,
            "token_index": 0,
            "atom_index": 0,
            "prior_coords": 0.0,
            "apo_coords": 0.0,
            "apo_mask": False,
            "pad_mask": False,
            "label_coords": 0.0,
            "resolved_mask": False,
        }

        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            to_shape = (total_length,) + tensor.shape[1:]
            padded_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:L] = tensor
            fields[name] = padded_tensor
        return self.from_dict(fields)


@dataclasses.dataclass(frozen=True)
class BondTensor(TensorLayout):
    """Bond-level layout information for molecular structures.

    Shape: [Nbond, ...] or [B, Nbond, ...]

    Attributes
    ----------
    asym_id: torch.Tensor
        Chain asym indices of the connecting atoms in the bond of shape [Nbond, 2].
    token_index: torch.Tensor
        Token indices of the connecting atoms in the bond of shape [Nbond, 2].
    atom_index: torch.Tensor
        Atom indices of the connecting atoms in the bond of shape [Nbond, 2].
    bond_type: torch.Tensor
        Bond types of shape [Nbond,], indicating the type of each bond.
    pad_mask: torch.Tensor
        Boolean mask of shape [Nbond,], indicating valid (non-padded) bonds.
    is_ligand_ligand: torch.Tensor
        Boolean tensor of shape [Nbond,], indicating whether the bond is between
        two ligand atoms.
    is_polymer_ligand: torch.Tensor
        Boolean tensor of shape [Nbond,], indicating whether the bond is between polymer
        and ligand atoms.
    """

    asym_id: torch.Tensor  # [Nbond, 2], long
    token_index: torch.Tensor  # [Nbond, 2], long
    atom_index: torch.Tensor  # [Nbond, 2], long
    bond_type: torch.Tensor  # [Nbond,], long
    pad_mask: torch.Tensor  # [Nbond,], bool
    is_ligand_ligand: torch.Tensor  # [Nbond,], bool
    is_polymer_ligand: torch.Tensor  # [Nbond,], bool

    @property
    def layout_shape(self) -> tuple[int, ...]:
        return self.pad_mask.shape

    @property
    def ndim_unbatched(self) -> int:
        """[ClassVar] The number of dimensions of the layout."""
        return 1

    def __post_init__(self):
        shape = self.layout_shape
        attributes = [
            ("asym_id", torch.long, (*shape, 2)),
            ("token_index", torch.long, (*shape, 2)),
            ("atom_index", torch.long, (*shape, 2)),
            ("bond_type", torch.long, shape),
            ("pad_mask", torch.bool, shape),
            ("is_ligand_ligand", torch.bool, shape),
            ("is_polymer_ligand", torch.bool, shape),
        ]
        for name, dtype, shape in attributes:
            check_tensor(getattr(self, name), name=name, dtype=dtype, shape=shape)

    def pad(self, *pad_shape: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        self._check_pad_input(pad_shape)

        if pad_shape == self.layout_shape:
            return self

        total_length = pad_shape[0]  # single dimension
        L = len(self)

        # NOTE: (SeoSeongwhan) (0, 0) means self-looping, which doesn't exist in data.
        # Therefore, we can simply remove an element (0, 0) in model forward safely.
        # This trick is inspired by AlphaFold3's implementation:
        # https://github.com/google-deepmind/alphafold3/blob/aed6e82cb10a751eee1a2b9b4ec9acf464812ff8/src/alphafold3/model/network/evoformer.py#L162-L163
        pad_values = {
            "asym_id": 0,
            "token_index": 0,
            "atom_index": 0,
            "bond_type": 0,
            "pad_mask": False,
            "is_ligand_ligand": False,
            "is_polymer_ligand": False,
        }

        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            to_shape = (total_length,) + tensor.shape[1:]
            padded_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:L] = tensor
            fields[name] = padded_tensor
        return self.from_dict(fields)


@dataclasses.dataclass(frozen=True)
class SequenceTensor(TensorLayout):
    """Sequence information for sequence embedding.

    Attributes
    ----------
    asym_id: np.ndarray (int)
        Asymmetric unit IDs of shape [L,], starting from 1 for each chain.
    chain_type: np.ndarray (int)
        Chain types of shape [L,], indicating the type of each chain.
    seq_token_id: np.ndarray (int)
        Sequence tokens of shape [L,] (aatype, base, atom, ...)
    bb_struct_token_id: np.ndarray (int)
        Backbone structure tokens of shape [L,].
    fa_struct_token_id: np.ndarray (int)
        Full-atom structure tokens of shape [L,].
    pos_id: np.ndarray (int)
        Residue indices of shape [L,], used for residue-level operations,
        starting from 0 (BOS).
    mlm_mask: np.ndarray (bool)
        Mask tensor of shape [L,], indicating to mask tokens for sequence
        embedding. (see ESMFold stochastic sampling strategy)

    Cached Properties
    -----------------
    is_protein: np.ndarray (bool)
        Boolean tensor indicating whether the chain is protein.
    is_dna: np.ndarray (bool)
        Boolean tensor indicating whether the chain is dna.
    is_rna: np.ndarray (bool)
        Boolean tensor indicating whether the chain is rna.
    is_ligand: np.ndarray (bool)
        Boolean tensor indicating whether the chain is ligand.
    """

    chain_type: torch.Tensor  # [L,], int
    asym_id: torch.Tensor  # [L,], int
    seq_token_id: torch.Tensor  # [L,], int
    bb_struct_token_id: torch.Tensor  # [L,], int
    fa_struct_token_id: torch.Tensor  # [L,], int
    pos_id: torch.Tensor  # [L,], int
    mlm_mask: torch.Tensor  # [L,], bool
    pad_mask: torch.Tensor  # [L,], bool

    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.seq_token_id.shape  # [L,]

    @property
    def ndim_unbatched(self) -> int:
        """[ClassVar] The number of dimensions of the layout."""
        return 1

    def __post_init__(self):
        shape = self.layout_shape
        attributes = [
            ("chain_type", torch.long, shape),
            ("asym_id", torch.long, shape),
            ("seq_token_id", torch.long, shape),
            ("bb_struct_token_id", torch.long, shape),
            ("fa_struct_token_id", torch.long, shape),
            ("pos_id", torch.long, shape),
            ("mlm_mask", torch.bool, shape),
            ("pad_mask", torch.bool, shape),
        ]
        for name, dtype, shape in attributes:
            check_tensor(getattr(self, name), name=name, dtype=dtype, shape=shape)

    @cached_property
    def is_protein(self) -> torch.Tensor:
        """Boolean tensor indicating whether the chain is protein."""
        return self.chain_type == C.chain.ChainType.PROTEIN.value

    @cached_property
    def is_dna(self) -> torch.Tensor:
        """Boolean tensor indicating whether the chain is dna."""
        return self.chain_type == C.chain.ChainType.DNA.value

    @cached_property
    def is_rna(self) -> torch.Tensor:
        """Boolean tensor indicating whether the chain is rna."""
        return self.chain_type == C.chain.ChainType.RNA.value

    @cached_property
    def is_ligand(self) -> torch.Tensor:
        """Boolean tensor indicating whether the chain is ligand."""
        return self.chain_type == C.chain.ChainType.LIGAND.value

    def pad(self, *pad_shape: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        self._check_pad_input(pad_shape)

        if pad_shape == self.layout_shape:
            return self

        total_length = pad_shape[0]  # single dimension
        L = len(self)

        pad_values = {
            "chain_type": -1,
            "asym_id": -1,
            "seq_token_id": C.sequence.PAD_TOKEN_INDEX,
            "bb_struct_token_id": -1,
            "fa_struct_token_id": -1,
            "pos_id": -1,
            "pad_mask": False,
            "mlm_mask": False,
        }

        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            to_shape = (total_length,) + tensor.shape[1:]
            padded_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:L] = tensor
            fields[name] = padded_tensor
        return self.from_dict(fields)


@dataclasses.dataclass(frozen=True)
class ConstraintTensor(TensorLayout):
    """Constraint-level layout information for molecular structures.

    Shape: [Nconstraint, ...] or [B, Nconstraint, ...]

    Attributes
    ----------
    asym_id: torch.Tensor
        Chain asym id pairs in the constraint of shape [Nconstraint, 2].
    token_index: torch.Tensor
        Token index pairs in the constraint of shape [Nconstraint, 2].
    atom_index: torch.Tensor
        Atom index pairs in the constraint of shape [Nconstraint, 2].
        For polymer, the atom index is the center atom index of the token,
        i.e., Protein: 1(CA), RNA: 11(C1'), DNA: 10(C1'), Ligand: 0.
    lower_bound: torch.Tensor
        Minimum distance constraints of shape [Nconstraint,],
        -1 indicates no minimum distance constraint.
    upper_bound: torch.Tensor
        Maximum distance constraints of shape [Nconstraint,],
        -1 indicates no maximum distance constraint.
    pad_mask: torch.Tensor
        Boolean mask of shape [Nconstraint,], indicating valid (non-padded) constraints.
    """

    asym_id: torch.Tensor  # [Nconstraint, 2], long
    token_index: torch.Tensor  # [Nconstraint, 2], long
    atom_index: torch.Tensor  # [Nconstraint, 2], long
    lower_bound: torch.Tensor  # [Nconstraint,], float32
    upper_bound: torch.Tensor  # [Nconstraint,], float32
    pad_mask: torch.Tensor  # [Nconstraint,], bool

    @property
    def layout_shape(self) -> tuple[int, ...]:
        return self.pad_mask.shape

    @property
    def ndim_unbatched(self) -> int:
        """[ClassVar] The number of dimensions of the layout."""
        return 1

    def __post_init__(self):
        shape = self.layout_shape
        attributes = [
            ("asym_id", torch.long, (*shape, 2)),
            ("token_index", torch.long, (*shape, 2)),
            ("atom_index", torch.long, (*shape, 2)),
            ("lower_bound", torch.float32, shape),
            ("upper_bound", torch.float32, shape),
            ("pad_mask", torch.bool, shape),
        ]
        for name, dtype, shape in attributes:
            check_tensor(getattr(self, name), name=name, dtype=dtype, shape=shape)

    def pad(self, *pad_shape: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        self._check_pad_input(pad_shape)

        if pad_shape == self.layout_shape:
            return self

        total_length = pad_shape[0]  # single dimension
        L = len(self)

        # NOTE: (SeoSeongwhan) (0, 0) means self-looping, which doesn't exist in data.
        # Therefore, we can simply remove an element (0, 0) in model forward safely.
        pad_values = {
            "asym_id": 0,
            "token_index": 0,
            "atom_index": 0,
            "lower_bound": -1.0,
            "upper_bound": -1.0,
            "pad_mask": False,
        }

        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            to_shape = (total_length,) + tensor.shape[1:]
            padded_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:L] = tensor
            fields[name] = padded_tensor
        return self.from_dict(fields)


@dataclasses.dataclass(frozen=True)
class FoldingInput:
    """Input of co-folding"""

    chain: ChainTensor
    token: TokenTensor
    atom: AtomTensor
    bond: BondTensor
    sequence: SequenceTensor
    constraint: ConstraintTensor
    crop_mode: torch.Tensor | None = None

    def __post_init__(self):
        # check all layouts are on the same device and have same batch status
        device = self.chain.device
        is_batched = self.chain.is_batched
        batch_size = self.chain.batch_size if is_batched else None

        for name in ["token", "atom", "bond", "sequence", "constraint"]:
            layout = getattr(self, name)
            assert layout.device == device, f"{name} layout must be on the same device."
            assert layout.is_batched == is_batched, (
                f"{name} layout must be batched or non-batched as same as chain layout."
            )
            if is_batched:
                assert layout.batch_size == batch_size, (
                    f"{name} layout must have the same batch size as chain layout."
                )
        if self.crop_mode is None:
            shape = (batch_size,) if is_batched else ()
            crop_mode = torch.zeros(shape, dtype=torch.long, device=device)
            object.__setattr__(self, "crop_mode", crop_mode)
        else:
            assert self.crop_mode.device == device, (
                "crop_mode must be on the same device as the layouts."
            )
            assert self.crop_mode.dtype == torch.long, "crop_mode must be long."
            expected_shape = (batch_size,) if is_batched else ()
            assert self.crop_mode.shape == expected_shape, (
                f"crop_mode must have shape {expected_shape}, got "
                f"{tuple(self.crop_mode.shape)}."
            )

    def to(self, device: str | torch.device) -> Self:
        return self.__class__(
            chain=self.chain.to(device),
            token=self.token.to(device),
            atom=self.atom.to(device),
            bond=self.bond.to(device),
            sequence=self.sequence.to(device),
            constraint=self.constraint.to(device),
            crop_mode=self.crop_mode.to(device),
        )

    @property
    def device(self) -> torch.device:
        return self.token.device

    @property
    def is_batched(self) -> bool:
        return self.token.is_batched

    @property
    def batch_size(self) -> int:
        assert self.is_batched, "Input is not batched."
        return self.token.batch_size

    @property
    def num_chains(self) -> int:
        return len(self.chain)

    @property
    def num_tokens(self) -> int:
        return len(self.token)

    @property
    def num_atoms(self) -> int:
        return len(self.atom)

    @property
    def num_bonds(self) -> int:
        return len(self.bond)

    @property
    def num_sequence_tokens(self) -> int:
        return len(self.sequence)

    @property
    def num_constraints(self) -> int:
        return len(self.constraint)

    def add_batch_dim(self, deepcopy: bool = False) -> Self:
        """Add a batch dimension"""
        if self.is_batched:
            raise ValueError("Input is already batched.")
        return self.__class__(
            chain=self.chain.add_batch_dim(deepcopy),
            token=self.token.add_batch_dim(deepcopy),
            atom=self.atom.add_batch_dim(deepcopy),
            bond=self.bond.add_batch_dim(deepcopy),
            sequence=self.sequence.add_batch_dim(deepcopy),
            constraint=self.constraint.add_batch_dim(deepcopy),
            crop_mode=(
                self.crop_mode.unsqueeze(0).clone()
                if deepcopy
                else self.crop_mode.unsqueeze(0)
            ),
        )

    @classmethod
    def from_list(cls, data_list: list[Self], pad_to_max: bool = False) -> Self:
        """Create a Batched Input from a list of data.

        Parameters
        ----------
        data_list: list[FoldingInput]
            A list of FoldingInput instances to be batched.
        pad_to_max: bool
            Whether to pad all layouts to the maximum length in the batch.
            Default is False (If False, all layouts should have the same length).

        Returns
        -------
        batch: FoldingInput
            A batched FoldingInput instance.
        """
        # Check all data are non-batched
        for data in data_list:
            assert not data.is_batched, "All inputs in data_list must be non-batched."

        # Check all data are on the same device
        ref_device = data_list[0].device
        for data in data_list:
            assert data.device == ref_device, (
                "All inputs in data_list must be on the same device."
            )

        if pad_to_max:
            # Pad to the maximum length in the batch
            # Determine max lengths for each layout
            max_chains = max(len(data.chain) for data in data_list)
            max_tokens = max(len(data.token) for data in data_list)
            max_atoms = max(len(data.atom) for data in data_list)
            max_bonds = max(len(data.bond) for data in data_list)
            max_sequence_tokens = max(len(data.sequence) for data in data_list)
            max_constraints = max(len(data.constraint) for data in data_list)
            # Pad each layout to the maximum length
            data_list = [
                data.pad(
                    max_tokens,
                    max_chains,
                    max_atoms,
                    max_bonds,
                    max_sequence_tokens,
                    max_constraints,
                )
                for data in data_list
            ]
        else:
            # Check all layouts have the same length
            ref_num_chains = len(data_list[0].chain)
            ref_num_tokens = len(data_list[0].token)
            ref_num_atoms = len(data_list[0].atom)
            ref_num_bonds = len(data_list[0].bond)
            ref_num_sequence = len(data_list[0].sequence)
            ref_num_constraints = len(data_list[0].constraint)
            for data in data_list:
                assert len(data.chain) == ref_num_chains, (
                    "All chain layouts must have the same length."
                )
                assert len(data.token) == ref_num_tokens, (
                    "All token layouts must have the same length."
                )
                assert len(data.atom) == ref_num_atoms, (
                    "All atom layouts must have the same length."
                )
                assert len(data.bond) == ref_num_bonds, (
                    "All bond layouts must have the same length."
                )
                assert len(data.sequence) == ref_num_sequence, (
                    "All sequence layouts must have the same length."
                )
                assert len(data.constraint) == ref_num_constraints, (
                    "All constraint layouts must have the same length."
                )

        batched_chain = ChainTensor.from_list([data.chain for data in data_list])
        batched_token = TokenTensor.from_list([data.token for data in data_list])
        batched_atom = AtomTensor.from_list([data.atom for data in data_list])
        batched_bond = BondTensor.from_list([data.bond for data in data_list])
        batched_sequence = SequenceTensor.from_list([data.sequence for data in data_list])
        batched_constraint = ConstraintTensor.from_list(
            [data.constraint for data in data_list]
        )
        batched_crop_mode = torch.stack([data.crop_mode for data in data_list], dim=0)

        return cls(
            chain=batched_chain,
            token=batched_token,
            atom=batched_atom,
            bond=batched_bond,
            sequence=batched_sequence,
            constraint=batched_constraint,
            crop_mode=batched_crop_mode,
        )

    def to_list(self, deepcopy: bool = False) -> list[Self]:
        chain_list = self.chain.to_list(deepcopy)
        token_list = self.token.to_list(deepcopy)
        atom_list = self.atom.to_list(deepcopy)
        bond_list = self.bond.to_list(deepcopy)
        sequence_list = self.sequence.to_list(deepcopy)
        constraint_list = self.constraint.to_list(deepcopy)

        data_list: list[Self] = []
        batch_size = self.batch_size
        for b in range(batch_size):
            data_list.append(
                self.__class__(
                    chain=chain_list[b],
                    token=token_list[b],
                    atom=atom_list[b],
                    bond=bond_list[b],
                    sequence=sequence_list[b],
                    constraint=constraint_list[b],
                    crop_mode=(
                        self.crop_mode[b].clone() if deepcopy else self.crop_mode[b]
                    ),
                )
            )
        return data_list

    def __repr__(self) -> str:
        """FoldingInput summary representation."""
        device = self.device

        # Summary statistics
        num_chains = self.num_chains
        num_tokens = self.num_tokens
        num_atoms = self.num_atoms
        num_bonds = len(self.bond)

        if self.is_batched:
            return (
                f"FoldingInput(\n"
                f"  batch_size: {self.batch_size}\n"
                f"  num_chains: {num_chains}\n"
                f"  num_tokens: {num_tokens}\n"
                f"  num_atoms: {num_atoms}\n"
                f"  num_bonds: {num_bonds}\n"
                f"  num_constraints: {len(self.constraint)}\n"
                f"  device: {device}\n"
                f")"
            )
        else:
            return (
                f"FoldingInput(\n"
                f"  num_chains: {num_chains}\n"
                f"  num_tokens: {num_tokens}\n"
                f"  num_atoms: {num_atoms}\n"
                f"  num_bonds: {num_bonds}\n"
                f"  num_constraints: {len(self.constraint)}\n"
                f"  device: {device}\n"
                f")"
            )

    def copy_with(self, **kwargs) -> Self:
        """Create a copy of the FoldingInput with modified attributes.

        Parameters
        ----------
        kwargs: dict
            Attributes to be modified in the new instance.

        Returns
        -------
        new_instance: FoldingInput
            A new instance of FoldingInput with modified attributes.
        """
        return dataclasses.replace(self, **kwargs)

    # === Padding functions for preparing model inputs === #
    def pad_to_multiple_of(self, multiple: int = 32) -> Self:
        """Pad all layouts to the multiple of 32 tokens for LocalAtomAttention and
        model efficiency."""
        assert multiple > 0, f"multiple must be a positive integer, but got {multiple}."
        # 4 * 24 is divided by 32, which is the minimum unit of LocalAttention
        assert multiple % 4 == 0, (
            f"multiple must be a multiple of 4 for LocalAttention, but got {multiple}."
        )
        max_tokens = ((self.num_tokens + multiple - 1) // multiple) * multiple
        return self.pad_to_max_token(max_tokens)

    def pad_to_max_token(self, max_tokens: int) -> Self:
        """Pad all layouts based on the given max_tokens."""
        # Determine max_chains, max_atoms, max_bonds based on max_tokens
        max_chains = max_tokens // 4  # min 4 tokens per chain
        max_atoms = max_tokens * 24  # max 24 atoms per token
        max_bonds = max_tokens * 10  # max 10 bonds per token
        max_constraint = max_tokens  # max 1 constraints per token

        return self.pad(
            max_chains=max_chains,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            max_bonds=max_bonds,
            max_constraints=max_constraint,
        )

    def pad(
        self,
        max_tokens: int | None = None,
        max_chains: int | None = None,
        max_atoms: int | None = None,
        max_bonds: int | None = None,
        max_sequence_tokens: int | None = None,
        max_constraints: int | None = None,
    ) -> Self:
        """Pad all layouts to the specified maximum sizes."""
        max_tokens = max_tokens if max_tokens is not None else len(self.token)
        max_chains = max_chains if max_chains is not None else len(self.chain)
        max_atoms = max_atoms if max_atoms is not None else len(self.atom)
        max_bonds = max_bonds if max_bonds is not None else len(self.bond)
        max_sequence_tokens = (
            max_sequence_tokens if max_sequence_tokens is not None else len(self.sequence)
        )
        max_constraints = (
            max_constraints if max_constraints is not None else len(self.constraint)
        )

        return self.__class__(
            chain=self.chain.pad(max_chains),
            token=self.token.pad(max_tokens),
            atom=self.atom.pad(max_atoms),
            bond=self.bond.pad(max_bonds),
            sequence=self.sequence.pad(max_sequence_tokens),
            constraint=self.constraint.pad(max_constraints),
            crop_mode=self.crop_mode,
        )
