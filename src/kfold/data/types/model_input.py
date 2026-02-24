import dataclasses
from functools import cached_property
from typing import Self

import torch

import kfold.constants as C
from kfold.data.layout import TensorLayout
from kfold.utils.misc import check_tensor

__all__ = [
    "FoldingInput",
]


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

        check_tensor(self.chain_type, name="chain_type", dtype=torch.long, shape=shape)
        check_tensor(self.entity_id, name="entity_id", dtype=torch.long, shape=shape)
        check_tensor(self.asym_id, name="asym_id", dtype=torch.long, shape=shape)
        check_tensor(self.sym_id, name="sym_id", dtype=torch.long, shape=shape)
        check_tensor(self.num_tokens, name="num_tokens", dtype=torch.long, shape=shape)
        check_tensor(
            self.num_residues, name="num_residues", dtype=torch.long, shape=shape
        )
        check_tensor(self.num_atoms, name="num_atoms", dtype=torch.long, shape=shape)

    def pad(self, *pad_shape: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        self._check_pad_input(pad_shape)

        if pad_shape == self.layout_shape:
            return self

        total_length = pad_shape[0]  # single dimension
        L = len(self)

        # value: PAD_IDX means padding
        pad_values = {
            "chain_type": -1,
            "entity_id": -1,
            "asym_id": -1,
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
    token_index: torch.Tensor (long)
        Token indices of shape [Ntoken,], mapping each token to its position.
    org_token_index: torch.Tensor (long)
        Original token indices of shape [Ntoken,], before cropping.
    res_type: torch.Tensor (float32)
        Sequence tokens of shape [Ntoken, 32] (aatype, atom, ...)
        One-hot vector
    chain_type: torch.Tensor (long)
        Chain types of shape [Ntoken,], indicating the type of each token.
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
    disto_index: torch.Tensor (long)
        Representative atom indices of shape [Ntoken,], Cβ
    frames_index: torch.Tensor (long)
        Frame indices of shape [Ntoken, 3].
    frames_mask: torch.Tensor (bool)
        Boolean tensor of shape [Ntoken,], indicating whether token's frame is resolved.
    pad_mask: torch.Tensor (bool)
        Mask tensor of shape [Ntoken,], indicating valid tokens.
    pocket_contact_type: torch.Tensor (long)
        Pocket contact types of shape [Ntoken,], indicating pocket contact information.
    interaction_type: torch.Tensor (bool)
        Multi-hot interaction types of shape [Ntoken, NUM_INTERACTION_TYPES].

    # For model training
    center_coords: torch.Tensor (float32)
        Center coordinates of shape [Ntoken, 3].
    disto_coords: torch.Tensor (float32)
        Representative atom center coordinates of shape [Ntoken, 3].
    center_mask: torch.Tensor (bool)
        Mask tensor of shape [Ntoken,], indicating center atom of tokens to be resolved.
    disto_mask: torch.Tensor (bool)
        Mask tensor of shape [Ntoken,], indicating disto atom of tokens to be resolved.
    """

    token_index: torch.Tensor  # [Ntoken,], long
    org_token_index: torch.Tensor  # [Ntoken,], long
    residue_index: torch.Tensor  # [Ntoken,], long
    seq_token_index: torch.Tensor  # [Ntoken,], long
    res_type: torch.Tensor  # [Ntoken, 32], float32
    chain_type: torch.Tensor  # [Ntoken,], long
    entity_id: torch.Tensor  # [Ntoken,], long
    asym_id: torch.Tensor  # [Ntoken,], long, same to sequence_id
    sym_id: torch.Tensor  # [Ntoken,], long
    center_index: torch.Tensor  # [Ntoken,], long
    disto_index: torch.Tensor  # [Ntoken,], long
    frames_index: torch.Tensor  # [Ntoken, 3], long
    frames_mask: torch.Tensor  # [Ntoken,], bool
    pad_mask: torch.Tensor  # [Ntoken,], bool
    pocket_contact_type: torch.Tensor  # [Ntoken,], bool
    interaction_type: torch.Tensor  # [Ntoken, NUM_INTERACTION_TYPES], bool

    # For model training
    center_coords: torch.Tensor  # [Ntoken, 3], long
    disto_coords: torch.Tensor  # [Ntoken, 3], long
    center_mask: torch.Tensor  # [Ntoken,], bool
    disto_mask: torch.Tensor  # [Ntoken,], bool

    @property
    def layout_shape(self) -> tuple[int, ...]:
        return self.pad_mask.shape

    @property
    def ndim_unbatched(self) -> int:
        """[ClassVar] The number of dimensions of the layout."""
        return 1

    def __post_init__(self):
        shape = self.layout_shape
        check_tensor(self.token_index, name="token_index", dtype=torch.long, shape=shape)
        check_tensor(
            self.org_token_index, name="org_token_index", dtype=torch.long, shape=shape
        )
        check_tensor(
            self.seq_token_index, name="seq_token_index", dtype=torch.long, shape=shape
        )
        check_tensor(
            self.res_type, name="res_type", dtype=torch.float32, shape=(*shape, 32)
        )
        check_tensor(self.chain_type, name="chain_type", dtype=torch.long, shape=shape)
        check_tensor(self.entity_id, name="entity_id", dtype=torch.long, shape=shape)
        check_tensor(self.asym_id, name="asym_id", dtype=torch.long, shape=shape)
        check_tensor(self.sym_id, name="sym_id", dtype=torch.long, shape=shape)
        check_tensor(
            self.residue_index, name="residue_index", dtype=torch.long, shape=shape
        )
        check_tensor(self.disto_index, name="disto_index", dtype=torch.long, shape=shape)
        check_tensor(
            self.center_index, name="center_index", dtype=torch.long, shape=shape
        )
        check_tensor(
            self.frames_index, name="frames_index", dtype=torch.long, shape=(*shape, 3)
        )
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=shape)
        check_tensor(self.frames_mask, name="frames_mask", dtype=torch.bool, shape=shape)
        check_tensor(
            self.pocket_contact_type,
            name="pocket_contact_type",
            dtype=torch.long,
            shape=shape,
        )
        check_tensor(
            self.interaction_type,
            name="interaction_type",
            dtype=torch.bool,
            shape=(*shape, C.NUM_INTERACTION_TYPES),
        )

        # For model training
        check_tensor(
            self.center_coords,
            name="center_coords",
            dtype=torch.float32,
            shape=(*shape, 3),
        )
        check_tensor(
            self.disto_coords, name="disto_coords", dtype=torch.float32, shape=(*shape, 3)
        )
        check_tensor(self.center_mask, name="center_mask", dtype=torch.bool, shape=shape)
        check_tensor(self.disto_mask, name="disto_mask", dtype=torch.bool, shape=shape)

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

        # value: PAD_IDX means padding
        pad_values = {
            "token_index": -1,
            "org_token_index": -1,
            "seq_token_index": -1,
            "res_type": 0,  # 0 = pad
            "chain_type": -1,
            "entity_id": -1,
            "asym_id": -1,
            "sym_id": -1,
            "residue_index": -1,
            "disto_index": -1,
            "center_index": -1,
            "frames_index": -1,
            "frames_mask": False,
            "pad_mask": False,
            "pocket_contact_type": 0,
            "interaction_type": False,
            # For model training
            "center_coords": 0.0,
            "disto_coords": 0.0,
            "center_mask": False,
            "disto_mask": False,
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

    ref_atom_name_chars: torch.Tensor  # [Natom, 4, 64], float32
    ref_element: torch.Tensor  # [Natom, 128], float32
    ref_charge: torch.Tensor  # [Natom,], float32
    ref_pos: torch.Tensor  # [Natom, 3], float32
    ref_mask: torch.Tensor  # [Natom,], bool
    ref_space_uid: torch.Tensor  # [Natom,], long
    token_index: torch.Tensor  # [Natom,], long
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
        check_tensor(
            self.ref_atom_name_chars,
            name="ref_atom_name_chars",
            dtype=torch.float32,
            shape=(*shape, 4, 64),
        )
        check_tensor(
            self.ref_element, name="ref_element", dtype=torch.float32, shape=(*shape, 128)
        )
        check_tensor(self.ref_charge, name="ref_charge", dtype=torch.float32, shape=shape)
        check_tensor(self.ref_pos, name="ref_pos", dtype=torch.float32, shape=(*shape, 3))
        check_tensor(self.ref_mask, name="ref_mask", dtype=torch.bool, shape=shape)
        check_tensor(
            self.ref_space_uid, name="ref_space_uid", dtype=torch.long, shape=shape
        )
        check_tensor(self.token_index, name="token_index", dtype=torch.long, shape=shape)
        check_tensor(
            self.apo_coords, name="apo_coords", dtype=torch.float32, shape=(*shape, 3)
        )
        check_tensor(
            self.prior_coords,
            name="prior_coords",
            dtype=torch.float32,
            shape=(*shape, -1, 3),
        )
        check_tensor(self.apo_mask, name="apo_mask", dtype=torch.bool, shape=shape)
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=shape)
        check_tensor(
            self.label_coords,
            name="label_coords",
            dtype=torch.float32,
            shape=(*shape, 3),
        )
        check_tensor(
            self.resolved_mask, name="resolved_mask", dtype=torch.bool, shape=shape
        )

    def pad(self, *pad_shape: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        self._check_pad_input(pad_shape)

        if pad_shape == self.layout_shape:
            return self

        total_length = pad_shape[0]  # single dimension
        L = len(self)

        # value: PAD_IDX means padding
        pad_values = {
            "ref_atom_name_chars": 0.0,  # max value for atom_name encoding
            "ref_element": 0.0,  # max value for element encoding
            "ref_charge": 0.0,
            "ref_pos": 0.0,
            "ref_mask": False,
            "ref_space_uid": -1,
            "token_index": 0,
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
        check_tensor(self.asym_id, name="asym_id", dtype=torch.long, shape=(*shape, 2))
        check_tensor(
            self.token_index, name="token_index", dtype=torch.long, shape=(*shape, 2)
        )
        check_tensor(
            self.atom_index, name="atom_index", dtype=torch.long, shape=(*shape, 2)
        )
        check_tensor(self.bond_type, name="bond_type", dtype=torch.long, shape=shape)
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=shape)
        check_tensor(
            self.is_ligand_ligand,
            name="is_ligand_ligand",
            dtype=torch.bool,
            shape=shape,
        )
        check_tensor(
            self.is_polymer_ligand,
            name="is_polymer_ligand",
            dtype=torch.bool,
            shape=shape,
        )

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
    input_id: np.ndarray (int)
        Sequence tokens of shape [L,] (aatype, base, atom, ...)
        NOTE: this may differ from the res_type in TokenArray,
        since vocab is different for sequence embedding and co-folding.
    residue_index: np.ndarray (int)
        Residue indices of shape [L,], used for residue-level operations,
        starting from 0 (BOS).
    entity_id: np.ndarray (int)
        Entity IDs of shape [L,], starting from 1.
    chain_type: np.ndarray (int)
        Chain types of shape [L,], indicating the type of each chain.

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

    input_id: torch.Tensor  # [L,], int
    residue_index: torch.Tensor  # [L,], int
    chain_type: torch.Tensor  # [L,], int
    entity_id: torch.Tensor  # [L,], int
    pad_mask: torch.Tensor  # [L,], bool

    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.input_id.shape  # [L,]

    def __post_init__(self):
        shape = self.layout_shape
        check_tensor(self.input_id, name="res_type", dtype=torch.long, shape=shape)
        check_tensor(
            self.residue_index, name="residue_index", dtype=torch.long, shape=shape
        )
        check_tensor(self.chain_type, name="chain_type", dtype=torch.long, shape=shape)
        check_tensor(self.entity_id, name="entity_id", dtype=torch.long, shape=shape)
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=shape)

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
class FoldingInput:
    """Input of co-folding"""

    chain: ChainTensor
    token: TokenTensor
    atom: AtomTensor
    bond: BondTensor
    sequence: SequenceTensor

    def __post_init__(self):
        # check all layouts are on the same device
        device = self.chain.device
        assert self.token.device == device, "token layout must be on the same device."
        assert self.atom.device == device, "atom layout must be on the same device."
        assert self.bond.device == device, "bond layout must be on the same device."
        assert self.sequence.device == device, (
            "sequence layout must be on the same device."
        )

        # check all layouts are non-batched or batched
        is_batched = self.chain.is_batched
        assert self.token.is_batched == is_batched, (
            "token layout must be batched or non-batched as same as chain layout."
        )
        assert self.atom.is_batched == is_batched, (
            "atom layout must be batched or non-batched as same as chain layout."
        )
        assert self.bond.is_batched == is_batched, (
            "bond layout must be batched or non-batched as same as chain layout."
        )
        assert self.sequence.is_batched == is_batched, (
            "sequence layout must be batched or non-batched as same as chain layout."
        )

        # check the batch size if batched
        if is_batched:
            batch_size = self.chain.batch_size
            assert self.token.batch_size == batch_size, (
                "token layout must have the same batch size as chain layout."
            )
            assert self.atom.batch_size == batch_size, (
                "atom layout must have the same batch size as chain layout."
            )
            assert self.bond.batch_size == batch_size, (
                "bond layout must have the same batch size as chain layout."
            )
            assert self.sequence.batch_size == batch_size, (
                "sequence layout must have the same batch size as chain layout."
            )

    def to(self, device: str | torch.device) -> Self:
        return self.__class__(
            chain=self.chain.to(device),
            token=self.token.to(device),
            atom=self.atom.to(device),
            bond=self.bond.to(device),
            sequence=self.sequence.to(device),
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
            # Pad each layout to the maximum length
            data_list = [
                data.pad(max_tokens, max_chains, max_atoms, max_bonds)
                for data in data_list
            ]
        else:
            # Check all layouts have the same length
            ref_num_chains = len(data_list[0].chain)
            ref_num_tokens = len(data_list[0].token)
            ref_num_atoms = len(data_list[0].atom)
            ref_num_bonds = len(data_list[0].bond)
            ref_num_sequence = len(data_list[0].sequence)
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

        batched_chain = ChainTensor.from_list([data.chain for data in data_list])
        batched_token = TokenTensor.from_list([data.token for data in data_list])
        batched_atom = AtomTensor.from_list([data.atom for data in data_list])
        batched_bond = BondTensor.from_list([data.bond for data in data_list])
        batched_sequence = SequenceTensor.from_list([data.sequence for data in data_list])

        return cls(
            chain=batched_chain,
            token=batched_token,
            atom=batched_atom,
            bond=batched_bond,
            sequence=batched_sequence,
        )

    def to_list(self, deepcopy: bool = False) -> list[Self]:
        chain_list = self.chain.to_list(deepcopy)
        token_list = self.token.to_list(deepcopy)
        atom_list = self.atom.to_list(deepcopy)
        bond_list = self.bond.to_list(deepcopy)
        sequence_list = self.sequence.to_list(deepcopy)

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
                )
            )
        return data_list

    def get_atom_to_token(self) -> torch.Tensor:
        """Return [Natom, Ntoken] or [B, Natom, Ntoken] dense mapping matrix
        from atoms to tokens.
        """
        return self.atom_to_token

    @cached_property
    def atom_to_token(self) -> torch.Tensor:
        """Return [Natom, Ntoken] or [B, Natom, Ntoken] dense mapping matrix
        from atoms to tokens.
        """

        if not self.is_batched:
            Natom = int(self.atom.pad_mask.sum().item())
            atom_indices = torch.arange(Natom, device=self.device)
            token_indices = self.atom.token_index[:Natom]
            mapping = torch.zeros(
                (self.num_atoms, self.num_tokens),
                dtype=torch.float32,
                device=self.device,
            )
            mapping[atom_indices, token_indices] = 1.0
        else:
            batch_size = self.chain.batch_size
            num_atoms = self.atom.pad_mask.sum(dim=1).tolist()
            mapping = torch.zeros(
                (batch_size, self.num_atoms, self.num_tokens),
                dtype=torch.float32,
                device=self.device,
            )
            for b in range(batch_size):
                Natom = num_atoms[b]
                atom_indices = torch.arange(Natom, device=self.device)
                token_indices = self.atom.token_index[b, :Natom]
                mapping[b, atom_indices, token_indices] = 1.0
        return mapping

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

        return self.pad(
            max_chains=max_chains,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            max_bonds=max_bonds,
        )

    def pad(
        self,
        max_tokens: int | None = None,
        max_chains: int | None = None,
        max_atoms: int | None = None,
        max_bonds: int | None = None,
        max_sequence: int | None = None,
    ) -> Self:
        """Pad all layouts to the specified maximum sizes."""
        max_tokens = max_tokens if max_tokens is not None else len(self.token)
        max_chains = max_chains if max_chains is not None else len(self.chain)
        max_atoms = max_atoms if max_atoms is not None else len(self.atom)
        max_bonds = max_bonds if max_bonds is not None else len(self.bond)
        max_sequence = max_sequence if max_sequence is not None else len(self.sequence)

        return self.__class__(
            chain=self.chain.pad(max_chains),
            token=self.token.pad(max_tokens),
            atom=self.atom.pad(max_atoms),
            bond=self.bond.pad(max_bonds),
            sequence=self.sequence.pad(max_sequence),
        )
