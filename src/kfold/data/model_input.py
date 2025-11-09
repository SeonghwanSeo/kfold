import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Self

import torch

import kfold.constants as C
from kfold.utils.misc import check_tensor

# NOTE (seoseonghwan): I do not consider batching since
# most cofolding model use batch_size of 1 with multiple
# diffusion time steps.

# NOTE (seoseonghwan): currently, I select PAD_IDX as 65535
# since this value is appropriate to monitor the tensor values
# during debugging (not too small or too large).
# I think this value can be arbitrary value since we already
# use pad_mask to indicate the padded positions. Therefore,
# we don't have to change this value even if the number of
# chains/tokens/atoms exceed this value.
PAD_IDX = 2**16 - 1  # 65535

__all__ = [
    "ChainLayout",
    "TokenLayout",
    "AtomLayout",
    "BondLayout",
    "FoldingInput",
]

# common type alias
torch_float = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
torch_int = (torch.int32, torch.int64)


# === Base class for dataclass with tensor === #
class TensorObj:
    @cached_property
    def device(self) -> torch.device:
        for tensor in self.to_dict().values():
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        else:
            raise ValueError("No tensor found in the dataclass.")

    def __getitem__(self, idx: int | slice | torch.Tensor) -> Self:
        fields = {
            name: self.slicing(name, tensor, idx)
            for name, tensor in self.to_dict().items()
        }
        return self.from_dict(fields)

    def slicing(self, name: str, value: Any, idx: int | slice | torch.Tensor) -> Any:
        """Slice a field value."""
        assert isinstance(value, torch.Tensor), (
            f"Field '{name}' must be a torch.Tensor to support slicing,"
            f" but got {type(value)}."
            f" Please override slicing() method for custom behavior."
        )
        return value[idx]

    def to(self, device: str | torch.device) -> Self:
        fields = {
            name: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in self.to_dict().items()
        }
        return self.from_dict(fields)

    def keys(self) -> list[str]:
        """Get the field names of the dataclass."""
        # Ensure that self is a dataclass
        assert hasattr(self, "__dataclass_fields__"), (
            f"{self.__class__.__name__} must be a dataclass to use TensorObj.keys()"
        )
        return list(self.__dataclass_fields__.keys())  # type: ignore

    def to_dict(self) -> dict[str, torch.Tensor | Any]:
        """Convert the dataclass fields to a dictionary.
        Note: dataclasses.asdict() is not used to avoid deep copying.
        """
        field_names = self.keys()
        field_dict = {name: getattr(self, name) for name in field_names}
        return field_dict

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor | Any]) -> Self:
        return cls(**data)

    def clone(self, deepcopy: bool = True) -> Self:
        """Create a copy of the object."""
        if deepcopy:
            return copy.deepcopy(self)
        else:
            return self.from_dict(self.to_dict())

    def copy_with(self, deepcopy: bool = True, **kwargs) -> Self:
        """Create a copy of the object with optional field updates."""
        data = self.to_dict()
        assert kwargs.keys() <= data.keys(), (
            f"Invalid field names: {kwargs.keys() - data.keys()}"
        )
        if deepcopy:
            for k in kwargs:
                data.pop(k)
            data = copy.deepcopy(data)
        data.update(kwargs)
        return self.from_dict(data)

    # === Save / Load methods === #
    def get_state(self) -> dict[str, torch.Tensor | Any]:
        """Get the state dictionary of the object.
        We may want to convert datatypes here to reduce the size on disk.
        """
        # convert datatypes if necessary
        return self.to_dict()

    @classmethod
    def load_state(cls, state: dict[str, torch.Tensor | Any]) -> Self:
        """Set the state of the object from the state dictionary.
        Restore datatypes if necessary.
        """
        # convert datatypes if necessary
        return cls.from_dict(state)


class Layout(TensorObj, ABC):
    """Base class for layout information.
    Layout shape: [L, ...] or [B, L, ...]

    NOTE: padding is only supported for non-batched layout.
    """

    @property
    @abstractmethod
    def _layout_shape(self) -> tuple[int] | tuple[int, int]:
        """Get the shape of the layout: [L,] or [B, L]"""

    @property
    def is_batched(self) -> bool:
        """Whether the layout is batched."""
        return len(self._layout_shape) == 2

    def __len__(self) -> int:
        """Get the length of the layout."""
        return self.length

    @property
    def batch_size(self) -> int:
        """Batch size of layout."""
        assert self.is_batched, "Layout is not batched."
        return self._layout_shape[0]

    @property
    def length(self) -> int:
        """Get the length of the layout."""
        return self._layout_shape[-1]

    def __repr__(self) -> str:
        """Enhanced repr with layout-specific info."""
        class_name = self.__class__.__name__

        # Layout meta info
        length = len(self)
        device = self.device

        if self.is_batched:
            shape_desc = f"batch_size={self.batch_size}, length={length}, device={device}"
        else:
            shape_desc = f"length={length}, device={device}"

        # Fields
        fields = self.to_dict()
        field_strs = []
        for name, value in fields.items():
            if isinstance(value, torch.Tensor):
                dtype_str = str(value.dtype).replace("torch.", "")
                shape_str = "x".join(map(str, value.shape))
                field_strs.append(f"  {name}: [{dtype_str}, {shape_str}]")

        fields_repr = "\n".join(field_strs)
        return f"{class_name}({shape_desc})\n{fields_repr}"

    # === Methods for unbatched layout === #
    def __getitem__(self, idx: int | slice | torch.Tensor) -> Self:
        """Get a subset of the layout."""
        # TODO: can we support this for batched layout?
        assert not self.is_batched, "Batched layout is not supported."
        # Ensure that idx is not integer
        if isinstance(idx, int):
            raise ValueError(
                f"Integer indexing is not supported for {self.__class__.__name__}. "
                f"Use slicing instead: obj[{idx}:{idx + 1}]"
            )
        return super().__getitem__(idx)

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Batched layout is not supported."
        assert self.length <= total_length, (
            f"Cannot pad to a smaller length: {total_length} < {self.length}"
        )
        raise NotImplementedError

    # === Methods for batched layout === #

    @classmethod
    def from_list(cls, data_list: list[Self]) -> Self:
        """Create a Batched Layout from a list of data."""
        assert len(data_list) > 0, "data_list must not be empty."
        ref_data = data_list[0]
        ref_length = len(ref_data)
        ref_device = ref_data.device

        field_dict: dict[str, list[torch.Tensor]] = {k: [] for k in ref_data.keys()}
        for data in data_list:
            assert not data.is_batched, "All layouts in data_list must be non-batched."
            assert len(data) == ref_length, (
                "All layouts in data_list must have the same length."
            )
            assert data.device == ref_device, (
                "All layouts in data_list must be on the same device."
            )
            for name, tensor in data.to_dict().items():
                field_dict[name].append(tensor)

        batched_fields = {
            name: torch.stack(tensors, dim=0) for name, tensors in field_dict.items()
        }
        return cls.from_dict(batched_fields)

    def to_list(self, copy: bool = False) -> list[Self]:
        """Unpack a Batched Layout into a list of data."""
        assert self.is_batched, "Layout is not batched."
        if copy:
            data_dict = {
                name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in self.to_dict().items()
            }
        else:
            data_dict = self.to_dict()

        data_list: list[Self] = []
        batch_size = self.batch_size
        for i in range(batch_size):
            fields = {
                name: tensor[i] if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in data_dict.items()
            }
            data_list.append(self.from_dict(fields))
        return data_list


# === Layout dataclasses (chain-level, token-level, atom-level, bond-level) === #
@dataclass(frozen=True, slots=True)
class ChainLayout(Layout):
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
    def _layout_shape(self) -> tuple[int] | tuple[int, int]:
        return self.pad_mask.shape  # [Nchain,] or [B, Nchain]

    def __post_init__(self):
        # shape: [Nchain,] or [B, Nchain]
        shape = self._layout_shape

        check_tensor(self.chain_type, name="chain_type", dtype=torch_int, shape=(*shape,))
        check_tensor(self.entity_id, name="entity_id", dtype=torch_int, shape=(*shape,))
        check_tensor(self.asym_id, name="asym_id", dtype=torch_int, shape=(*shape,))
        check_tensor(self.sym_id, name="sym_id", dtype=torch_int, shape=(*shape,))
        check_tensor(self.num_tokens, name="num_tokens", dtype=torch_int, shape=(*shape,))
        check_tensor(
            self.num_residues, name="num_residues", dtype=torch_int, shape=(*shape,)
        )
        check_tensor(self.num_atoms, name="num_atoms", dtype=torch_int, shape=(*shape,))

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        L = len(self)
        assert L <= total_length, f"Cannot pad to a smaller length: {total_length} < {L}"

        if L == total_length:
            return self

        # value: PAD_IDX means padding
        pad_values = {
            "chain_type": PAD_IDX,
            "entity_id": PAD_IDX,
            "asym_id": PAD_IDX,
            "sym_id": PAD_IDX,
            "num_tokens": PAD_IDX,
            "num_residues": PAD_IDX,
            "num_atoms": PAD_IDX,
            "pad_mask": False,
        }

        pad_size = total_length - L
        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            pad_shape = [pad_size, *tensor.shape[1:]]
            pad_tensor = torch.full(
                pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            fields[name] = torch.cat([tensor, pad_tensor], dim=0)
        return self.from_dict(fields)

    @cached_property
    def is_protein(self) -> torch.Tensor:
        """Boolean tensor indicating whether the chain is protein."""
        return self.chain_type == C.chain.ChainType.Protein.value

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
        return self.chain_type == C.chain.ChainType.Ligand.value


@dataclass(frozen=True, slots=True)
class TokenLayout(Layout):
    """Token-level layout information.

    Attributes
    ----------
    token_index: torch.Tensor (long)
        Token indices of shape [L,], mapping each token to its position.
        starting from 1.
    res_type: torch.Tensor (long)
        Sequence tokens of shape [L,] (aatype, atom, ...)
    chain_type: torch.Tensor (long)
        Chain types of shape [L,], indicating the type of each token.
    residue_index: torch.Tensor (long)
        Residue indices of shape [L,], used for residue-level operations.
        example)
            4-len polymer(protein/RNA/DNA):
                residue_index: [0, 1, 2, 3]
            6-sized ligand:
                residue_index: [0, 0, 0, 0, 0, 0]
    disto_index: torch.Tensor (long)
        Distortion indices of shape [L,], used for distortion handling.
    center_index: torch.Tensor (long)
        Center indices of shape [L,], used for centering operations.
    frames_index: torch.Tensor (long)
        Frame indices of shape [L, 3], used for frame transformations.
    disto_mask: torch.Tensor (bool)
        Mask tensor of shape [L,], indicating disto atom of tokens to be resolved.
    resolved_mask: torch.Tensor (bool)
        Mask tensor of shape [L,], indicating tokens to be resolved.
    frames_mask: torch.Tensor (bool)
        Boolean tensor of shape [L,], indicating whether the token's frame is resolved.
    pad_mask: torch.Tensor (bool)
        Mask tensor of shape [L,], indicating valid tokens.
    pocket_contact_type: torch.Tensor (int)
        Pocket contact types of shape [L,], indicating pocket contact information.
    """

    token_index: torch.Tensor  # [L,], long
    res_type: torch.Tensor  # [L,], long
    chain_type: torch.Tensor  # [L,], long
    entity_id: torch.Tensor  # [L,], long
    asym_id: torch.Tensor  # [L,], int23, same to sequence_id
    sym_id: torch.Tensor  # [L,], long
    residue_index: torch.Tensor  # [L,], long
    disto_index: torch.Tensor  # [L,], long
    center_index: torch.Tensor  # [L,], long
    frames_index: torch.Tensor  # [L, 3], long
    resolved_mask: torch.Tensor  # [L,], bool
    disto_mask: torch.Tensor  # [L,], bool
    frames_mask: torch.Tensor  # [L,], bool
    pad_mask: torch.Tensor  # [L,], bool
    cyclic_period: torch.Tensor  # [L,], long
    pocket_contact_type: torch.Tensor  # [L,], bool

    @property
    def _layout_shape(self) -> tuple[int] | tuple[int, int]:
        return self.pad_mask.shape  # [Nchain,] or [B, Nchain]

    def __post_init__(self):
        shape = self._layout_shape
        check_tensor(
            self.token_index, name="token_index", dtype=torch_int, shape=(*shape,)
        )
        check_tensor(self.res_type, name="res_type", dtype=torch_int, shape=(*shape,))
        check_tensor(self.chain_type, name="chain_type", dtype=torch_int, shape=(*shape,))
        check_tensor(self.entity_id, name="entity_id", dtype=torch_int, shape=(*shape,))
        check_tensor(self.asym_id, name="asym_id", dtype=torch_int, shape=(*shape,))
        check_tensor(self.sym_id, name="sym_id", dtype=torch_int, shape=(*shape,))
        check_tensor(
            self.residue_index, name="residue_index", dtype=torch_int, shape=(*shape,)
        )
        check_tensor(
            self.disto_index, name="disto_index", dtype=torch_int, shape=(*shape,)
        )
        check_tensor(
            self.center_index, name="center_index", dtype=torch_int, shape=(*shape,)
        )
        check_tensor(
            self.frames_index, name="frames_index", dtype=torch_int, shape=(*shape, 3)
        )
        check_tensor(
            self.resolved_mask, name="resolved_mask", dtype=torch.bool, shape=(*shape,)
        )
        check_tensor(
            self.disto_mask, name="disto_mask", dtype=torch.bool, shape=(*shape,)
        )
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=(*shape,))
        check_tensor(
            self.frames_mask, name="frames_mask", dtype=torch.bool, shape=(*shape,)
        )
        check_tensor(
            self.cyclic_period, name="cyclic_period", dtype=torch_int, shape=(*shape,)
        )
        check_tensor(
            self.pocket_contact_type,
            name="pocket_contact_type",
            dtype=torch_int,
            shape=(*shape,),
        )

    @cached_property
    def is_protein(self) -> torch.Tensor:
        """Boolean tensor of shape [L,], indicating whether the token is protein."""
        return self.chain_type == C.chain.ChainType.Protein.value

    @cached_property
    def is_dna(self) -> torch.Tensor:
        """Boolean tensor of shape [L,], indicating whether the token is dna."""
        return self.chain_type == C.chain.ChainType.DNA.value

    @cached_property
    def is_rna(self) -> torch.Tensor:
        """Boolean tensor of shape [L,], indicating whether the token is rna."""
        return self.chain_type == C.chain.ChainType.RNA.value

    @cached_property
    def is_ligand(self) -> torch.Tensor:
        """Boolean tensor of shape [L,], indicating whether the token is ligand."""
        return self.chain_type == C.chain.ChainType.Ligand.value

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."

        L = len(self)
        assert L <= total_length, f"Cannot pad to a smaller length: {total_length} < {L}"

        if L == total_length:
            return self

        # value: PAD_IDX means padding
        pad_values = {
            "token_index": PAD_IDX,
            "res_type": 0,
            "chain_type": PAD_IDX,
            "entity_id": 0,
            "asym_id": 0,
            "sym_id": 0,
            "residue_index": PAD_IDX,
            "disto_index": PAD_IDX,
            "center_index": PAD_IDX,
            "frames_index": PAD_IDX,
            "resolved_mask": False,
            "disto_mask": False,
            "frames_mask": False,
            "pad_mask": False,
            "cyclic_period": 0,
            "pocket_contact_type": 0,
        }

        pad_size = total_length - L
        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            pad_shape = [pad_size] + list(tensor.shape)[1:]
            pad_tensor = torch.full(
                pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            fields[name] = torch.cat([tensor, pad_tensor], dim=0)
        return self.from_dict(fields)


@dataclass(frozen=True, slots=True)
class AtomLayout(Layout):
    """Atom-level layout information for molecular structures.

    Shape: [Nchain, ...] or [B, Nchain, ...]

    Attributes
    ----------
    ref_atom_name_chars: torch.Tensor (long)
        Encoded atom name of shape [Natom, 4].
        To be encoded as one-hot vector of size 64.
    ref_element: torch.Tensor (long)
        One-hot encoded atomic numbers of shape [Natom,].
        To be encoded as one-hot vector of size 128.
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
        Apo (unbound) state coordinates of shape [Natom, Napo, 3],
        where Napo is the number of apo conformations.
    resolved_mask: torch.Tensor (bool)
        Boolean mask of shape [Natom,] indicating atoms to be resolved.
    pad_mask: torch.Tensor (bool)
        Boolean mask of shape [Natom,] indicating valid (non-padded) atoms.
    label_coords: torch.Tensor (float32)
        Holo (bound) state coordinates of shape [Natom, Nholo, 3],
        where Nholo is the number of ensemble holo conformations.
        This is used as the ground truth for training, and may be set to 0
        for inference.
    """

    ref_atom_name_chars: torch.Tensor  # [Natom, 4], long, to be encoded one-hot (64-dim)
    ref_element: torch.Tensor  # [Natom], long, to be encoded one-hot (128-dim)
    ref_charge: torch.Tensor  # [Natom,], float32
    ref_pos: torch.Tensor  # [Natom, Nholo, 3], float32
    ref_space_uid: torch.Tensor  # [Natom,], long
    token_index: torch.Tensor  # [Natom,], long
    apo_coords: torch.Tensor  # [Natom, Napo, 3], float32
    resolved_mask: torch.Tensor  # [Natom,], bool
    pad_mask: torch.Tensor  # [Natom,], bool
    label_coords: torch.Tensor  # [Natom, 3], float32

    @property
    def _layout_shape(self) -> tuple[int] | tuple[int, int]:
        return self.pad_mask.shape

    def __post_init__(self):
        shape = self._layout_shape
        check_tensor(
            self.ref_atom_name_chars,
            name="ref_atom_name_chars",
            dtype=torch_int,
            shape=(*shape, 4),
        )
        check_tensor(
            self.ref_element, name="ref_element", dtype=torch_int, shape=(*shape,)
        )
        check_tensor(
            self.ref_charge, name="ref_charge", dtype=torch_float, shape=(*shape,)
        )
        check_tensor(self.ref_pos, name="ref_pos", dtype=torch_float, shape=(*shape, 3))
        check_tensor(
            self.ref_space_uid, name="ref_space_uid", dtype=torch_int, shape=(*shape,)
        )
        check_tensor(
            self.token_index, name="token_index", dtype=torch_int, shape=(*shape,)
        )
        check_tensor(
            self.apo_coords, name="apo_coords", dtype=torch_float, shape=(*shape, -1, 3)
        )
        check_tensor(
            self.resolved_mask, name="resolved_mask", dtype=torch.bool, shape=(*shape,)
        )
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=(*shape,))
        check_tensor(
            self.label_coords,
            name="label_coords",
            dtype=torch_float,
            shape=(*shape, -1, 3),
        )

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        L = len(self)
        assert L <= total_length, f"Cannot pad to a smaller length: {total_length} < {L}"

        if L == total_length:
            return self

        # value: PAD_IDX means padding
        pad_values = {
            "ref_atom_name_chars": 63,  # max value for atom_name encoding
            "ref_element": 127,  # max value for element encoding
            "ref_charge": 0.0,
            "ref_pos": 0.0,
            "ref_space_uid": PAD_IDX,
            "token_index": 0,
            "apo_coords": 0.0,
            "resolved_mask": False,
            "pad_mask": False,
            "label_coords": 0.0,
        }

        pad_size = total_length - L
        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            pad_shape = [pad_size] + list(tensor.shape)[1:]
            pad_tensor = torch.full(
                pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            fields[name] = torch.cat([tensor, pad_tensor], dim=0)
        return self.from_dict(fields)


@dataclass(frozen=True, slots=True)
class BondLayout(Layout):
    """Bond-level layout information for molecular structures.

    Shape: [Nchain, ...] or [B, Nchain, ...]

    Attributes
    ----------
    asym_id: torch.Tensor
        Chain asym indices of the connecting atoms in the bond of shape [Nbond,].
    token_index: torch.Tensor
        Token indices of the connecting atoms in the bond of shape [Nbond,].
    atom_index: torch.Tensor
        Atom indices of the connecting atoms in the bond of shape [Nbond,].
        Atom indices of the first atom in the bond of shape [Nbond,].
    atom_index_2: torch.Tensor
        Atom indices of the second atom in the bond of shape [Nbond,].
    """

    asym_id: torch.Tensor  # [Nbond,], long
    token_index: torch.Tensor  # [Nbond,], long
    atom_index: torch.Tensor  # [Nbond,], long
    bond_type: torch.Tensor  # [Nbond,], long
    pad_mask: torch.Tensor  # [Nbond,], bool

    @property
    def _layout_shape(self) -> tuple[int] | tuple[int, int]:
        return self.pad_mask.shape

    def __post_init__(self):
        shape = self._layout_shape
        check_tensor(self.asym_id, name="asym_id", dtype=torch_int, shape=(*shape, 2))
        check_tensor(
            self.token_index, name="token_index", dtype=torch_int, shape=(*shape, 2)
        )
        check_tensor(
            self.atom_index, name="atom_index", dtype=torch_int, shape=(*shape, 2)
        )
        check_tensor(self.bond_type, name="bond_type", dtype=torch_int, shape=(*shape,))
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=(*shape,))

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        assert not self.is_batched, "Padding batched layout is not supported."
        L = len(self)
        assert L <= total_length, f"Cannot pad to a smaller length: {total_length} < {L}"

        if L == total_length:
            return self

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
        }

        pad_size = total_length - L
        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            pad_shape = [pad_size] + list(tensor.shape)[1:]
            pad_tensor = torch.full(
                pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            fields[name] = torch.cat([tensor, pad_tensor], dim=0)
        return self.from_dict(fields)


@dataclass(frozen=True, slots=False)
class FoldingInput:
    """Input of co-folding"""

    chain: ChainLayout
    token: TokenLayout
    atom: AtomLayout
    bond: BondLayout
    metadata: dict[str, Any] | None = None  # Optional metadata

    def __post_init__(self):
        # check all layouts are on the same device
        device = self.chain.device
        assert self.token.device == device, "token layout must be on the same device."
        assert self.atom.device == device, "atom layout must be on the same device."
        assert self.bond.device == device, "bond layout must be on the same device."

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

    def to(self, device: str | torch.device) -> Self:
        return self.__class__(
            chain=self.chain.to(device),
            token=self.token.to(device),
            atom=self.atom.to(device),
            bond=self.bond.to(device),
            metadata=self.metadata,
        )

    @property
    def device(self) -> torch.device:
        return self.token.res_type.device

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
    def from_list(cls, data_list: list[Self]) -> Self:
        """Create a Batched Input from a list of data."""
        # Check all data are non-batched
        for data in data_list:
            assert not data.is_batched, "All inputs in data_list must be non-batched."

        # Check all data are on the same device
        ref_device = data_list[0].device
        for data in data_list:
            assert data.device == ref_device, (
                "All inputs in data_list must be on the same device."
            )

        batched_chain = ChainLayout.from_list([data.chain for data in data_list])
        batched_token = TokenLayout.from_list([data.token for data in data_list])
        batched_atom = AtomLayout.from_list([data.atom for data in data_list])
        batched_bond = BondLayout.from_list([data.bond for data in data_list])

        return cls(
            chain=batched_chain,
            token=batched_token,
            atom=batched_atom,
            bond=batched_bond,
            metadata=None,
        )

    def to_list(self, copy: bool = False) -> list[Self]:
        chain_list = self.chain.to_list(copy=copy)
        token_list = self.token.to_list(copy=copy)
        atom_list = self.atom.to_list(copy=copy)
        bond_list = self.bond.to_list(copy=copy)

        data_list: list[Self] = []
        batch_size = self.batch_size
        for b in range(batch_size):
            data_list.append(
                self.__class__(
                    chain=chain_list[b],
                    token=token_list[b],
                    atom=atom_list[b],
                    bond=bond_list[b],
                    metadata=None,
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

    def extract_chains(self, asym_ids: list[int]) -> Self:
        """Extract specific chains using asym ids"""
        assert not self.is_batched, "Batched input is not supported."
        raise NotImplementedError("extract_chains() is not implemented yet.")

    def __repr__(self) -> str:
        """FoldingInput summary representation."""
        device = self.device

        # Summary statistics
        num_chains = self.num_chains
        num_tokens = self.num_tokens
        num_atoms = self.num_atoms
        num_bonds = len(self.bond)

        # Metadata
        if self.metadata:
            metadata_keys = list(self.metadata.keys())
        else:
            metadata_keys = []

        if self.is_batched:
            return (
                f"FoldingInput(\n"
                f"  batch_size: {self.batch_size}\n"
                f"  num_chains: {num_chains}\n"
                f"  num_tokens: {num_tokens}\n"
                f"  num_atoms: {num_atoms}\n"
                f"  num_bonds: {num_bonds}\n"
                f"  metadata: {metadata_keys}\n"
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
                f"  metadata: {metadata_keys}\n"
                f"  device: {device}\n"
                f")"
            )
