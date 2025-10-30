import copy
from dataclasses import dataclass
from typing import Any, Self

import torch

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
    "ChainInput",
    "FoldingInput",
]

# common type alias
torch_float = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
torch_int = (torch.int32, torch.int64)


class TensorObj:
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


class Layout(TensorObj):
    """Base class for layout information."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int | slice | torch.Tensor) -> Self:
        """Get a subset of the layout."""
        # Ensure that idx is not integer
        if isinstance(idx, int):
            raise ValueError(
                f"Integer indexing is not supported for {self.__class__.__name__}. "
                f"Use slicing instead: obj[{idx}:{idx + 1}]"
            )
        return super().__getitem__(idx)

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ChainLayout(Layout):
    """Chain-level layout information.

    Attributes
    ----------
    chain_type: torch.Tensor
        Chain types of shape [Nchain,], indicating the type of each chain.
    entity_id: torch.Tensor
        Entity IDs of shape [Nchain,], starting from 1.
    asym_id: torch.Tensor
        Asymmetric unit IDs of shape [Nchain,], starting from 1.
    sym_id: torch.Tensor
        Symmetry IDs of shape [Nchain,], starting from 1.
    num_tokens: torch.Tensor
        Number of tokens per chain of shape [Nchain,].

    # NOTE: this layout might not be used in model.forward(),
    # but it is useful for data processing and analysis.

    """

    chain_type: torch.Tensor  # [Nchain,], int
    entity_id: torch.Tensor  # [Nchain,], int
    asym_id: torch.Tensor  # [Nchain,], int
    sym_id: torch.Tensor  # [Nchain,], int
    num_tokens: torch.Tensor  # [Nchain,], int
    num_atoms: torch.Tensor  # [Nchain,], int
    pad_mask: torch.Tensor  # [Nchain,], bool

    def __len__(self) -> int:
        return self.entity_id.shape[0]

    def __post_init__(self):
        N = len(self)
        check_tensor(self.chain_type, name="chain_type", dtype=torch_int, shape=(N,))
        check_tensor(self.entity_id, name="entity_id", dtype=torch_int, shape=(N,))
        check_tensor(self.asym_id, name="asym_id", dtype=torch_int, shape=(N,))
        check_tensor(self.sym_id, name="sym_id", dtype=torch_int, shape=(N,))
        check_tensor(self.num_tokens, name="num_tokens", dtype=torch_int, shape=(N,))
        check_tensor(self.num_atoms, name="num_atoms", dtype=torch_int, shape=(N,))

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        N = len(self)
        assert N <= total_length, f"Cannot pad to a smaller length: {total_length} < {N}"

        if N == total_length:
            return self

        # value: PAD_IDX means padding
        pad_values = {
            "chain_type": PAD_IDX,
            "entity_id": PAD_IDX,
            "asym_id": PAD_IDX,
            "sym_id": PAD_IDX,
            "num_tokens": PAD_IDX,
            "num_atoms": PAD_IDX,
            "pad_mask": False,
        }

        pad_size = total_length - N
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
class TokenLayout(Layout):
    """Token-level layout information.

    Attributes
    ----------
    token_type: torch.Tensor
        Sequence tokens of shape [L,] (aatype, atom, ...)
    chain_type: torch.Tensor
        Chain types of shape [L,], indicating the type of each token.
    residue_idx: torch.Tensor:
        Residue indices of shape [L,], used for residue-level operations.
        example)
            4-len polymer(protein/RNA/DNA):
                residue_idx: [0, 1, 2, 3]
            6-sized ligand:
                residue_idx: [0, 0, 0, 0, 0, 0]
    disto_idx: torch.Tensor
        Distortion indices of shape [L,], used for distortion handling.
    center_idx: torch.Tensor
        Center indices of shape [L,], used for centering operations.
    resolve_mask: torch.Tensor
        Mask tensor of shape [L,], indicating tokens to be resolved.
    pad_mask: torch.Tensor
        Mask tensor of shape [L,], indicating valid tokens.
    is_pocket: torch.Tensor
        Boolean tensor of shape [L,], indicating whether the token is part of a pocket
    """

    token_type: torch.Tensor  # [L,], int
    chain_type: torch.Tensor  # [L,], int
    asym_id: torch.Tensor  # [L,], int, same to sequence_id
    entity_id: torch.Tensor  # [L,], int
    sym_id: torch.Tensor  # [L,], int
    residue_idx: torch.Tensor  # [L,], int
    disto_idx: torch.Tensor  # [L,], int
    center_idx: torch.Tensor  # [L,], int
    cyclic_period: torch.Tensor  # [L,], int
    resolve_mask: torch.Tensor  # [L,], bool
    pad_mask: torch.Tensor  # [L,], bool
    is_pocket: torch.Tensor  # [L,], bool

    def __len__(self) -> int:
        return self.token_type.shape[0]

    def __post_init__(self):
        L = len(self)
        check_tensor(self.token_type, name="token_type", dtype=torch_int, shape=(L,))
        check_tensor(self.chain_type, name="chain_type", dtype=torch_int, shape=(L,))
        check_tensor(self.asym_id, name="asym_id", dtype=torch_int, shape=(L,))
        check_tensor(self.entity_id, name="entity_id", dtype=torch_int, shape=(L,))
        check_tensor(self.sym_id, name="sym_id", dtype=torch_int, shape=(L,))
        check_tensor(self.residue_idx, name="residue_idx", dtype=torch_int, shape=(L,))
        check_tensor(self.disto_idx, name="disto_idx", dtype=torch_int, shape=(L,))
        check_tensor(self.center_idx, name="center_idx", dtype=torch_int, shape=(L,))
        check_tensor(
            self.cyclic_period, name="cyclic_period", dtype=torch_int, shape=(L,)
        )
        check_tensor(self.resolve_mask, name="resolve_mask", dtype=torch.bool, shape=(L,))
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=(L,))
        check_tensor(self.is_pocket, name="is_pocket", dtype=torch.bool, shape=(L,))

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        L = len(self)
        assert L <= total_length, f"Cannot pad to a smaller length: {total_length} < {L}"

        if L == total_length:
            return self

        # value: PAD_IDX means padding
        pad_values = {
            "token_type": PAD_IDX,
            "chain_type": PAD_IDX,
            "asym_id": PAD_IDX,
            "entity_id": PAD_IDX,
            "sym_id": PAD_IDX,
            "residue_idx": PAD_IDX,
            "disto_idx": PAD_IDX,
            "center_idx": PAD_IDX,
            "cyclic_period": 0,
            "resolve_mask": False,
            "pad_mask": False,
            "is_pocket": False,
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

    Attributes
    ----------
    atom_type: torch.Tensor
        Unique atom identifiers of shape [Natom,].
    element: torch.Tensor
        Atomic numbers of shape [Natom,].
    charge: torch.Tensor
        Formal charges of shape [Natom,].
    token_idx: torch.Tensor
        Token indices mapping atoms to their parent tokens of shape [Natom,].
    apo_coords: torch.Tensor
        Apo (unbound) state coordinates of shape [Natom, Napo, 3],
        where Napo is the number of apo conformations.
    resolve_mask: torch.Tensor
        Boolean mask of shape [Natom,] indicating atoms to be resolved.
    pad_mask: torch.Tensor
        Boolean mask of shape [Natom,] indicating valid (non-padded) atoms.
    label_coords: torch.Tensor
        Holo (bound) state coordinates of shape [Natom, 3].
        This is used as the ground truth for training, and may be set to 0
        for inference.
    """

    atom_type: torch.Tensor  # [Natom,], int
    element: torch.Tensor  # [Natom,], int (Atomic Num)
    charge: torch.Tensor  # [Natom,], int
    token_idx: torch.Tensor  # [Natom,], int
    apo_coords: torch.Tensor  # [Natom, Napo, 3], float
    resolve_mask: torch.Tensor  # [Natom,], bool
    pad_mask: torch.Tensor  # [Natom,], bool
    label_coords: torch.Tensor  # [Natom, 3], float

    def __len__(self) -> int:
        return self.atom_type.shape[0]

    def __post_init__(self):
        N = len(self)
        check_tensor(self.atom_type, name="atom_type", dtype=torch_int, shape=(N,))
        check_tensor(self.element, name="element", dtype=torch_int, shape=(N,))
        check_tensor(self.charge, name="charge", dtype=torch_int, shape=(N,))
        check_tensor(self.token_idx, name="token_idx", dtype=torch_int, shape=(N,))
        check_tensor(
            self.apo_coords, name="apo_coords", dtype=torch_float, shape=(N, -1, 3)
        )
        check_tensor(self.resolve_mask, name="resolve_mask", dtype=torch.bool, shape=(N,))
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=(N,))
        check_tensor(
            self.label_coords, name="label_coords", dtype=torch_float, shape=(N, 3)
        )

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        N = len(self)
        assert N <= total_length, f"Cannot pad to a smaller length: {total_length} < {N}"

        if N == total_length:
            return self

        # value: PAD_IDX means padding
        pad_values = {
            "atom_type": PAD_IDX,
            "element": PAD_IDX,
            "charge": PAD_IDX,
            "token_idx": PAD_IDX,
            "apo_coords": 0.0,
            "resolve_mask": False,
            "pad_mask": False,
            "label_coords": 0.0,
        }

        pad_size = total_length - N
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

    Attributes
    ----------
    asym_id_1: torch.Tensor
        Chain asym indices of the first atom in the bond of shape [Nbond,].
    asym_id_2: torch.Tensor
        Chain asym indices of the second atom in the bond of shape [Nbond,].
    token_idx_1: torch.Tensor
        Token indices of the first atom in the bond of shape [Nbond,].
    token_idx_2: torch.Tensor
        Token indices of the second atom in the bond of shape [Nbond,].
    atom_idx_1: torch.Tensor
        Atom indices of the first atom in the bond of shape [Nbond,].
    atom_idx_2: torch.Tensor
        Atom indices of the second atom in the bond of shape [Nbond,].
    """

    asym_id_1: torch.Tensor  # [Nbond,], int
    asym_id_2: torch.Tensor  # [Nbond,], int
    token_idx_1: torch.Tensor  # [Nbond,], int
    token_idx_2: torch.Tensor  # [Nbond,], int
    atom_idx_1: torch.Tensor  # [Nbond,], int
    atom_idx_2: torch.Tensor  # [Nbond,], int
    bond_type: torch.Tensor  # [Nbond,], int
    pad_mask: torch.Tensor  # [Nbond,], bool

    def __len__(self) -> int:
        return self.atom_idx_1.shape[0]

    def __post_init__(self):
        N = len(self)
        check_tensor(self.asym_id_1, name="asym_id_1", dtype=torch_int, shape=(N,))
        check_tensor(self.asym_id_2, name="asym_id_2", dtype=torch_int, shape=(N,))
        check_tensor(self.token_idx_1, name="token_idx_1", dtype=torch_int, shape=(N,))
        check_tensor(self.token_idx_2, name="token_idx_2", dtype=torch_int, shape=(N,))
        check_tensor(self.atom_idx_1, name="atom_idx_1", dtype=torch_int, shape=(N,))
        check_tensor(self.atom_idx_2, name="atom_idx_2", dtype=torch_int, shape=(N,))
        check_tensor(self.bond_type, name="bond_type", dtype=torch_int, shape=(N,))
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=(N,))

    def pad(self, total_length: int) -> Self:
        """Pad the layout to the total length."""
        N = len(self)
        assert N <= total_length, f"Cannot pad to a smaller length: {total_length} < {N}"

        if N == total_length:
            return self

        # value: PAD_IDX means padding
        pad_values = {
            "asym_id_1": PAD_IDX,
            "asym_id_2": PAD_IDX,
            "token_idx_1": PAD_IDX,
            "token_idx_2": PAD_IDX,
            "atom_idx_1": PAD_IDX,
            "atom_idx_2": PAD_IDX,
            "bond_type": PAD_IDX,
            "pad_mask": False,
        }

        pad_size = total_length - N
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
class ChainInput:
    """Input of co-folding"""

    chain_type: int
    entity_id: int
    asym_id: str
    sym_id: int
    token: TokenLayout
    atom: AtomLayout
    bond: BondLayout

    def to(self, device: str | torch.device) -> Self:
        return self.__class__(
            chain_type=self.chain_type,
            entity_id=self.entity_id,
            asym_id=self.asym_id,
            sym_id=self.sym_id,
            token=self.token.to(device),
            atom=self.atom.to(device),
            bond=self.bond.to(device),
        )

    @property
    def device(self) -> torch.device:
        return self.token.token_type.device


@dataclass(frozen=True, slots=False)
class FoldingInput:
    """Input of co-folding"""

    chain: ChainLayout
    token: TokenLayout
    atom: AtomLayout
    bond: BondLayout
    metadata: dict[str, Any] | None = None  # Optional metadata

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
        return self.token.token_type.device

    def to_chains(self) -> list[ChainInput]:
        """Split the FoldingInput into a list of ChainInput for each chain.

        WARNING: Inter-chain bond information will be lost.
        """
        Nchain = int(self.chain.pad_mask.sum().item())

        chain_type_list = self.chain.chain_type.tolist()
        entity_id_list = self.chain.entity_id.tolist()
        asym_id_list = self.chain.asym_id.tolist()
        sym_id_list = self.chain.sym_id.tolist()
        num_tokens_list = self.chain.num_tokens.tolist()
        num_atoms_list = self.chain.num_atoms.tolist()

        chains: list[ChainInput] = []
        token_offset = 0
        atom_offset = 0
        for i in range(Nchain):
            num_tokens = num_tokens_list[i]
            num_atoms = num_atoms_list[i]

            # get token layout
            chain_tokens = self.token[token_offset : token_offset + num_tokens]
            # update
            chain_tokens = chain_tokens.copy_with(
                disto_idx=chain_tokens.disto_idx - atom_offset,
                center_idx=chain_tokens.center_idx - atom_offset,
            )

            # get atom layout
            chain_atoms = self.atom[atom_offset : atom_offset + num_atoms]
            # update
            chain_atoms = chain_atoms.copy_with(
                token_idx=chain_atoms.token_idx - token_offset,
            )

            # get bond layout
            asym_id = asym_id_list[i]
            bond_mask = (self.bond.asym_id_1 == asym_id) & (
                self.bond.asym_id_2 == asym_id
            )
            chain_bonds = self.bond[bond_mask]
            # update
            chain_bonds = chain_bonds.copy_with(
                token_idx_1=chain_bonds.token_idx_1 - token_offset,
                token_idx_2=chain_bonds.token_idx_2 - token_offset,
                atom_idx_1=chain_bonds.atom_idx_1 - atom_offset,
                atom_idx_2=chain_bonds.atom_idx_2 - atom_offset,
            )

            chain_input = ChainInput(
                chain_type=chain_type_list[i],
                entity_id=entity_id_list[i],
                asym_id=asym_id_list[i],
                sym_id=sym_id_list[i],
                token=chain_tokens,
                atom=chain_atoms,
                bond=chain_bonds,
            )
            chains.append(chain_input)
            token_offset += num_tokens
            atom_offset += num_atoms
        return chains

    @classmethod
    def from_chains(cls, chains: list[ChainInput]) -> Self:
        """Combine a list of ChainInput into a single FoldingInput."""
        assert len(chains) > 0, "chains must not be empty"

        device = chains[0].device
        # check all chains are on the same device
        for chain in chains:
            assert chain.device == device, (
                f"All chains must be on the same device,"
                f" but got {chain.device} and {device}"
            )

        chain_fields = {
            "chain_type": [],
            "entity_id": [],
            "asym_id": [],
            "sym_id": [],
            "num_tokens": [],
            "num_atoms": [],
        }
        token_tensors = {name: [] for name in chains[0].token.to_dict().keys()}
        atom_tensors = {name: [] for name in chains[0].atom.to_dict().keys()}
        bond_tensors = {name: [] for name in chains[0].bond.to_dict().keys()}

        token_offset = 0
        atom_offset = 0
        for chain in chains:
            chain_fields["chain_type"].append(chain.chain_type)
            chain_fields["entity_id"].append(chain.entity_id)
            chain_fields["asym_id"].append(chain.asym_id)
            chain_fields["sym_id"].append(chain.sym_id)
            chain_fields["num_tokens"].append(len(chain.token))
            chain_fields["num_atoms"].append(len(chain.atom))

            for name, tensor in chain.token.to_dict().items():
                if name in ("disto_idx", "center_idx"):
                    tensor = tensor + atom_offset
                token_tensors[name].append(tensor)

            for name, tensor in chain.atom.to_dict().items():
                if name == "token_idx":
                    tensor = tensor + token_offset
                atom_tensors[name].append(tensor)

            for name, tensor in chain.bond.to_dict().items():
                if name in ("token_idx_1", "token_idx_2"):
                    tensor = tensor + token_offset
                elif name in ("atom_idx_1", "atom_idx_2"):
                    tensor = tensor + atom_offset
                bond_tensors[name].append(tensor)

            token_offset += len(chain.token)
            atom_offset += len(chain.atom)

        chain_layout = ChainLayout(
            chain_type=torch.tensor(
                chain_fields["chain_type"], dtype=torch.int, device=device
            ),
            entity_id=torch.tensor(
                chain_fields["entity_id"], dtype=torch.int, device=device
            ),
            asym_id=torch.tensor(chain_fields["asym_id"], dtype=torch.int, device=device),
            sym_id=torch.tensor(chain_fields["sym_id"], dtype=torch.int, device=device),
            num_tokens=torch.tensor(
                chain_fields["num_tokens"], dtype=torch.int, device=device
            ),
            num_atoms=torch.tensor(
                chain_fields["num_atoms"], dtype=torch.int, device=device
            ),
            pad_mask=torch.ones(len(chains), dtype=torch.bool, device=device),
        )

        token_layout = TokenLayout(
            **{name: torch.cat(tensors, dim=0) for name, tensors in token_tensors.items()}
        )

        atom_layout = AtomLayout(
            **{name: torch.cat(tensors, dim=0) for name, tensors in atom_tensors.items()}
        )

        bond_layout = BondLayout(
            **{name: torch.cat(tensors, dim=0) for name, tensors in bond_tensors.items()}
        )

        return cls(
            chain=chain_layout,
            token=token_layout,
            atom=atom_layout,
            bond=bond_layout,
        )
