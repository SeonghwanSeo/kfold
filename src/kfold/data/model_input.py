from dataclasses import dataclass
from functools import cached_property
from typing import Any, Self

import torch

import kfold.constants as C
from kfold.data.layout import TensorLayout
from kfold.utils.misc import check_tensor

PAD_IDX = 2**16 - 1  # 65535

__all__ = [
    "ChainLayout",
    "TokenLayout",
    "AtomLayout",
    "BondLayout",
    "FoldingInput",
]


# === Layout dataclasses (chain-level, token-level, atom-level, bond-level) === #
@dataclass(frozen=True, slots=True)
class ChainLayout(TensorLayout):
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

        check_tensor(
            self.chain_type, name="chain_type", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(self.entity_id, name="entity_id", dtype=torch.long, shape=(*shape,))
        check_tensor(self.asym_id, name="asym_id", dtype=torch.long, shape=(*shape,))
        check_tensor(self.sym_id, name="sym_id", dtype=torch.long, shape=(*shape,))
        check_tensor(
            self.num_tokens, name="num_tokens", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(
            self.num_residues, name="num_residues", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(self.num_atoms, name="num_atoms", dtype=torch.long, shape=(*shape,))

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
            to_shape = (pad_size,) + tensor.shape[1:]
            pad_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
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
class TokenLayout(TensorLayout):
    """Token-level layout information.

    Attributes
    ----------
    token_index: torch.Tensor (long)
        Token indices of shape [L,], mapping each token to its position.
    org_token_index: torch.Tensor (long)
        Original token indices of shape [L,], before cropping.
    res_type: torch.Tensor (float32)
        Sequence tokens of shape [L, 32] (aatype, atom, ...)
        One-hot vector
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
        Representative atom indices of shape [L,], Cβ
    center_index: torch.Tensor (long)
        Center indices of shape [L,], Cα
    frames_index: torch.Tensor (long)
        Frame indices of shape [L, 3].
    disto_coords: torch.Tensor (float32)
        Representative atom center coordinates of shape [L, 3].
    center_coords: torch.Tensor (float32)
        Center coordinates of shape [L, 3].
    disto_mask: torch.Tensor (bool)
        Mask tensor of shape [L,], indicating disto atom of tokens to be resolved.
    resolved_mask: torch.Tensor (bool)
        Mask tensor of shape [L,], indicating tokens to be resolved.
    frames_mask: torch.Tensor (bool)
        Boolean tensor of shape [L,], indicating whether the token's frame is resolved.
    pad_mask: torch.Tensor (bool)
        Mask tensor of shape [L,], indicating valid tokens.
    pocket_contact_type: torch.Tensor (long)
        Pocket contact types of shape [L,], indicating pocket contact information.
    """

    token_index: torch.Tensor  # [L,], long
    org_token_index: torch.Tensor  # [L,], long
    res_type: torch.Tensor  # [L, 32], float32
    chain_type: torch.Tensor  # [L,], long
    entity_id: torch.Tensor  # [L,], long
    asym_id: torch.Tensor  # [L,], long, same to sequence_id
    sym_id: torch.Tensor  # [L,], long
    residue_index: torch.Tensor  # [L,], long
    disto_index: torch.Tensor  # [L,], long
    center_index: torch.Tensor  # [L,], long
    frames_index: torch.Tensor  # [L, 3], long
    disto_coords: torch.Tensor  # [L, 3], long
    center_coords: torch.Tensor  # [L, 3], long
    resolved_mask: torch.Tensor  # [L,], bool
    disto_mask: torch.Tensor  # [L,], bool
    frames_mask: torch.Tensor  # [L,], bool
    pad_mask: torch.Tensor  # [L,], bool
    pocket_contact_type: torch.Tensor  # [L,], bool

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
            self.token_index, name="token_index", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(
            self.org_token_index,
            name="org_token_index",
            dtype=torch.long,
            shape=(*shape,),
        )
        check_tensor(
            self.res_type, name="res_type", dtype=torch.float32, shape=(*shape, 32)
        )
        check_tensor(
            self.chain_type, name="chain_type", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(self.entity_id, name="entity_id", dtype=torch.long, shape=(*shape,))
        check_tensor(self.asym_id, name="asym_id", dtype=torch.long, shape=(*shape,))
        check_tensor(self.sym_id, name="sym_id", dtype=torch.long, shape=(*shape,))
        check_tensor(
            self.residue_index, name="residue_index", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(
            self.disto_index, name="disto_index", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(
            self.center_index, name="center_index", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(
            self.frames_index, name="frames_index", dtype=torch.long, shape=(*shape, 3)
        )
        check_tensor(
            self.disto_coords, name="disto_coords", dtype=torch.float32, shape=(*shape, 3)
        )
        check_tensor(
            self.center_coords,
            name="center_coords",
            dtype=torch.float32,
            shape=(*shape, 3),
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
            self.pocket_contact_type,
            name="pocket_contact_type",
            dtype=torch.long,
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
            "token_index": 0,
            "org_token_index": 0,
            "res_type": 0,
            "chain_type": PAD_IDX,
            "entity_id": 0,
            "asym_id": 0,
            "sym_id": 0,
            "residue_index": PAD_IDX,
            "disto_index": PAD_IDX,
            "center_index": PAD_IDX,
            "frames_index": PAD_IDX,
            "disto_coords": 0.0,
            "center_coords": 0.0,
            "resolved_mask": False,
            "disto_mask": False,
            "frames_mask": False,
            "pad_mask": False,
            "pocket_contact_type": 0,
        }

        pad_size = total_length - L
        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            to_shape = (pad_size,) + tensor.shape[1:]
            pad_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            fields[name] = torch.cat([tensor, pad_tensor], dim=0)
        return self.from_dict(fields)


@dataclass(frozen=True, slots=True)
class AtomLayout(TensorLayout):
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

    ref_atom_name_chars: torch.Tensor  # [Natom, 4, 64], float32
    ref_element: torch.Tensor  # [Natom, 128], float32
    ref_charge: torch.Tensor  # [Natom,], float32
    ref_pos: torch.Tensor  # [Natom, Nholo, 3], float32
    ref_space_uid: torch.Tensor  # [Natom,], long
    token_index: torch.Tensor  # [Natom,], long
    apo_coords: torch.Tensor  # [Natom, Napo, 3], float32
    resolved_mask: torch.Tensor  # [Natom,], bool
    pad_mask: torch.Tensor  # [Natom,], bool
    label_coords: torch.Tensor  # [Natom, 3], float32

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
        check_tensor(
            self.ref_charge, name="ref_charge", dtype=torch.float32, shape=(*shape,)
        )
        check_tensor(self.ref_pos, name="ref_pos", dtype=torch.float32, shape=(*shape, 3))
        check_tensor(
            self.ref_space_uid, name="ref_space_uid", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(
            self.token_index, name="token_index", dtype=torch.long, shape=(*shape,)
        )
        check_tensor(
            self.apo_coords, name="apo_coords", dtype=torch.float32, shape=(*shape, -1, 3)
        )
        check_tensor(
            self.resolved_mask, name="resolved_mask", dtype=torch.bool, shape=(*shape,)
        )
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=(*shape,))
        check_tensor(
            self.label_coords,
            name="label_coords",
            dtype=torch.float32,
            shape=(*shape, -1, 3),
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
            "ref_space_uid": -1,
            "ref_token_index": PAD_IDX,
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
            to_shape = (pad_size,) + tensor.shape[1:]
            pad_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            fields[name] = torch.cat([tensor, pad_tensor], dim=0)
        return self.from_dict(fields)


@dataclass(frozen=True, slots=True)
class BondLayout(TensorLayout):
    """Bond-level layout information for molecular structures.

    Shape: [Nchain, ...] or [B, Nchain, ...]

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
    is_ligand_ligand: torch.Tensor  # [Nbond,], bool
    is_polymer_ligand: torch.Tensor  # [Nbond,], bool
    pad_mask: torch.Tensor  # [Nbond,], bool

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
        check_tensor(self.bond_type, name="bond_type", dtype=torch.long, shape=(*shape,))
        check_tensor(self.pad_mask, name="pad_mask", dtype=torch.bool, shape=(*shape,))
        check_tensor(
            self.is_ligand_ligand,
            name="is_ligand_ligand",
            dtype=torch.bool,
            shape=(*shape,),
        )
        check_tensor(
            self.is_polymer_ligand,
            name="is_polymer_ligand",
            dtype=torch.bool,
            shape=(*shape,),
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

        pad_size = total_length - L
        fields = {}
        for name, tensor in self.to_dict().items():
            pad_value = pad_values[name]
            to_shape = (pad_size,) + tensor.shape[1:]
            pad_tensor = torch.full(
                to_shape, pad_value, dtype=tensor.dtype, device=tensor.device
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
    metadata: Any = None

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
        batched_metadata = [data.metadata for data in data_list]

        return cls(
            chain=batched_chain,
            token=batched_token,
            atom=batched_atom,
            bond=batched_bond,
            metadata=batched_metadata,
        )

    def to_list(self, deepcopy: bool = False) -> list[Self]:
        chain_list = self.chain.to_list(deepcopy)
        token_list = self.token.to_list(deepcopy)
        atom_list = self.atom.to_list(deepcopy)
        bond_list = self.bond.to_list(deepcopy)

        data_list: list[Self] = []
        batch_size = self.batch_size
        for b in range(batch_size):
            data_list.append(
                self.__class__(
                    chain=chain_list[b],
                    token=token_list[b],
                    atom=atom_list[b],
                    bond=bond_list[b],
                    metadata=self.metadata[b] if self.metadata else None,
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
                f"  metadata: {self.metadata}\n"
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
                f"  metadata: {self.metadata}\n"
                f"  device: {device}\n"
                f")"
            )

    # === Padding functions for preparing model inputs === #
    def pad_to_multiple_of(self, multiple: int = 64) -> Self:
        """Pad all layouts to the multiple of 64 tokens for LocalAtomAttention and
        model efficiency."""
        assert multiple > 0, f"multiple must be a positive integer, but got {multiple}."
        assert multiple % 32 == 0, (
            f"multiple must be a multiple of 32 for LocalAttention, but got {multiple}."
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
    ) -> Self:
        """Pad all layouts to the specified maximum sizes."""
        max_tokens = max_tokens if max_tokens is not None else len(self.token)
        max_chains = max_chains if max_chains is not None else len(self.chain)
        max_atoms = max_atoms if max_atoms is not None else len(self.atom)
        max_bonds = max_bonds if max_bonds is not None else len(self.bond)

        return self.__class__(
            chain=self.chain.pad(max_chains),
            token=self.token.pad(max_tokens),
            atom=self.atom.pad(max_atoms),
            bond=self.bond.pad(max_bonds),
            metadata=self.metadata,
        )
