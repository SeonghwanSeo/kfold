import copy
import dataclasses
import io
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np

import kfold.constants as C
from kfold.data.layout import PlainLayout
from kfold.data.schema import Metadata
from kfold.utils.misc import check_array

__all__ = [
    "ChainArray",
    "TokenArray",
    "AtomArray",
    "BondArray",
    "TokenizedStructure",
]


def full_false(shape: tuple[int, ...]) -> np.ndarray:
    """Create an array of the given shape filled with False."""
    return np.zeros(shape, dtype=bool)


def full_minus_one(shape: tuple[int, ...]) -> np.ndarray:
    """Create an array of the given shape filled with -1."""
    return np.full(shape, -1, dtype=np.int64)


def full_nan(shape: tuple[int, ...]) -> np.ndarray:
    """Create an array of the given shape filled with NaN."""
    return np.full(shape, np.nan, dtype=np.float32)


# === Tokenized data structures === #
@dataclasses.dataclass(frozen=True, kw_only=True)
class ChainArray(PlainLayout[np.ndarray]):
    """Chain information.

    Shape: [Nchain, ...]

    Attributes
    ----------
    chain_type: np.ndarray (int)
        Chain types of shape [Nchain,], indicating the type of each chain.
    entity_id: np.ndarray (int)
        Entity IDs of shape [Nchain,], starting from 1.
    asym_id: np.ndarray (int)
        Asymmetric unit IDs of shape [Nchain,], starting from 1.
    sym_id: np.ndarray (int)
        Symmetry IDs of shape [Nchain,], starting from 1.
    num_residues: np.ndarray (int)
        Number of residues per chain of shape [Nchain,].
    num_tokens: np.ndarray (int)
        Number of tokens per chain of shape [Nchain,].
    num_atoms: np.ndarray (int)
        Number of atoms per chain of shape [Nchain,].

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
    residue_start: np.ndarray (int)
        Starting indices of residues for each chain.
    token_start: np.ndarray (int)
        Starting indices of tokens for each chain.
    """

    chain_type: np.ndarray  # [Nchain,], int
    entity_id: np.ndarray  # [Nchain,], int
    asym_id: np.ndarray  # [Nchain,], int
    sym_id: np.ndarray  # [Nchain,], int
    num_residues: np.ndarray  # [Nchain,], int
    num_tokens: np.ndarray  # [Nchain,], int
    num_atoms: np.ndarray  # [Nchain,], int

    # === Properties === #
    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.chain_type.shape  # [Nchain,]

    # === Batched layout === #
    def __post_init__(self):
        # shape: [Nchain,]
        shape = self.layout_shape

        check_array(self.chain_type, name="chain_type", dtype=np.integer, shape=shape)
        check_array(self.entity_id, name="entity_id", dtype=np.integer, shape=shape)
        check_array(self.asym_id, name="asym_id", dtype=np.integer, shape=shape)
        check_array(self.sym_id, name="sym_id", dtype=np.integer, shape=shape)
        check_array(self.num_residues, name="num_residues", dtype=np.integer, shape=shape)
        check_array(self.num_tokens, name="num_tokens", dtype=np.integer, shape=shape)
        check_array(self.num_atoms, name="num_atoms", dtype=np.integer, shape=shape)

    @cached_property
    def is_protein(self) -> np.ndarray:
        """Boolean tensor indicating whether the chain is protein."""
        return self.chain_type == C.chain.ChainType.PROTEIN.value

    @cached_property
    def is_dna(self) -> np.ndarray:
        """Boolean tensor indicating whether the chain is dna."""
        return self.chain_type == C.chain.ChainType.DNA.value

    @cached_property
    def is_rna(self) -> np.ndarray:
        """Boolean tensor indicating whether the chain is rna."""
        return self.chain_type == C.chain.ChainType.RNA.value

    @cached_property
    def is_ligand(self) -> np.ndarray:
        """Boolean tensor indicating whether the chain is ligand."""
        return self.chain_type == C.chain.ChainType.LIGAND.value

    @cached_property
    def residue_start(self) -> np.ndarray:
        """Starting indices of residues for each chain."""
        return np.cumsum(self.num_residues, dtype=np.int64) - self.num_residues

    @cached_property
    def token_start(self) -> np.ndarray:
        """Starting indices of tokens for each chain."""
        return np.cumsum(self.num_tokens, dtype=np.int64) - self.num_tokens

    @classmethod
    def get_empty(cls, num_chains: int) -> Self:
        """Get an empty ChainArray with the specified number of chains."""
        return cls(
            chain_type=full_minus_one((num_chains,)),
            entity_id=full_minus_one((num_chains,)),
            asym_id=full_minus_one((num_chains,)),
            sym_id=full_minus_one((num_chains,)),
            num_residues=full_minus_one((num_chains,)),
            num_tokens=full_minus_one((num_chains,)),
            num_atoms=full_minus_one((num_chains,)),
        )

    def sanity_check(self) -> None:
        """Perform sanity checks on the ChainArray."""
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if np.any(array < 0):
                raise ValueError(
                    f"ChainArray field '{field.name}' contains negative values."
                )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResidueArray(PlainLayout[np.ndarray]):
    """Residue information.

    Attributes
    ----------
    name: np.ndarray (str)
        Residue names of shape [L,], indicating the name of each residue.
    res_type: np.ndarray (int)
        Sequence tokens of shape [L,] (aatype, base, atom, ...)
    chain_type: np.ndarray (int)
        Chain types of shape [L,], indicating the type of each token.
    entity_id: np.ndarray (int)
        Entity IDs of shape [L,], starting from 1.
    asym_id: np.ndarray (int)
        Asymmetric unit IDs of shape [L,], starting from 1.
    sym_id: np.ndarray (int)
        Symmetry IDs of shape [L,], starting from 1.
    residue_index: np.ndarray (int)
        Residue indices of shape [L,], used for residue-level operations,
        starting from 1.
    num_tokens: np.ndarray (int)
        Number of tokens per residue of shape [L,].
    num_atoms: np.ndarray (int)
        Number of atoms per residue of shape [L,].
    is_standard: np.ndarray (bool)
        Boolean tensor of shape [L,], indicating whether the residue is standard.

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
    token_start: np.ndarray (int)
        Starting indices of tokens for each chain.
    """

    name: np.ndarray  # [L,], object(str)
    res_type: np.ndarray  # [L,], int
    chain_type: np.ndarray  # [L,], int
    entity_id: np.ndarray  # [L,], int
    asym_id: np.ndarray  # [L,], int, same to sequence_id
    sym_id: np.ndarray  # [L,], int
    residue_index: np.ndarray  # [L,], int
    num_tokens: np.ndarray  # [L,], int
    num_atoms: np.ndarray  # [L,], int
    is_standard: np.ndarray  # [L,], bool

    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.res_type.shape  # [Nresidue,]

    def __post_init__(self):
        shape = self.layout_shape
        check_array(self.name, name="name", dtype=np.dtype("<U6"), shape=shape)
        check_array(self.res_type, name="res_type", dtype=np.integer, shape=shape)
        check_array(self.chain_type, name="chain_type", dtype=np.integer, shape=shape)
        check_array(self.entity_id, name="entity_id", dtype=np.integer, shape=shape)
        check_array(self.asym_id, name="asym_id", dtype=np.integer, shape=shape)
        check_array(self.sym_id, name="sym_id", dtype=np.integer, shape=shape)
        check_array(
            self.residue_index, name="residue_index", dtype=np.integer, shape=shape
        )
        check_array(self.num_tokens, name="num_tokens", dtype=np.integer, shape=shape)
        check_array(self.num_atoms, name="num_atoms", dtype=np.integer, shape=shape)
        check_array(self.is_standard, name="is_standard", dtype=np.bool_, shape=shape)

    @cached_property
    def is_protein(self) -> np.ndarray:
        """Boolean tensor of shape [L,], indicating whether the token is protein."""
        return self.chain_type == C.chain.ChainType.PROTEIN.value

    @cached_property
    def is_dna(self) -> np.ndarray:
        """Boolean tensor of shape [L,], indicating whether the token is dna."""
        return self.chain_type == C.chain.ChainType.DNA.value

    @cached_property
    def is_rna(self) -> np.ndarray:
        """Boolean tensor of shape [L,], indicating whether the token is rna."""
        return self.chain_type == C.chain.ChainType.RNA.value

    @cached_property
    def is_ligand(self) -> np.ndarray:
        """Boolean tensor of shape [L,], indicating whether the token is ligand."""
        return self.chain_type == C.chain.ChainType.LIGAND.value

    @cached_property
    def token_start(self) -> np.ndarray:
        """Starting indices of tokens for each chain."""
        return np.cumsum(self.num_tokens, dtype=np.int64) - self.num_tokens

    # === Utility functions === #
    @cached_property
    def _get_residue_uid_to_index(self) -> dict[tuple[int, int], int]:
        """Get a mapping from (asym_id, residue_index) to global residue index."""
        uid_to_index: dict[tuple[int, int], int] = {
            (int(asym_id), int(res_idx)): res_i
            for res_i, (asym_id, res_idx) in enumerate(
                zip(self.asym_id, self.residue_index, strict=True)
            )
        }
        return uid_to_index

    def get_global_residue_idx(self, asym_id: int, residue_index: int) -> int:
        """Get the global residue idx from asym_id and residue_index.

        Parameters
        ----------
        asym_id: int
            Asymmetric unit ID of the chain which the residue belongs to. (1-based)
        residue_index: int
            Residue index within the chain. (1-based)

        Returns
        -------
        global_residue_index: int
            Global residue index in the structure. (0-based)
        """
        uid_to_index = self._get_residue_uid_to_index
        res_uid = (asym_id, residue_index)
        if res_uid not in uid_to_index:
            raise KeyError(
                f"Residue with asym_id={asym_id} and residue_index={residue_index} "
                f"not found."
            )
        return uid_to_index[res_uid]

    @classmethod
    def get_empty(cls, num_residues: int) -> Self:
        """Get an empty ResidueArray with the specified number of residues."""
        return cls(
            name=np.array([""] * num_residues, dtype=np.dtype("<U6")),
            res_type=full_minus_one((num_residues,)),
            chain_type=full_minus_one((num_residues,)),
            entity_id=full_minus_one((num_residues,)),
            asym_id=full_minus_one((num_residues,)),
            sym_id=full_minus_one((num_residues,)),
            residue_index=full_minus_one((num_residues,)),
            num_tokens=full_minus_one((num_residues,)),
            num_atoms=full_minus_one((num_residues,)),
            is_standard=full_false((num_residues,)),
        )

    def sanity_check(self) -> None:
        """Perform sanity checks on the ResidueArray."""
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if np.any(array < 0) and field.name not in ["name", "is_standard"]:
                raise ValueError(
                    f"ResidueArray field '{field.name}' contains negative values."
                )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenArray(PlainLayout[np.ndarray]):
    """Token information.

    Attributes
    ----------
    res_type: np.ndarray (int)
        Sequence tokens of shape [L,] (aatype, base, atom, ...)
    chain_type: np.ndarray (int)
        Chain types of shape [L,], indicating the type of each token.
    entity_id: np.ndarray (int)
        Entity IDs of shape [L,], starting from 1.
    asym_id: np.ndarray (int)
        Asymmetric unit IDs of shape [L,], starting from 1.
    sym_id: np.ndarray (int)
        Symmetry IDs of shape [L,], starting from 1.
    token_index: np.ndarray (int)
        Token indices of shape [L,], used for token-level operations,
    residue_index: np.ndarray (int)
        Residue indices of shape [L,], used for residue-level operations,
        starting from 1.
    num_atoms: np.ndarray (int)
        Number of atoms per token of shape [L,].
    disto_index: np.ndarray (int)
        Distogram atom index of shape [L,], used for distogram calculations.
    center_index: np.ndarray (int)
        Center atom index of shape [L,], used for center calculations.
    is_standard: np.ndarray (bool)
        Boolean tensor of shape [L,], indicating whether the token is standard.

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

    res_type: np.ndarray  # [L,], int
    chain_type: np.ndarray  # [L,], int
    entity_id: np.ndarray  # [L,], int
    asym_id: np.ndarray  # [L,], int, same to sequence_id
    sym_id: np.ndarray  # [L,], int
    token_index: np.ndarray  # [L,], int
    residue_index: np.ndarray  # [L,], int
    num_atoms: np.ndarray  # [L,], int
    disto_index: np.ndarray  # [L,], int
    center_index: np.ndarray  # [L,], int
    is_standard: np.ndarray  # [L,], bool

    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.res_type.shape  # [Ntoken,]

    def __post_init__(self):
        shape = self.layout_shape
        check_array(self.res_type, name="res_type", dtype=np.integer, shape=shape)
        check_array(self.chain_type, name="chain_type", dtype=np.integer, shape=shape)
        check_array(self.entity_id, name="entity_id", dtype=np.integer, shape=shape)
        check_array(self.asym_id, name="asym_id", dtype=np.integer, shape=shape)
        check_array(self.sym_id, name="sym_id", dtype=np.integer, shape=shape)
        check_array(
            self.residue_index, name="residue_index", dtype=np.integer, shape=shape
        )
        check_array(self.num_atoms, name="num_atoms", dtype=np.integer, shape=shape)
        check_array(self.disto_index, name="disto_index", dtype=np.integer, shape=shape)
        check_array(self.center_index, name="center_index", dtype=np.integer, shape=shape)
        check_array(self.is_standard, name="is_standard", dtype=np.bool_, shape=shape)

    @cached_property
    def is_protein(self) -> np.ndarray:
        """Boolean tensor of shape [L,], indicating whether the token is protein."""
        return self.chain_type == C.chain.ChainType.PROTEIN.value

    @cached_property
    def is_dna(self) -> np.ndarray:
        """Boolean tensor of shape [L,], indicating whether the token is dna."""
        return self.chain_type == C.chain.ChainType.DNA.value

    @cached_property
    def is_rna(self) -> np.ndarray:
        """Boolean tensor of shape [L,], indicating whether the token is rna."""
        return self.chain_type == C.chain.ChainType.RNA.value

    @cached_property
    def is_ligand(self) -> np.ndarray:
        """Boolean tensor of shape [L,], indicating whether the token is ligand."""
        return self.chain_type == C.chain.ChainType.LIGAND.value

    @classmethod
    def get_empty(cls, num_tokens: int) -> Self:
        """Get an empty TokenArray with the specified number of tokens."""
        return cls(
            res_type=full_minus_one((num_tokens,)),
            chain_type=full_minus_one((num_tokens,)),
            entity_id=full_minus_one((num_tokens,)),
            asym_id=full_minus_one((num_tokens,)),
            sym_id=full_minus_one((num_tokens,)),
            token_index=full_minus_one((num_tokens,)),
            residue_index=full_minus_one((num_tokens,)),
            num_atoms=full_minus_one((num_tokens,)),
            disto_index=full_minus_one((num_tokens,)),
            center_index=full_minus_one((num_tokens,)),
            is_standard=full_false((num_tokens,)),
        )

    def sanity_check(self) -> None:
        """Perform sanity checks on the ResidueArray."""
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if np.any(array < 0) and field.name not in ["is_standard"]:
                raise ValueError(
                    f"TokenArray field '{field.name}' contains negative values."
                )


@dataclasses.dataclass(frozen=True, kw_only=True)
class AtomArray(PlainLayout[np.ndarray]):
    """Atom information.

    Shape: [Ntoken, 24, ...]

    Attributes
    ----------
    ref_atom_name_chars: np.ndarray (int)
        Encoded atom name of shape [Ntoken, 24, 4].
    ref_element: np.ndarray (int)
        One-hot encoded atomic numbers of shape [Ntoken, 24,].
    ref_charge: np.ndarray (float)
        Formal charges of shape [Ntoken, 24,].
    ref_pos: np.ndarray (float32)
        Reference coordinates of shape [Ntoken, 24, 3].
        Generated from ETKDG or ccd
        (TODO (seonghwan): I think we can replace this to apo_coords)
    ref_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24] indicating valid reference atoms.
    coords: np.ndarray (float32)
        Holo (bound) state coordinates of shape [Ntoken, 24, 3],
        This is used as the ground truth for training, and may be set to 0
        for inference.
    apo_coords: np.ndarray (float32)
        Apo (unbound) state coordinates of shape [Ntoken, 24, 3],
    resolved_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24,] indicating atoms to be resolved.
    apo_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24] indicating valid apo atoms.
    apo_plddt: np.ndarray (float32)
        Predicted LDDT scores of shape [Ntoken, 24,].
    pad_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24,] indicating atoms to be resolved.
    """

    ref_uid: np.ndarray  # [Ntoken, 24], int
    ref_atom_name_chars: np.ndarray  # [Ntoken, 24, 4], int
    ref_element: np.ndarray  # [Ntoken, 24], int
    ref_charge: np.ndarray  # [Ntoken, 24,], float
    ref_pos: np.ndarray  # [Ntoken, 24, 3], float32
    ref_mask: np.ndarray  # [Ntoken, 24], bool
    coords: np.ndarray  # [Ntoken, 24, 3], float32
    resolved_mask: np.ndarray  # [Ntoken, 24], bool
    apo_coords: np.ndarray  # [Ntoken, 24, 3], float32
    apo_mask: np.ndarray  # [Ntoken, 24], bool
    apo_plddt: np.ndarray  # [Ntoken, 24], float32
    pad_mask: np.ndarray  # [Ntoken, 24], bool

    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.ref_element.shape

    def __post_init__(self):
        shape = self.layout_shape
        check_array(self.ref_uid, name="ref_uid", dtype=np.integer, shape=shape)
        check_array(
            self.ref_atom_name_chars,
            name="ref_atom_name_chars",
            dtype=np.integer,
            shape=(*shape, 4),
        )
        check_array(self.ref_element, name="ref_element", dtype=np.integer, shape=shape)
        check_array(self.ref_charge, name="ref_charge", dtype=np.floating, shape=shape)
        check_array(self.ref_pos, name="ref_pos", dtype=np.floating, shape=(*shape, 3))
        check_array(self.ref_mask, name="ref_mask", dtype=np.bool_, shape=shape)
        check_array(self.coords, name="coords", dtype=np.floating, shape=(*shape, 3))
        check_array(
            self.apo_coords, name="apo_coords", dtype=np.floating, shape=(*shape, 3)
        )
        check_array(self.resolved_mask, name="resolved_mask", dtype=np.bool_, shape=shape)
        check_array(self.apo_mask, name="apo_mask", dtype=np.bool_, shape=shape)
        check_array(self.pad_mask, name="pad_mask", dtype=np.bool_, shape=shape)

    @classmethod
    def get_empty(cls, num_tokens: int) -> Self:
        """Get an empty AtomArray with the specified number of tokens."""
        num_atoms = C.MAX_NUM_ATOMS_PER_TOKEN
        shape = (num_tokens, num_atoms)
        return cls(
            ref_uid=full_minus_one(shape),
            ref_atom_name_chars=full_minus_one((*shape, 4)),
            ref_element=full_minus_one(shape),
            ref_charge=full_nan(shape),
            ref_pos=full_nan((*shape, 3)),
            coords=full_nan((*shape, 3)),
            apo_coords=full_nan((*shape, 3)),
            apo_plddt=full_nan(shape),
            ref_mask=full_false(shape),
            resolved_mask=full_false(shape),
            apo_mask=full_false(shape),
            pad_mask=full_false(shape),
        )

    def sanity_check(self) -> None:
        """Perform sanity checks on the ResidueArray."""
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if np.any(array < 0) and field.name not in [
                "ref_charge",
                "ref_pos",
                "coords",
                "apo_coords",
                "apo_plddt",
                "ref_mask",
                "resolved_mask",
                "apo_mask",
                "pad_mask",
            ]:
                raise ValueError(
                    f"AtomArray field '{field.name}' contains negative values."
                )
            elif field.name == "ref_charge" and np.any(np.isnan(array)):
                raise ValueError(f"AtomArray field '{field.name}' contains NaN values.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class BondArray(PlainLayout[np.ndarray]):
    """Bond information.

    Shape: [Nbond, ...]

    Attributes
    ----------
    asym_id: np.ndarray
        Chain asym indices of the connecting atoms in the bond of shape [Nbond, 2].
    token_index: np.ndarray
        Token indices of the connecting atoms in the bond of shape [Nbond, 2].
    atom_index: np.ndarray
        Atom indices of the connecting atoms in the bond of shape [Nbond, 2].
    bond_type: np.ndarray
        Bond types of shape [Nbond,], indicating the type of each bond.
    """

    asym_id: np.ndarray  # [Nbond, 2], int
    token_index: np.ndarray  # [Nbond, 2], int
    atom_index: np.ndarray  # [Nbond, 2], int
    bond_type: np.ndarray  # [Nbond,], int

    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.bond_type.shape

    def __post_init__(self):
        shape = self.layout_shape
        check_array(self.asym_id, name="asym_id", dtype=np.integer, shape=(*shape, 2))
        check_array(
            self.token_index, name="token_index", dtype=np.integer, shape=(*shape, 2)
        )
        check_array(
            self.atom_index, name="atom_index", dtype=np.integer, shape=(*shape, 2)
        )
        check_array(self.bond_type, name="bond_type", dtype=np.integer, shape=shape)

    @classmethod
    def get_empty(cls, num_bonds: int) -> Self:
        """Get an empty BondArray with the specified number of bonds."""
        return cls(
            asym_id=full_minus_one((num_bonds, 2)),
            token_index=full_minus_one((num_bonds, 2)),
            atom_index=full_minus_one((num_bonds, 2)),
            bond_type=full_minus_one((num_bonds,)),
        )

    def sanity_check(self) -> None:
        """Perform sanity checks on the BondArray."""
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if np.any(array < 0):
                raise ValueError(
                    f"BondArray field '{field.name}' contains negative values."
                )


@dataclasses.dataclass(kw_only=True)
class TokenizedStructure:
    """Tokenized representation of a molecular structure.

    Attributes
    ----------
    chain: ChainArray
        Chain information.
    token: TokenArray
        Token information.
    atom: AtomArray
        Atom information.
    bond: BondArray
        Bond information.
    metadata: Metadata
        Metadata information.
    """

    chain: ChainArray
    residue: ResidueArray
    token: TokenArray
    atom: AtomArray
    bond: BondArray
    metadata: Metadata | None = None

    @property
    def num_chains(self) -> int:
        """Number of chains in the structure."""
        return len(self.chain)

    @property
    def num_residues(self) -> int:
        """Number of residues in the structure."""
        return len(self.residue)

    @property
    def num_tokens(self) -> int:
        """Number of tokens in the structure."""
        return len(self.token)

    @cached_property
    def num_atoms(self) -> int:
        """Number of atoms in the structure."""
        return int(self.token.num_atoms.sum())

    @property
    def num_bonds(self) -> int:
        """Number of bonds in the structure."""
        return len(self.bond)

    def __repr__(self) -> str:
        """FoldingInput summary representation."""
        # Summary statistics
        num_chains = self.num_chains
        num_residues = self.num_residues
        num_tokens = self.num_tokens
        num_bonds = len(self.bond)
        return (
            f"TokenizedStructure(\n"
            f"  num_chains: {num_chains}\n"
            f"  num_residues: {num_residues}\n"
            f"  num_tokens: {num_tokens}\n"
            f"  num_bonds: {num_bonds}\n"
            f")"
        )

    @classmethod
    def get_empty(
        cls,
        num_chains: int,
        num_residues: int,
        num_tokens: int,
        num_bonds: int,
        metadata: Metadata | None = None,
    ) -> Self:
        """Get an empty TokenizedStructure with the specified sizes."""
        return cls(
            chain=ChainArray.get_empty(num_chains),
            residue=ResidueArray.get_empty(num_residues),
            token=TokenArray.get_empty(num_tokens),
            atom=AtomArray.get_empty(num_tokens),
            bond=BondArray.get_empty(num_bonds),
            metadata=metadata,
        )

    # === PDB/MMCIF writing === #
    def write(
        self,
        path: Path | str,
        conformer_id: int = 0,
        is_predicted: bool = True,
        save_apo: bool = False,
    ) -> None:
        """Write to PDB or MMCIF file based on the file extension."""
        from kfold.utils.writer import KFoldWriter

        KFoldWriter.write(self, path, conformer_id, is_predicted, save_apo)

    def to_pdb(
        self,
        path: Path | str,
        conformer_id: int = 0,
        is_predicted: bool = True,
        save_apo: bool = False,
    ) -> None:
        """Write to PDB file."""
        from kfold.utils.writer import KFoldWriter

        KFoldWriter.write_pdb(self, path, conformer_id, is_predicted, save_apo)

    def to_mmcif(
        self,
        path: Path | str,
        conformer_id: int = 0,
        is_predicted: bool = True,
        save_apo: bool = False,
    ) -> None:
        """Write to MMCIF file."""
        from kfold.utils.writer import KFoldWriter

        KFoldWriter.write_mmcif(self, path, conformer_id, is_predicted, save_apo)

    # === Numpy serialization for model training === #
    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Convert to a flat dictionary for NPZ storage.

        Returns a dictionary where tensor fields are converted to numpy arrays
        with hierarchical keys like 'chain.asym_id', 'token.token_type', etc.
        """
        chain_dict = self.chain.to_dict()
        residue_dict = self.residue.to_dict()
        token_dict = self.token.to_dict()
        atom_dict = self.atom.to_dict()
        bond_dict = self.bond.to_dict()

        result = {
            **{f"chain.{key}": value for key, value in chain_dict.items()},
            **{f"residue.{key}": value for key, value in residue_dict.items()},
            **{f"token.{key}": value for key, value in token_dict.items()},
            **{f"atom.{key}": value for key, value in atom_dict.items()},
            **{f"bond.{key}": value for key, value in bond_dict.items()},
        }
        return result

    @classmethod
    def from_npz_dict(cls, data: dict[str, np.ndarray]) -> Self:
        """Reconstruct from NPZ dictionary."""
        reconstructed = {}
        for prefix, struct_cls in [
            ("chain.", ChainArray),
            ("residue.", ResidueArray),
            ("token.", TokenArray),
            ("atom.", AtomArray),
            ("bond.", BondArray),
        ]:
            struct_data = {
                key[len(prefix) :]: value
                for key, value in data.items()
                if key.startswith(prefix)
            }
            reconstructed[prefix[:-1]] = struct_cls(**struct_data)
        return cls(**reconstructed)

    def dump_npz(self, path: Path | str) -> None:
        """Save to compressed NPZ file."""
        path = Path(path)
        data = self.to_npz_dict()
        np.savez_compressed(path, **data)

    def save_npz(self, path: Path | str) -> None:
        """Save to compressed NPZ file."""
        self.dump_npz(path)

    @classmethod
    def load_npz(cls, path: Path | str | io.BytesIO) -> Self:
        """Load from NPZ file."""
        with np.load(path) as data:
            return cls.from_npz_dict(dict(data))

    # === Utility functions === #
    def to(self, *args, **kwargs) -> Self:
        """
        No-op for device/dtype movement.

        This method is present for API compatibility, but does nothing because
        this structure only contains numpy arrays, which do not support device
        or dtype movement like PyTorch tensors.
        """
        return self

    def copy(self, deepcopy: bool = False) -> Self:
        """Create a copy of the structure."""
        if deepcopy:
            return copy.deepcopy(self)
        else:
            return self.__class__(
                chain=self.chain,
                residue=self.residue,
                token=self.token,
                atom=self.atom,
                bond=self.bond,
                metadata=self.metadata,
            )

    def copy_with(self, **kwargs) -> Self:
        """Create a copy of the structure with updated fields.

        Parameters
        ----------
        **kwargs
            Fields to update. Can include 'chain', 'token', 'atom', 'bond', 'metadata'.

        Returns
        -------
        copied_structure: TokenizedStructure
            Copied tokenized structure with updated fields.
        """
        return dataclasses.replace(self, **kwargs)

    def crop(self, token_indices: np.ndarray) -> Self:
        """Crop the structure to the specified token indices.

        Parameters
        ----------
        token_indices: np.ndarray (int)
            Token indices to keep of shape [K,], where K is the number of tokens
            to keep.

        Returns
        -------
        cropped_structure: TokenizedStructure
            Cropped tokenized structure.
        """
        token_indices = np.sort(np.unique(token_indices))

        cropped_token = self.token[token_indices]
        cropped_atom = self.atom[token_indices]

        token_bonds = self.bond.token_index
        bond_mask = np.isin(token_bonds, token_indices).all(axis=1)
        cropped_bond = self.bond[bond_mask]  # type: ignore

        # Remove excluding residues
        # NOTE: (SeonghwanSeo) Since residue_index is defined per chain, we
        # use (2**32 * asym_id + res_idx) to uniquely identify residues.
        assert cropped_token.residue_index.max() < 2**31, (
            "residue_index should be less than 2**31"
        )
        assert cropped_token.asym_id.max() < 2**31, "asym_id should be less than 2**31"
        residue_uids = (
            self.residue.asym_id.astype(np.int64) << 32
        ) + self.residue.residue_index
        cropped_token_residue_uids = (
            cropped_token.asym_id.astype(np.int64) << 32
        ) + cropped_token.residue_index
        residue_mask = np.isin(residue_uids, cropped_token_residue_uids)
        cropped_residue = self.residue[residue_mask]

        # safe update
        cropped_residue = cropped_residue.copy(deepcopy=True)
        cropped_residue_uids = residue_uids[residue_mask]
        for cidx in range(len(cropped_residue)):
            res_uid = cropped_residue_uids[cidx]
            # Compute num_tokens, num_atoms
            token_mask = cropped_token_residue_uids == res_uid
            num_tokens = np.sum(token_mask).item()
            num_atoms = np.sum(cropped_token.num_atoms[token_mask]).item()
            cropped_residue.num_tokens[cidx] = num_tokens
            cropped_residue.num_atoms[cidx] = num_atoms

        # Remove excluding chains
        token_asym_ids = np.unique(cropped_token.asym_id)
        chain_mask = np.isin(self.chain.asym_id, token_asym_ids)
        cropped_chain = self.chain[chain_mask]  # type: ignore

        # safe update
        cropped_chain = cropped_chain.copy(deepcopy=True)
        for cidx in range(len(cropped_chain)):
            asym_id = cropped_chain.asym_id[cidx]

            # Compute num_residues, num_tokens, num_atoms
            residue_mask = cropped_residue.asym_id == asym_id
            num_residues = np.sum(residue_mask).item()

            token_mask = cropped_token.asym_id == asym_id
            num_tokens = np.sum(token_mask).item()
            num_atoms = np.sum(cropped_token.num_atoms[token_mask]).item()

            cropped_chain.num_tokens[cidx] = num_tokens
            cropped_chain.num_residues[cidx] = num_residues
            cropped_chain.num_atoms[cidx] = num_atoms

        return self.__class__(
            chain=cropped_chain,
            residue=cropped_residue,
            token=cropped_token,
            atom=cropped_atom,
            bond=cropped_bond,
            metadata=self.metadata,
        )

    def reassign_token_indices(self) -> Self:
        """Reassign token indices to be consecutive from 0 to Ntoken-1.

        Returns
        -------
        new_struct: TokenizedStructure
            Structure with reassigned token indices.
        """
        Ntoken = self.num_tokens
        old_token_indices = self.token.token_index
        new_token_indices = np.arange(Ntoken, dtype=old_token_indices.dtype)

        # Update token structure
        new_token = self.token.copy_with(token_index=new_token_indices)

        # Update token indices in bond
        bond = self.bond
        token_index_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(old_token_indices)
        }
        old_bond_token_indices = bond.token_index
        new_bond_token_indices = np.array(
            [
                [token_index_mapping[int(idx)] for idx in bond_pair]
                for bond_pair in old_bond_token_indices
            ],
            dtype=old_bond_token_indices.dtype,
        ).reshape(-1, 2)
        new_bond = bond.copy_with(token_index=new_bond_token_indices)

        # Create new structure
        new_struct = self.copy_with(token=new_token, bond=new_bond)
        return new_struct

    def replace_atom_coords(
        self,
        atom_coords: np.ndarray,
        is_apo: bool = False,
    ) -> Self:
        """Replace coordinates in structure

        Parameters
        ----------
        atom_coords: np.ndarray
            Shape: [Nsample, Natom, 3] or [Nsample, Ntoken, 24, 3]

        Returns
        -------
        new_struct: TokenizedStructure
            Structure with replaced coordinates

        """
        Nsample = atom_coords.shape[0]
        num_tokens = self.num_tokens
        num_atoms = self.num_atoms
        max_atoms_per_token = 24

        if atom_coords.ndim == 3:
            assert num_atoms <= atom_coords.shape[1], (
                f"Coordinate atom count ({atom_coords.shape[1]}) should be same or "
                f"larger than total atoms ({num_atoms})"
            )
            # Create new coords array [num_tokens, 24, Nsample, 3]
            new_coords = np.zeros(
                (num_tokens, max_atoms_per_token, Nsample, 3), dtype=atom_coords.dtype
            )
            coords_to_assign = atom_coords[:, :num_atoms].transpose(1, 0, 2)
            new_coords[self.atom.pad_mask] = coords_to_assign
        else:
            assert atom_coords.shape[1] <= num_tokens, (
                f"Coordinate token count ({atom_coords.shape[1]}) should be same or "
                f"smaller than total tokens ({num_tokens})"
            )
            assert atom_coords.shape[2] == max_atoms_per_token, (
                f"Coordinate atom per token count ({atom_coords.shape[2]}) should be "
                f"same to max atoms per token (24)"
            )
            # [Nsample, Ntoken_with_pad, 24, 3] -> [Ntoken, 24, Nsample, 3]
            new_coords = np.ascontiguousarray(
                atom_coords[:, :num_tokens].transpose(1, 2, 0, 3)
            )

        # Update structure
        atom_struct = self.atom
        if is_apo:
            new_atom_struct = atom_struct.copy_with(apo_coords=new_coords)
        else:
            new_atom_struct = atom_struct.copy_with(coords=new_coords)
        new_struct = self.copy_with(atom=new_atom_struct)
        return new_struct
