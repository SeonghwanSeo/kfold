import copy
import dataclasses
from functools import cached_property
from typing import Self

import numpy as np

import kfold.constants as C
from kfold.data.layout import PlainLayout
from kfold.utils.misc import check_array

__all__ = ["TokenizedStructure"]


def full_false(shape: tuple[int, ...]) -> np.ndarray:
    """Create an array of the given shape filled with False."""
    return np.zeros(shape, dtype=bool)


def full_minus_one(shape: tuple[int, ...]) -> np.ndarray:
    """Create an array of the given shape filled with -1."""
    return np.full(shape, -1, dtype=np.int64)


def full_zero(shape: tuple[int, ...]) -> np.ndarray:
    """Create an array of the given shape filled with zero."""
    return np.zeros(shape, dtype=np.float32)


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

    def validate(self) -> None:
        """Perform sanity checks on the ChainArray."""
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if np.any(array < 0):
                raise ValueError(
                    f"ChainArray field '{field.name}' contains negative values."
                )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenArray(PlainLayout[np.ndarray]):
    """Token information.

    Attributes
    ----------
    chain_type: np.ndarray (int)
        Chain types of shape [L,], indicating the type of each token.
    entity_id: np.ndarray (int)
        Entity IDs of shape [L,], starting from 1.
    asym_id: np.ndarray (int)
        Asymmetric unit IDs of shape [L,], starting from 1.
    sym_id: np.ndarray (int)
        Symmetry IDs of shape [L,], starting from 1.
    res_type: np.ndarray (int)
        Sequence tokens of shape [L,] (aatype, base, atom, ...)
    num_atoms: np.ndarray (int)
        Number of atoms per token of shape [L,].
    is_standard: np.ndarray (bool)
        Boolean tensor of shape [L,], indicating whether the token is standard.
    token_index: np.ndarray (int)
        Token indices of shape [L,], used for token-level operations,
        starting from 0.
    residue_index: np.ndarray (int)
        Residue indices of shape [L,], used for residue-level operations,
        starting from 1.
    seq_token_index: np.ndarray (int)
        Sequence token indices of shape [L,], used for sequence embedding,
        starting from 0.
    center_index: np.ndarray (int)
        Center atom index of shape [L,], used for center calculations.
    disto_index: np.ndarray (int)
        Distogram atom index of shape [L,], used for distogram calculations.

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

    chain_type: np.ndarray  # [L,], int
    entity_id: np.ndarray  # [L,], int
    asym_id: np.ndarray  # [L,], int, same to sequence_id
    sym_id: np.ndarray  # [L,], int
    res_type: np.ndarray  # [L,], int
    num_atoms: np.ndarray  # [L,], int
    is_standard: np.ndarray  # [L,], bool
    token_index: np.ndarray  # [L,], int
    residue_index: np.ndarray  # [L,], int
    seq_token_index: np.ndarray  # [L,], int
    center_index: np.ndarray  # [L,], int
    disto_index: np.ndarray  # [L,], int

    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.res_type.shape  # [Ntoken,]

    def __post_init__(self):
        shape = self.layout_shape
        check_array(self.chain_type, name="chain_type", dtype=np.integer, shape=shape)
        check_array(self.entity_id, name="entity_id", dtype=np.integer, shape=shape)
        check_array(self.asym_id, name="asym_id", dtype=np.integer, shape=shape)
        check_array(self.sym_id, name="sym_id", dtype=np.integer, shape=shape)
        check_array(self.res_type, name="res_type", dtype=np.integer, shape=shape)
        check_array(self.is_standard, name="is_standard", dtype=np.bool_, shape=shape)
        check_array(self.num_atoms, name="num_atoms", dtype=np.integer, shape=shape)
        check_array(self.token_index, name="token_index", dtype=np.integer, shape=shape)
        check_array(
            self.residue_index, name="residue_index", dtype=np.integer, shape=shape
        )
        check_array(self.center_index, name="center_index", dtype=np.integer, shape=shape)
        check_array(self.disto_index, name="disto_index", dtype=np.integer, shape=shape)

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
            token_index=full_minus_one((num_tokens,)),
            residue_index=full_minus_one((num_tokens,)),
            seq_token_index=full_minus_one((num_tokens,)),
            res_type=full_minus_one((num_tokens,)),
            chain_type=full_minus_one((num_tokens,)),
            entity_id=full_minus_one((num_tokens,)),
            asym_id=full_minus_one((num_tokens,)),
            sym_id=full_minus_one((num_tokens,)),
            num_atoms=full_minus_one((num_tokens,)),
            center_index=full_minus_one((num_tokens,)),
            disto_index=full_minus_one((num_tokens,)),
            is_standard=full_false((num_tokens,)),
        )

    def validate(self) -> None:
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
        Atomic numbers of shape [Ntoken, 24].
    ref_charge: np.ndarray (float)
        Formal charges of shape [Ntoken, 24].
    ref_pos: np.ndarray (float32)
        Reference coordinates of shape [Ntoken, 24, 3].
        Generated from ETKDG or ccd
    ref_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24] indicating valid reference atoms.
    apo_coords: np.ndarray (float32)
        Apo (unbound) state coordinates of shape [Ntoken, 24, 3],
    prior_coords: np.ndarray (float32)
        Prior state coordinates of shape [Ntoken, 24, Nprior, 3],
        NOTE All values must be finite, NaN is not allowed.
    label_coords: np.ndarray (float32)
        Holo (bound) state coordinates of shape [Ntoken, 24, 3],
        This is used as the ground truth for training, and may be set to 0
        for inference.
    resolved_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24,] indicating atoms to be resolved.
    apo_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24] indicating valid apo atoms.
    pad_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24,] indicating atoms to be resolved.
    """

    ref_atom_name_chars: np.ndarray  # [Ntoken, 24, 4], int
    ref_element: np.ndarray  # [Ntoken, 24], int
    ref_charge: np.ndarray  # [Ntoken, 24], float
    ref_pos: np.ndarray  # [Ntoken, 24, 3], float32
    ref_mask: np.ndarray  # [Ntoken, 24], bool
    apo_coords: np.ndarray  # [Ntoken, 24, 3], float32
    prior_coords: np.ndarray  # [Ntoken, 24, Nprior, 3], float32
    label_coords: np.ndarray  # [Ntoken, 24, 3], float32
    resolved_mask: np.ndarray  # [Ntoken, 24], bool
    apo_mask: np.ndarray  # [Ntoken, 24], bool
    pad_mask: np.ndarray  # [Ntoken, 24], bool

    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.ref_element.shape

    def __post_init__(self):
        shape = self.layout_shape
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
        check_array(
            self.apo_coords, name="apo_coords", dtype=np.floating, shape=(*shape, 3)
        )
        check_array(
            self.prior_coords,
            name="prior_coords",
            dtype=np.floating,
            shape=(*shape, -1, 3),
        )
        check_array(
            self.label_coords, name="label_coords", dtype=np.floating, shape=(*shape, 3)
        )
        check_array(self.apo_mask, name="apo_mask", dtype=np.bool_, shape=shape)
        check_array(self.pad_mask, name="pad_mask", dtype=np.bool_, shape=shape)
        check_array(self.resolved_mask, name="resolved_mask", dtype=np.bool_, shape=shape)

    @classmethod
    def get_empty(
        cls,
        num_tokens: int,
        num_priors: int = 0,
    ) -> Self:
        """Get an empty AtomArray with the specified number of tokens."""
        num_atoms = C.MAX_NUM_ATOMS_PER_TOKEN
        shape = (num_tokens, num_atoms)
        return cls(
            ref_atom_name_chars=full_minus_one((*shape, 4)),
            ref_element=full_minus_one(shape),
            ref_charge=full_nan(shape),
            ref_pos=full_nan((*shape, 3)),
            apo_coords=full_nan((*shape, 3)),
            prior_coords=full_nan((*shape, num_priors, 3)),
            ref_mask=full_false(shape),
            apo_mask=full_false(shape),
            pad_mask=full_false(shape),
            label_coords=full_nan((*shape, 3)),
            resolved_mask=full_false(shape),
        )

    def validate(self) -> None:
        """Perform sanity checks on the ResidueArray."""
        pad_mask = self.pad_mask
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if field.name not in [
                "ref_charge",
                "ref_pos",
                "apo_coords",
                "prior_coords",
                "label_coords",
                "ref_mask",
                "apo_mask",
                "pad_mask",
                "resolved_mask",
            ]:
                if np.any(array[pad_mask] < 0):
                    raise ValueError(
                        f"AtomArray field '{field.name}' contains negative values."
                    )
                if not np.any(array[~pad_mask] < 0):
                    raise ValueError(
                        f"AtomArray field '{field.name}' contains no negative values "
                        f"in the padded region."
                    )
            elif field.name in (
                "ref_charge",
                "prior_coords",
            ):
                if not np.all(np.isfinite(array[pad_mask])):
                    raise ValueError(
                        f"AtomArray field '{field.name}' contains NaN or Inf values."
                    )
                if np.any(np.isfinite(array[~pad_mask])):
                    raise ValueError(
                        f"AtomArray field '{field.name}' contains finite values in "
                        f"the padded region."
                    )


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

    def validate(self) -> None:
        """Perform sanity checks on the BondArray."""
        for field in dataclasses.fields(self):
            array = getattr(self, field.name)
            if np.any(array < 0):
                raise ValueError(
                    f"BondArray field '{field.name}' contains negative values."
                )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SequenceArray(PlainLayout[np.ndarray]):
    """Full sequence information for sequence embedding.

    Attributes
    ----------
    chain_type: np.ndarray (int)
        Chain types of shape [L,], indicating the type of each chain.
    entity_id: np.ndarray (int)
        Entity IDs of shape [L,], starting from 1.
    seq_token_id: np.ndarray (int)
        Sequence tokens of shape [L,] (aatype, base, atom, ...)
        NOTE: this may differ from the res_type in TokenArray,
        since vocab is different for sequence embedding and co-folding.
    bb_struct_token_id: np.ndarray (int)
        Backbone structure tokens of shape [L,], used for backbone structure embedding.
    fa_struct_token_id: np.ndarray (int)
        Full-atom structure tokens of shape [L,], used for full-atom structure embedding
    pos_id: np.ndarray (int)
        Position indices of shape [L,], starting from 0.
    mlm_mask: np.ndarray (bool)
        Boolean tensor of shape [L,], indicating whether to mask the token for
        sequence embedding (see ESMFold stochastic sampling strategy).

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

    chain_type: np.ndarray  # [L,], int
    entity_id: np.ndarray  # [L,], int
    seq_token_id: np.ndarray  # [L,], int
    bb_struct_token_id: np.ndarray  # [L,], int
    fa_struct_token_id: np.ndarray  # [L,], int
    pos_id: np.ndarray  # [L,], int
    mlm_mask: np.ndarray  # [L,], bool

    @cached_property
    def layout_shape(self) -> tuple[int, ...]:
        return self.seq_token_id.shape  # [L,]

    def __post_init__(self):
        shape = self.layout_shape
        check_array(self.chain_type, name="chain_type", dtype=np.integer, shape=shape)
        check_array(self.entity_id, name="entity_id", dtype=np.integer, shape=shape)
        check_array(self.seq_token_id, name="seq_token_id", dtype=np.integer, shape=shape)
        check_array(
            self.bb_struct_token_id,
            name="bb_struct_token_id",
            dtype=np.integer,
            shape=shape,
        )
        check_array(
            self.fa_struct_token_id,
            name="fa_struct_token_id",
            dtype=np.integer,
            shape=shape,
        )
        check_array(self.pos_id, name="pos_id", dtype=np.integer, shape=shape)
        check_array(self.mlm_mask, name="mlm_mask", dtype=np.bool_, shape=shape)

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
    def get_empty(cls, sequence_length: int) -> Self:
        """Get an empty SequenceArray with the specified sequence length."""
        return cls(
            chain_type=full_minus_one((sequence_length,)),
            entity_id=full_minus_one((sequence_length,)),
            seq_token_id=full_minus_one((sequence_length,)),
            bb_struct_token_id=full_minus_one((sequence_length,)),
            fa_struct_token_id=full_minus_one((sequence_length,)),
            pos_id=full_minus_one((sequence_length,)),
            mlm_mask=full_false((sequence_length,)),
        )

    def validate(self) -> None:
        """Perform sanity checks on the ResidueArray."""
        for field in dataclasses.fields(self):
            if field.name in ["mlm_mask"]:
                # This field is boolean, so negative values are not applicable.
                continue
            elif field.name in ["fa_struct_token_id", "bb_struct_token_id"]:
                # These fields can be -1 for missing residues.
                continue
            array = getattr(self, field.name)
            if np.any(array < 0):
                raise ValueError(
                    f"TokenArray field '{field.name}' contains negative values."
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
    sequence: SequenceArray
        Sequence information for sequence embedding.
    """

    id: str
    chain: ChainArray
    token: TokenArray
    atom: AtomArray
    bond: BondArray
    sequence: SequenceArray

    @property
    def num_chains(self) -> int:
        """Number of chains in the structure."""
        return len(self.chain)

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
        num_tokens = self.num_tokens
        num_bonds = len(self.bond)
        return (
            f"TokenizedStructure(\n"
            f"  num_chains: {num_chains}\n"
            f"  num_tokens: {num_tokens}\n"
            f"  num_bonds: {num_bonds}\n"
            f")"
        )

    @classmethod
    def get_empty(
        cls,
        id: str,
        num_chains: int,
        num_tokens: int,
        num_bonds: int,
        num_sequence_tokens: int,
        num_priors: int = 0,
    ) -> Self:
        """Get an empty TokenizedStructure with the specified sizes."""
        return cls(
            id=id,
            chain=ChainArray.get_empty(num_chains),
            token=TokenArray.get_empty(num_tokens),
            atom=AtomArray.get_empty(num_tokens, num_priors=num_priors),
            bond=BondArray.get_empty(num_bonds),
            sequence=SequenceArray.get_empty(num_sequence_tokens),
        )

    def validate(self) -> None:
        """Perform sanity checks on the TokenizedStructure."""
        self.chain.validate()
        self.token.validate()
        self.atom.validate()
        self.bond.validate()
        self.sequence.validate()

    # === Utility functions === #
    def to(self, *args, **kwargs) -> Self:
        """No-op for device/dtype movement for pytorch lightning compatibility."""
        return self

    def copy(self, deepcopy: bool = False) -> Self:
        """Create a copy of the structure."""
        if deepcopy:
            return copy.deepcopy(self)
        else:
            return self.__class__(
                id=self.id,
                chain=self.chain,
                token=self.token,
                atom=self.atom,
                bond=self.bond,
                sequence=self.sequence,
            )

    def copy_with(self, **kwargs) -> Self:
        """Create a copy of the structure with updated fields.

        Parameters
        ----------
        **kwargs
            Fields to update.

        Returns
        -------
        copied_structure: TokenizedStructure
            Copied tokenized structure with updated fields.
        """
        return dataclasses.replace(self, **kwargs)

    def crop(
        self,
        token_indices: np.ndarray,
        sequence_token_indices: np.ndarray | None = None,
    ) -> Self:
        """Crop the structure to the specified token indices.

        Parameters
        ----------
        token_indices: np.ndarray (int)
            Token indices to keep of shape [K,], where K is the number of tokens
            to keep.
        sequence_token_indices: np.ndarray (int) | None
            Sequence token indices to keep of shape [M,], where M is the number of
            sequence tokens to keep. If None, include all sequence tokens corresponding
            to the remaining chains.

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

        # Remove excluding chains
        token_asym_ids = np.unique(cropped_token.asym_id)
        chain_mask = np.isin(self.chain.asym_id, token_asym_ids)
        cropped_chain = self.chain[chain_mask].copy(deepcopy=True)
        for cidx in range(len(cropped_chain)):
            asym_id = cropped_chain.asym_id[cidx]
            mask = cropped_token.asym_id == asym_id

            num_tokens = np.sum(mask).item()
            num_residues = len(np.unique(cropped_token.residue_index[mask]))
            num_atoms = np.sum(cropped_token.num_atoms[mask]).item()

            cropped_chain.num_tokens[cidx] = num_tokens
            cropped_chain.num_residues[cidx] = num_residues
            cropped_chain.num_atoms[cidx] = num_atoms

        if sequence_token_indices is None:
            # Retain all sequence tokens corresponding to the remaining chains
            entity_ids = np.unique(cropped_token.entity_id)
            sequence_token_indices = np.where(
                np.isin(self.sequence.entity_id, entity_ids)
            )[0]

        if len(sequence_token_indices) == len(self.sequence):
            # keep all sequence tokens, no need to index
            cropped_sequence = self.sequence
        else:
            # Keep only the specified sequence tokens
            cropped_sequence = self.sequence[sequence_token_indices]
            # Create a map from original sequence indices to new sequence indices
            # The map must be the size of the ORIGINAL sequence
            seq_token_idx_map = np.full(len(self.sequence), fill_value=-1, dtype=np.int64)

            # sequence_token_indices could be a boolean mask or an integer array.
            # This assignment works for both in NumPy.
            seq_token_idx_map[sequence_token_indices] = np.arange(len(cropped_sequence))

            org_seq_token_idx = cropped_token.seq_token_index
            new_seq_token_idx = seq_token_idx_map[org_seq_token_idx]
            assert np.all(new_seq_token_idx[org_seq_token_idx >= 0] >= 0), (
                "Some tokens are mapped to invalid sequence token indices."
                "Please check the input sequence_token_indices."
            )
            cropped_token = cropped_token.copy_with(seq_token_index=new_seq_token_idx)

        return self.__class__(
            id=self.id,
            chain=cropped_chain,
            token=cropped_token,
            atom=cropped_atom,
            bond=cropped_bond,
            sequence=cropped_sequence,
        )
