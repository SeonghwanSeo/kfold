import dataclasses
import io
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np

import kfold.constants as C
from kfold.data.layout import PlainLayout
from kfold.data.metadata import Metadata
from kfold.utils.misc import check_array

__all__ = ["Chain", "Token", "Atom", "Bond", "TokenizedStructure"]


# === Tokenized data structures === #
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Chain(PlainLayout[np.ndarray]):
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
    num_tokens: np.ndarray (int)
        Number of tokens per chain of shape [Nchain,].
    num_residues: np.ndarray (int)
        Number of residues per chain of shape [Nchain,].
    num_atoms: np.ndarray (int)
        Number of atoms per chain of shape [Nchain,].
    """

    chain_type: np.ndarray  # [Nchain,], int
    entity_id: np.ndarray  # [Nchain,], int
    asym_id: np.ndarray  # [Nchain,], int
    sym_id: np.ndarray  # [Nchain,], int
    num_tokens: np.ndarray  # [Nchain,], int
    num_residues: np.ndarray  # [Nchain,], int
    num_atoms: np.ndarray  # [Nchain,], int

    # === Properties === #
    @property
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
        check_array(self.num_tokens, name="num_tokens", dtype=np.integer, shape=shape)
        check_array(self.num_residues, name="num_residues", dtype=np.integer, shape=shape)
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Residue(PlainLayout[np.ndarray]):
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
    resolved_mask: np.ndarray (bool)
        Mask tensor of shape [L,], indicating residues to be resolved.
    is_standard: np.ndarray (bool)
        Boolean tensor of shape [L,], indicating whether the residue is standard.
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
    resolved_mask: np.ndarray  # [L,], bool
    is_standard: np.ndarray  # [L,], bool

    @property
    def layout_shape(self) -> tuple[int, ...]:
        return self.res_type.shape  # [Nresidue,]

    def __post_init__(self):
        shape = self.layout_shape
        check_array(self.name, name="name", dtype=np.dtype("<U5"), shape=shape)
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
        check_array(self.resolved_mask, name="resolved_mask", dtype=np.bool_, shape=shape)
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Token(PlainLayout[np.ndarray]):
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
    disto_index: np.ndarray (int)
        Distogram atom index of shape [L,], used for distogram calculations.
    center_index: np.ndarray (int)
        Center atom index of shape [L,], used for center calculations.
    resolved_mask: np.ndarray (bool)
        Mask tensor of shape [L,], indicating tokens to be resolved.
    is_standard: np.ndarray (bool)
        Boolean tensor of shape [L,], indicating whether the token is standard.
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
    resolved_mask: np.ndarray  # [L,], bool
    is_standard: np.ndarray  # [L,], bool

    @property
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
        check_array(self.resolved_mask, name="resolved_mask", dtype=np.bool_, shape=shape)
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Atom(PlainLayout[np.ndarray]):
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
    coords: np.ndarray (float32)
        Holo (bound) state coordinates of shape [Ntoken, 24, Nholo, 3],
        where Nholo is the number of ensemble holo conformations.
        This is used as the ground truth for training, and may be set to 0
        for inference.
    apo_coords: np.ndarray (float32)
        Apo (unbound) state coordinates of shape [Ntoken, 24, Napo, 3],
        where Napo is the number of apo conformations.
    resolved_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24,] indicating atoms to be resolved.
    pad_mask: np.ndarray (bool)
        Boolean mask of shape [Ntoken, 24,] indicating atoms to be resolved.
    """

    ref_atom_name_chars: np.ndarray  # [Ntoken, 24, 4], int
    ref_element: np.ndarray  # [Ntoken, 24], int
    ref_charge: np.ndarray  # [Ntoken, 24,], float
    ref_pos: np.ndarray  # [Ntoken, 24, 3], float32
    coords: np.ndarray  # [Ntoken, 24, Nholo, 3], float32
    apo_coords: np.ndarray  # [Ntoken, 24, Napo, 3], float32
    resolved_mask: np.ndarray  # [Ntoken, 24], bool
    apo_mask: np.ndarray  # [Ntoken, 24, Napo], bool
    pad_mask: np.ndarray  # [Ntoken, 24], bool

    @property
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
        check_array(
            self.coords,
            name="coords",
            dtype=np.floating,
            shape=(*shape, -1, 3),
        )
        check_array(
            self.apo_coords, name="apo_coords", dtype=np.floating, shape=(*shape, -1, 3)
        )
        check_array(self.resolved_mask, name="resolved_mask", dtype=np.bool_, shape=shape)
        check_array(self.apo_mask, name="apo_mask", dtype=np.bool_, shape=(*shape, -1))
        check_array(self.pad_mask, name="pad_mask", dtype=np.bool_, shape=shape)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Bond(PlainLayout[np.ndarray]):
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

    @property
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenizedStructure:
    """Tokenized representation of a molecular structure.

    Attributes
    ----------
    chain: Chain
        Chain information.
    token: Token
        Token information.
    atom: Atom
        Atom information.
    bond: Bond
        Bond information.
    metadata: Metadata
        Metadata information.
    """

    chain: Chain
    residue: Residue
    token: Token
    atom: Atom
    bond: Bond
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
            ("chain.", Chain),
            ("residue.", Residue),
            ("token.", Token),
            ("atom.", Atom),
            ("bond.", Bond),
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
