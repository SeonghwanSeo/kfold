import copy
import dataclasses
import io
import pathlib
from functools import cached_property
from typing import Self

import msgpack
import numpy as np

import kfold.constants as C
from kfold.data.schema import Metadata
from kfold.utils.misc import check_array

__all__ = [
    "Chain",
    "Residue",
    "Atom",
    "Bond",
    "RefStructure",
]


# === Helper functions === #
def pack_metadata(metadata: Metadata) -> np.ndarray:
    """Pack Metadata into a numpy bytes array."""
    metadata_dict = metadata.to_dict()
    metadata_serialized = msgpack.packb(metadata_dict)
    return np.array(metadata_serialized, dtype=np.bytes_)


def unpack_metadata(data: np.ndarray) -> Metadata:
    """Unpack Metadata from a numpy bytes array."""
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    metadata_dict = msgpack.unpackb(data)
    return Metadata.from_dict(metadata_dict)


# === Data structures === #
@dataclasses.dataclass(frozen=True)
class Chain:
    """Chain information.

    Shape: [Nchain, ...]

    Attributes
    ----------
    chain_type: int
        Chain type.
    entity_id: int
        Entity ID.
    asym_id: int
        Asymmetric unit ID.
    sym_id: int
        Symmetry ID.
    residue: Residue
        Residue information.
    atom: Atom
        Atom information.
    bond: Bond
        Intra-chain bond information.
    """

    chain_type: int
    entity_id: int
    asym_id: int
    sym_id: int
    residue: "Residue"
    atom: "Atom"
    bond: "Bond"
    smiles: str | None = None  # optional SMILES string for small molecule

    @property
    def ctype(self) -> C.ChainType:
        """Chain type as enum."""
        return C.ChainType(self.chain_type)

    @property
    def num_residues(self) -> int:
        """Number of residues in the chain."""
        return len(self.residue)

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the chain."""
        return len(self.atom)

    @property
    def num_bonds(self) -> int:
        """Number of bonds in the chain."""
        return len(self.bond)

    def get_sequence(self) -> str:
        """Get the amino acid / nucleotide sequence of the chain."""
        if self.ctype.is_nonpolymer:
            raise ValueError("Non-polymer chains do not have a sequence.")
        unk = "X" if self.ctype.is_protein else "N"
        return "".join(
            [
                C.residue.convert_ccd_name_to_one_letter(v, unk)
                for v in self.residue.name.tolist()
            ]
        )

    def get_ccd_sequence(self) -> list[str]:
        """Get the amino acid / nucleotide sequence of the chain."""
        return self.residue.name.tolist()

    def find_atom_index(self, residue_index: int, atom_name: str) -> int:
        """Find atom index given residue index and atom name.

        Parameters
        ----------
        residue_index: int
            1-based residue index.
        atom_name: str
            Atom name.

        Returns
        -------
        atom_index: int
            0-based atom index.

        Raises
        ------
        KeyError
            If the atom is not found.
        """
        # Get the range of atom indices for the given residue
        atom_range = self.residue.iter_residue_atoms(residue_index)
        for atom_index in atom_range:
            if self.atom.name[atom_index] == atom_name:
                return atom_index
        raise KeyError(f"Atom '{atom_name}' not found in residue index {residue_index}.")

    def iter_residue_atoms(self, residue_index: int) -> range:
        """Get the range of atom indices for a given residue index."""
        # residue_index: 1-based index
        start = self.residue.atom_starts[residue_index - 1]
        end = start + self.residue.num_atoms[residue_index - 1]
        return range(start, end)

    def __repr__(self) -> str:
        """FoldingInput summary representation."""
        # Summary statistics
        return (
            "Chain(\n"
            + f"  chain_type: {self.ctype.name}\n"
            + f"  entity_id: {self.entity_id}\n"
            + f"  asym_id: {self.asym_id}\n"
            + f"  sym_id: {self.sym_id}\n"
            + f"  num_residues: {self.num_residues}\n"
            + f"  num_atoms: {self.num_atoms}\n"
            + f"  num_bonds: {self.num_bonds}\n"
            + (f"  smiles: {self.smiles}\n" if self.smiles is not None else "")
            + ")"
        )

    def copy_with(self, deepcopy: bool = False, **kwargs) -> Self:
        """Create a copy of the Chain with modified fields."""
        if deepcopy:
            # Deep copy all fields
            out = copy.deepcopy(self)
        else:
            out = self
        return dataclasses.replace(out, **kwargs)

    # === Numpy serialization === #
    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Convert to a flat dictionary for NPZ storage.

        Returns a dictionary where tensor fields are converted to numpy arrays
        with hierarchical keys like 'chain.asym_id', 'token.token_type', etc.
        """
        result: dict[str, np.ndarray] = {
            "chain_type": np.array(self.chain_type, dtype=np.uint8),
            "entity_id": np.array(self.entity_id, dtype=np.uint16),
            "asym_id": np.array(self.asym_id, dtype=np.uint16),
            "sym_id": np.array(self.sym_id, dtype=np.uint16),
        }
        for prefix, struct in [
            ("residue.", self.residue),
            ("atom.", self.atom),
            ("bond.", self.bond),
        ]:
            for field in dataclasses.fields(struct):
                key = prefix + field.name
                value = getattr(struct, field.name)
                assert value is not None, f"{key} is None"
                result[key] = value
        if self.smiles is not None:
            result["smiles"] = np.array(self.smiles, dtype=np.dtype("U"))
        return result

    @classmethod
    def from_npz_dict(cls, data: dict[str, np.ndarray]) -> Self:
        """Reconstruct from NPZ dictionary."""
        reconstructed = {}
        for prefix, struct_cls in [
            ("residue.", Residue),
            ("atom.", Atom),
            ("bond.", Bond),
        ]:
            struct_data = {
                key[len(prefix) :]: value
                for key, value in data.items()
                if key.startswith(prefix)
            }
            reconstructed[prefix[:-1]] = struct_cls(**struct_data)
        if "smiles" in data:
            reconstructed["smiles"] = data["smiles"].item()
        if "apo_type" in data:
            reconstructed["apo_type"] = tuple(x.item() for x in data["apo_type"])
        return cls(
            chain_type=data["chain_type"].item(),
            entity_id=data["entity_id"].item(),
            asym_id=data["asym_id"].item(),
            sym_id=data["sym_id"].item(),
            **reconstructed,
        )


@dataclasses.dataclass(frozen=True)
class Residue:
    """Residue information.

    Attributes
    ----------
    names: np.ndarray (str, <U6)
        CCD IDs of shape [L,].
        Use up to 6-character names for custom CCDs.
    num_atoms: np.ndarray (int)
        Number of atoms per residue of shape [L,].
    is_standard: np.ndarray (bool)
        Whether the residue is a standard amino acid or nucleic acid of shape [L,].
        NOTE: UNK, DN, N are considered standard residues.
    """

    name: np.ndarray  # [L,], str
    num_atoms: np.ndarray  # [L,], int
    is_standard: np.ndarray  # [L,], bool

    def __len__(self) -> int:
        return len(self.name)

    def __post_init__(self):
        shape = (len(self),)
        check_array(self.name, name="names", dtype=np.str_, shape=shape)
        check_array(self.num_atoms, name="num_atoms", dtype=np.integer, shape=shape)
        check_array(self.is_standard, name="is_standard", dtype=bool, shape=shape)

    @cached_property
    def atom_starts(self) -> np.ndarray:
        """Starting indices of atoms for each residue."""
        dtype = np.int64
        return np.concatenate(
            [np.array([0], dtype=dtype), np.cumsum(self.num_atoms, dtype=dtype)[:-1]]
        )

    def iter_residue_atoms(self, residue_index: int) -> range:
        """Get the range of atom indices for a given residue index."""
        # residue_index: 1-based index
        start = self.atom_starts[residue_index - 1]
        end = start + self.num_atoms[residue_index - 1]
        return range(start, end)

    @classmethod
    def get_default_dtype(cls) -> dict[str, type | np.dtype]:
        """Get default dtypes for each field."""
        return {
            "name": np.dtype("<U6"),
            "num_atoms": np.uint8,  # max 255 atoms per residue
            "is_standard": bool,
            "is_linked": bool,
        }


@dataclasses.dataclass(frozen=True)
class Atom:
    """Atom information.

    Attributes
    ----------
    name: np.ndarray (str, <U4)
        atom names of shape [Natom], str
    label_coords: np.ndarray (float32)
        Holo (bound) state coordinates of shape [Natom, 3],
        This is for model training, so that this field is
        filled to 0 during inference.
    apo_coords: np.ndarray (float32)
        Apo (unbound) state coordinates of shape [Natom, 3],
    bfactor: np.ndarray (float16)
        B-factor values for each residue of shape [L,].
    apo_plddt: np.ndarray (float16)
        predicted LDDT scores for each apo conformation of shape [Napo, L],
        where Napo is the number of available apo conformations.
        For experimental structures, this can be filled to 100.
    """

    name: np.ndarray  # [Natom,], str
    is_resolved: np.ndarray  # [Natom,], bool
    label_coords: np.ndarray  # [Natom, 3], float32
    apo_coords: np.ndarray  # [Natom, 3], float32
    bfactor: np.ndarray  # [L,], int
    apo_plddt: np.ndarray  # [L], int

    def __len__(self) -> int:
        return len(self.name)

    def __post_init__(self):
        shape = self.name.shape
        check_array(self.name, name="name", dtype=np.str_, shape=shape)
        check_array(self.is_resolved, name="is_resolved", dtype=bool, shape=shape)
        check_array(
            self.label_coords, name="label_coords", dtype=np.floating, shape=(*shape, 3)
        )
        check_array(
            self.apo_coords, name="apo_coords", dtype=np.floating, shape=(*shape, 3)
        )
        check_array(self.bfactor, name="bfactor", dtype=np.floating, shape=shape)
        check_array(self.apo_plddt, name="apo_plddt", dtype=np.floating, shape=shape)

    @classmethod
    def get_default_dtype(cls) -> dict[str, type | np.dtype]:
        """Get default dtypes for each field."""
        return {
            "name": np.dtype("<U4"),
            "is_resolved": bool,
            "label_coords": np.float32,
            "apo_coords": np.float32,
            "bfactor": np.float16,
            "apo_plddt": np.float16,
        }


@dataclasses.dataclass(frozen=True)
class Bond:
    """Intra-chain Bond information.

    Shape: [Nbond, ...]

    Attributes
    ----------
    asym_id: np.ndarray
        Chain asym indices of the connecting atoms in the bond of shape [Nbond, 2].
    residue_index: np.ndarray
        Residue indices of the connecting atoms in the bond of shape [Nbond, 2].
    atom_name: np.ndarray
        Atom names of the connecting atoms in the bond of shape [Nbond, 2].
    bond_type: np.ndarray
        Bond types of shape [Nbond,], indicating the type of each bond.

    # TODO: is there any inter-residue bond in RCSB?
    """

    residue_index: np.ndarray  # [Nbond, 2], int
    atom_name: np.ndarray  # [Nbond, 2], int
    bond_type: np.ndarray  # [Nbond,], int

    def __len__(self) -> int:
        return len(self.bond_type)

    def __post_init__(self):
        shape = self.bond_type.shape
        check_array(
            self.residue_index, name="residue_index", dtype=np.integer, shape=(*shape, 2)
        )
        check_array(
            self.atom_name, name="atom_name", dtype=np.dtype("<U4"), shape=(*shape, 2)
        )
        check_array(self.bond_type, name="bond_type", dtype=np.integer, shape=shape)

    @classmethod
    def get_default_dtype(cls) -> dict[str, type | np.dtype]:
        """Get default dtypes for each field."""
        return {
            "residue_index": np.uint32,
            "atom_name": np.dtype("<U4"),
            "bond_type": np.uint8,  # 0-5
        }


@dataclasses.dataclass(frozen=True)
class CovalentConnection:
    """Inter-chain & inter-molecule covalent connection information."""

    asym_id: tuple[int, int]
    residue_index: tuple[int, int]
    atom_names: tuple[str, str]

    # === Numpy serialization === #
    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Convert to a flat dictionary for NPZ storage."""
        result: dict[str, np.ndarray] = {
            "asym_id": np.array(self.asym_id, dtype=np.uint16),
            "residue_index": np.array(self.residue_index, dtype=np.uint16),
            "atom_names": np.array(self.atom_names, dtype=np.dtype("<U4")),
        }
        return result

    @classmethod
    def from_npz_dict(cls, data: dict[str, np.ndarray]) -> Self:
        """Reconstruct from NPZ dictionary."""
        return cls(
            asym_id=tuple(data["asym_id"].tolist()),
            residue_index=tuple(data["residue_index"].tolist()),
            atom_names=tuple(data["atom_names"].tolist()),
        )


@dataclasses.dataclass(kw_only=True)
class RefStructure:
    """Reference Structure information.

    Attributes
    ----------
    chain: list[Chain]
        Chain information.
    connect: list[CovalentConnection]
        Covalent connection information.
    metadata: Metadata
        Metadata information.
        # NOTE: Metadata might not include cluster info.
    """

    chains: list[Chain]
    connections: list[CovalentConnection]
    metadata: Metadata

    @property
    def entity_ids(self) -> list[int]:
        """List of entity IDs in the structure."""
        return [chain.entity_id for chain in self.chains]

    @property
    def asym_ids(self) -> list[int]:
        """List of asym IDs in the structure."""
        return [chain.asym_id for chain in self.chains]

    @property
    def num_chains(self) -> int:
        """Number of chains in the structure."""
        return len(self.chains)

    @property
    def num_polymer_chains(self) -> int:
        """Number of polymer chains in the structure."""
        return sum(chain.ctype.is_polymer for chain in self.chains)

    @property
    def num_nonpolymer_chains(self) -> int:
        """Number of non-polymer chains in the structure."""
        return sum(chain.ctype.is_nonpolymer for chain in self.chains)

    @property
    def num_residues(self) -> int:
        """Number of residues in the structure."""
        return sum(chain.num_residues for chain in self.chains)

    @cached_property
    def num_atoms(self) -> int:
        """Number of atoms in the structure."""
        return sum(chain.num_atoms for chain in self.chains)

    @property
    def num_bonds(self) -> int:
        """Number of bonds in the structure."""
        return sum(chain.num_bonds for chain in self.chains)

    @property
    def num_connections(self) -> int:
        """Number of covalent connections in the structure."""
        return len(self.connections)

    def get_chain_by_asym_id(self, asym_id: int) -> Chain:
        """Get chain by asym_id."""
        for chain in self.chains:
            if chain.asym_id == asym_id:
                return chain
        raise KeyError(f"Chain with asym_id {asym_id} not found.")

    def __repr__(self) -> str:
        """FoldingInput summary representation."""
        # Summary statistics
        num_chains = self.num_chains
        num_connections = self.num_connections

        return (
            "RefStructure(\n"
            + f"  num_chains: {num_chains}\n"
            + f"  num_interfaces: {len(self.metadata.interfaces)}\n"
            + f"  num_connections: {num_connections}\n"
            + "  chains: ["
            + ", ".join(f"{chain.ctype.name}" for chain in self.chains)
            + "]\n"
            "  interfaces: ["
            + ", ".join(
                f"{iface.asym_ids[0]}-{iface.asym_ids[1]}"
                for iface in self.metadata.interfaces
            )
            + "]\n"
            + ")"
        )

    def sanity_check(self) -> None:
        """Perform sanity checks on the structure."""
        # Check that asym_ids are unique
        asym_ids = [chain.asym_id for chain in self.chains]
        if len(asym_ids) != len(set(asym_ids)):
            raise ValueError("Duplicate asym_ids found in chains.")

        # Check that connections refer to valid asym_ids
        valid_asym_ids = set(asym_ids)
        for conn in self.connections:
            for asym_id in conn.asym_id:
                if asym_id not in valid_asym_ids:
                    raise ValueError(f"Connection refers to invalid asym_id {asym_id}.")

        # Check consistency between structure and metadata
        assert self.num_chains == self.metadata.num_chains
        meta_asym_ids = set(self.metadata.asym_ids)
        struct_asym_ids = set(asym_ids)
        if meta_asym_ids != struct_asym_ids:
            raise ValueError("Mismatch between metadata asym_ids and structure asym_ids.")

    # === Numpy serialization for model training === #
    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Convert to a flat dictionary for NPZ storage.

        Returns a dictionary where tensor fields are converted to numpy arrays
        with hierarchical keys like 'chain.0.asym_id', ...
        """
        result: dict[str, np.ndarray] = {}

        # Chains
        for i, chain in enumerate(self.chains):
            prefix = f"chain.{i}."
            chain_dict = chain.to_npz_dict()
            for key, value in chain_dict.items():
                result[prefix + key] = value

        # Connections
        for i, conn in enumerate(self.connections):
            prefix = f"connect.{i}."
            conn_dict = conn.to_npz_dict()
            for key, value in conn_dict.items():
                result[prefix + key] = value

        # Metadata
        result["_metadata"] = pack_metadata(self.metadata)

        return result

    @classmethod
    def from_npz_dict(cls, data: dict[str, np.ndarray]) -> Self:
        """Reconstruct from NPZ dictionary."""
        # Chains
        chains: list[Chain] = []
        i = 0
        while True:
            prefix = f"chain.{i}."
            chain_keys = [key for key in data.keys() if key.startswith(prefix)]
            if not chain_keys:
                break
            chain_data = {key[len(prefix) :]: data[key] for key in chain_keys}
            chains.append(Chain.from_npz_dict(chain_data))
            i += 1

        connections: list[CovalentConnection] = []
        j = 0
        while True:
            prefix = f"connect.{j}."
            connection_keys = [key for key in data.keys() if key.startswith(prefix)]
            if not connection_keys:
                break
            connection_data = {key[len(prefix) :]: data[key] for key in connection_keys}
            connections.append(CovalentConnection.from_npz_dict(connection_data))
            j += 1

        metadata = unpack_metadata(data["_metadata"])

        return cls(
            chains=chains,
            connections=connections,
            metadata=metadata,
        )

    def dump_npz(self, path: pathlib.Path | str) -> None:
        """Save to compressed NPZ file."""
        path = pathlib.Path(path)
        np.savez_compressed(
            path,
            **self.to_npz_dict(),
        )

    def save_npz(self, path: pathlib.Path | str) -> None:
        """Save to compressed NPZ file."""
        self.dump_npz(path)

    @classmethod
    def load_npz(cls, path: pathlib.Path | str | io.BytesIO) -> Self:
        """Load from NPZ file."""
        with np.load(path) as data:
            return cls.from_npz_dict(dict(data))
