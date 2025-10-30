# starting from https://github.com/jwohlwend/boltz/blob/v1.0.0/src/boltz/data/types.py
from dataclasses import dataclass, fields
from io import BytesIO
from pathlib import Path
from typing import Self

import numpy as np
from rdkit.Chem.rdchem import BondType, ChiralType

# Constants
RDKit_CHIRALITY_MAP = {
    0: ChiralType.CHI_UNSPECIFIED,
    1: ChiralType.CHI_TETRAHEDRAL_CW,
    2: ChiralType.CHI_TETRAHEDRAL_CCW,
}
BOLTZ_BOND_TYPES: dict[int, BondType] = {
    0: BondType.OTHER,
    1: BondType.SINGLE,
    2: BondType.DOUBLE,
    3: BondType.TRIPLE,
    4: BondType.AROMATIC,
}


@dataclass(frozen=True)
class NumpySerializable:
    """Serializable datatype."""

    @classmethod
    def load(cls, path: str | Path | BytesIO) -> Self:
        """Load the object from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        return cls(**np.load(path))

    @classmethod
    def loads(cls, data: bytes) -> Self:
        """Load the object from bytes.

        Parameters
        ----------
        data : Bytes
            The bytes containing the serialized object.

        Returns
        -------
        Serializable
            The loaded object.
        """
        buffer = BytesIO(data)
        return cls.load(buffer)

    def dump(self, path: str | Path | BytesIO) -> None:
        """Dump the object to an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        state = {field.name: getattr(self, field.name) for field in fields(self)}
        np.savez_compressed(path, **state)

    def dumps(self) -> bytes:
        """Dump the object to bytes.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        buffer = BytesIO()
        self.dump(buffer)
        return buffer.getvalue()


####################################################################################################
# STRUCTURE
####################################################################################################

Atom: list[tuple[str, np.dtype]] = [
    ("name", np.dtype("4i1")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("coords", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("chirality", np.dtype("i1")),
]

Bond: list[tuple[str, np.dtype]] = [
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

Residue: list[tuple[str, np.dtype]] = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

Chain: list[tuple[str, np.dtype]] = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
]

Connection: list[tuple[str, np.dtype]] = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
]

Interface: list[tuple[str, np.dtype]] = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
]


@dataclass(frozen=True)
class BoltzStructure(NumpySerializable):
    """Structure datatype."""

    atoms: np.ndarray  # (Natoms,)
    bonds: np.ndarray  # (Nbonds,)
    residues: np.ndarray  # (Nresidues,)
    chains: np.ndarray  # (Nchains,)
    connections: np.ndarray  # (Nconnections,)
    interfaces: np.ndarray  # (Ninterfaces,)
    mask: np.ndarray  # (Nchains,)

    @classmethod
    def load(cls, path: str | Path | BytesIO) -> Self:
        """Load a structure from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Structure
            The loaded structure.

        """
        structure = np.load(path)
        struct = cls(
            atoms=structure["atoms"],
            bonds=structure["bonds"],
            residues=structure["residues"],
            chains=structure["chains"],
            connections=structure["connections"].astype(Connection, copy=False),
            interfaces=structure["interfaces"],
            mask=structure["mask"],
        )
        structure.close()
        return struct
