"""Implemented from https://github.com/jwohlwend/boltz"""

from dataclasses import dataclass, field, fields
from typing import Self

import kfold.constants as C


@dataclass(frozen=True, slots=True)
class JsonSerializable:
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create ExperimentRecord from dictionary."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ExperimentRecord(JsonSerializable):
    """Metadata record from RCSB PDB."""

    pdb_id: str | None = None
    resolution: float | None = None
    method: str | None = None
    deposited: str | None = None
    released: str | None = None
    revised: str | None = None
    pH: float | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class PredictionRecord(JsonSerializable):
    """Metadata record from structure prediction."""

    # TODO: add more fields if necessary
    model: str | None = None  # e.g., "AlphaFold2"
    plddt: float | None = None


@dataclass(frozen=True, slots=True)
class ChainInfo(JsonSerializable):
    chain_name: str  # User/Author-defined chain name
    chain_type: int  # C.ChainType enum value
    entity_id: int  # starts from 1
    asym_id: int  # starts from 1
    sym_id: int  # starts from 1
    num_residues: int
    cluster_id: str | None = None
    valid: bool = True
    smiles: str | None = None
    description: str | None = None

    @property
    def ctype(self) -> C.ChainType:
        return C.ChainType(self.chain_type)


@dataclass(frozen=True, slots=True)
class InterfaceInfo(JsonSerializable):
    asym_ids: tuple[int, int]
    # covalent bond interface. If True, the chains must be sampled together.
    is_bonded: bool = False
    valid: bool = True

    def __post_init__(self):
        assert len(self.asym_ids) == 2, "Interface must involve exactly two chains."


@dataclass(frozen=True, slots=True)
class Metadata:
    id: str
    source: str  # e.g., "rcsb"
    exp: ExperimentRecord | None = None
    prediction: PredictionRecord | None = None
    chains: list[ChainInfo]
    interfaces: list[InterfaceInfo] = field(
        default_factory=list
    )  # only used in training.

    def __repr__(self) -> str:
        return (
            f"Metadata(id={self.id}, source={self.source}, "
            f"num_chains={self.num_chains}, num_interfaces={len(self.interfaces)})"
        )

    def __post_init__(self):
        # FIXME: we may want to add more sources later
        assert self.source in {
            "rcsb",  # experimentally determined structures from RCSB PDB
            "prediction",  # synthetic structures from structure prediction
            "query",  # User query for inference
        }, f"Unsupported source: {self.source}"

        if self.source == "query":
            assert self.exp is None and self.prediction is None, (
                "Query metadata should not have exp or prediction records."
            )
            assert len(self.interfaces) == 0, "Query metadata should not have interfaces."

        # Check all asym_ids are unique
        asym_id_set = set()
        for chain in self.chains:
            if chain.asym_id in asym_id_set:
                raise ValueError(f"Duplicate asym_id found: {chain.asym_id}")
            asym_id_set.add(chain.asym_id)

    @property
    def asym_ids(self) -> list[int]:
        asym_ids = [chain.asym_id for chain in self.chains]
        return asym_ids

    def get_chain_by_asym_id(self, asym_id: int) -> ChainInfo:
        for chain in self.chains:
            if chain.asym_id == asym_id:
                return chain
        raise ValueError(f"Chain with asym_id {asym_id} not found.")

    @property
    def num_chains(self) -> int:
        return len(self.chains)

    @property
    def num_residues(self) -> int:
        return sum(chain.num_residues for chain in self.chains)

    @property
    def num_valid_chains(self) -> int:
        return sum(1 for chain in self.chains if chain.valid)

    @property
    def num_valid_residues(self) -> int:
        return sum(chain.num_residues for chain in self.chains if chain.valid)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "exp": self.exp.to_dict() if self.exp else None,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "chains": [chain.to_dict() for chain in self.chains],
            "interfaces": [interface.to_dict() for interface in self.interfaces],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        exp = ExperimentRecord.from_dict(data["exp"]) if data.get("exp") else None
        prediction = (
            PredictionRecord.from_dict(data["prediction"])
            if data.get("prediction")
            else None
        )
        chains = [ChainInfo.from_dict(c) for c in data.get("chains", [])]
        interfaces = [InterfaceInfo.from_dict(i) for i in data.get("interfaces", [])]
        return cls(
            id=data["id"],
            source=data["source"],
            exp=exp,
            prediction=prediction,
            chains=chains,
            interfaces=interfaces,
        )
