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
    num_chains: int | None = None
    num_interfaces: int | None = None
    pH: float | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class PredictionRecord(JsonSerializable):
    """Metadata record from structure prediction."""

    # TODO: add more fields if necessary
    model_name: str | None = None
    plddt: float | None = None
    pae_mean: float | None = None


@dataclass(frozen=True, slots=True)
class ChainInfo(JsonSerializable):
    chain_type: C.chain.ChainType
    chain_name: str  # same to auth_asym_id
    entity_id: int  # starts from 1
    asym_id: int  # starts from 1
    sym_id: int  # starts from 1
    num_residues: int
    cluster_id: str
    valid: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        data = super(ChainInfo, self).to_dict()
        data["chain_type"] = self.chain_type.name
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create ChainInfo from dictionary."""
        data = data.copy()
        data["chain_type"] = C.chain.ChainType[data["chain_type"]]
        return cls(**data)


@dataclass(frozen=True, slots=True)
class InterfaceInfo(JsonSerializable):
    # NOTE(seonghwanseo): I change (asym_id1, asym_id2) to {asym_id} to
    # support multi-chain interaces (e.g., Protac)
    asym_ids: tuple[int, ...]
    # covalent bond interface. If True, the chains must be sampled together.
    is_bonded: bool = False
    valid: bool = True

    def __post_init__(self):
        assert len(self.asym_ids) >= 2, "Interface must involve at least two chains."


@dataclass
class Metadata:
    id: str
    source: str  # e.g., "rcsb"
    exp: ExperimentRecord | None = None
    prediction: PredictionRecord | None = None
    chains: list[ChainInfo] = field(default_factory=list)
    interfaces: list[InterfaceInfo] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Metadata(id={self.id}, source={self.source}, "
            f"num_chains={self.num_chains}, num_interfaces={len(self.interfaces)})"
        )

    def __post_init__(self):
        # FIXME: we may want to add more sources later
        assert self.source in {"rcsb"}, f"Unsupported source: {self.source}"

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
