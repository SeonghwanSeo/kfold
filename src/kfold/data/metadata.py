"""Implemented from https://github.com/jwohlwend/boltz"""

from dataclasses import dataclass

import kfold.constants as C


@dataclass
class RCSBRecord:
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


@dataclass
class PredictionRecord:
    """Metadata record from structure prediction."""

    # TODO: add more fields if necessary
    model_name: str | None = None
    plddt: float | None = None
    pae_mean: float | None = None


@dataclass
class ChainInfo:
    chain_type: C.chain.ChainType
    chain_name: str  # same to auth_asym_id
    ccd: str | None  # None for protein/nucleic acid chains
    entity_id: int  # starts from 1
    asym_id: int  # starts from 1
    sym_id: int  # starts from 1
    num_residues: int
    num_tokens: int
    num_atoms: int


@dataclass
class InterfaceInfo:
    # NOTE(seonghwanseo): I change (asym_id1, asym_id2) to {asym_id} to
    # support multi-chain interaces (e.g., Protac)
    asym_ids: list[int]
    # covalent bond interface. If True, the chains must be sampled together.
    is_bonded: bool = False

    def __post_init__(self):
        assert len(self.asym_ids) >= 2, "Interface must involve at least two chains."


@dataclass
class Metadata:
    id: str
    source: str  # e.g., "rcsb"
    rcsb: RCSBRecord | None = None
    prediction: PredictionRecord | None = None
    chains: list[ChainInfo] = []
    interfaces: list[InterfaceInfo] = []

    def __post_init__(self):
        # FIXME: we may wand to add more sources later
        assert self.source in {"rcsb"}, f"Unsupported source: {self.source}"

    @property
    def num_chains(self) -> int:
        return len(self.chains)
