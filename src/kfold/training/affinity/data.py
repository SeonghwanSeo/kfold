"""Canonical affinity records and deterministic corpus preparation helpers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from rdkit import Chem

from .sequence import normalize_protein_sequence


class Endpoint(StrEnum):
    KI = "Ki"
    KD = "Kd"
    IC50 = "IC50"
    EC50 = "EC50"


_ENDPOINT_ALIASES = {
    "ki": Endpoint.KI,
    "kd": Endpoint.KD,
    "ic50": Endpoint.IC50,
    "ec50": Endpoint.EC50,
}


@dataclass(frozen=True, kw_only=True)
class AffinityRecord:
    """One quantitative source assay record used by the affinity head."""

    record_id: str
    origin: str
    source: str
    system_id: str
    protein_uniprot: str
    sequence: str
    canonical_smiles: str
    source_smiles: str
    endpoint: Endpoint
    p_activity: float
    assay_descriptor: str
    assay_key: str
    source_record_key: str
    source_assay_id: str | None = None
    source_record_id: str | None = None
    source_document_id: str | None = None
    split: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["endpoint"] = self.endpoint.value
        return data


@dataclass(frozen=True)
class AffinityBatchIndex:
    """Dataset index plus a logical same-assay group for one physical batch."""

    record_index: int
    logical_group_id: int


def canonicalize_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES or raise for invalid source chemistry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def parse_endpoint(value: str) -> Endpoint:
    normalized = value.strip().lower().replace(" ", "")
    try:
        return _ENDPOINT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported quantitative endpoint: {value!r}") from exc


def is_exact_relation(value: str | None) -> bool:
    """Accept only numeric values without a censored/range qualifier."""
    return value is None or value.strip() in {"", "="}


def p_activity_from_nm(value_nm: float) -> float:
    """Convert a positive nanomolar activity to its negative-log10 molar scale."""
    if not math.isfinite(value_nm) or value_nm <= 0.0:
        raise ValueError(
            f"Activity must be a positive finite value in nM, got {value_nm}."
        )
    return 9.0 - math.log10(value_nm)


def p_activity_scale(p_activity: float) -> int:
    """Return the deterministic 1e-4 p-scale bucket used for corpus matching.

    Decimal conversion removes platform/float-formatting variation at a
    half-step boundary.  The bucket is only an identity aid: the original
    floating-point activity remains the regression target.
    """
    value = Decimal(str(p_activity))
    if not value.is_finite():
        raise ValueError(f"Activity must be finite, got {p_activity!r}.")
    return int((value * Decimal("10000")).to_integral_value(rounding=ROUND_HALF_UP))


def assay_key(
    *,
    source: str,
    protein_uniprot: str,
    endpoint: Endpoint,
    descriptor: str,
    source_assay_id: str | None = None,
) -> str:
    """Define the only population within which ranking comparisons are valid."""
    descriptor = descriptor.strip() or "unannotated"
    assay_id = (source_assay_id or "").strip() or "unannotated"
    return "|".join(
        (
            source.strip(),
            protein_uniprot.strip(),
            endpoint.value,
            f"assay_id={assay_id}",
            f"description={descriptor}",
        )
    )


def source_record_key(
    *,
    source: str,
    protein_uniprot: str,
    canonical_smiles: str,
    endpoint: Endpoint,
    p_activity: float,
    descriptor: str,
    source_record_id: str | None = None,
) -> str:
    """Create a stable cross-corpus deduplication key."""
    payload = "\x1f".join(
        (
            source,
            protein_uniprot,
            canonical_smiles,
            endpoint.value,
            f"p_scale={p_activity_scale(p_activity)}",
            descriptor.strip(),
            (source_record_id or "").strip(),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def system_key(*, sequence: str, canonical_smiles: str) -> str:
    """Return the cache identity shared by all labels for one sequence--ligand system."""
    payload = "\x1f".join((normalize_protein_sequence(sequence), canonical_smiles))
    return hashlib.sha256(payload.encode()).hexdigest()


def ligand_key(*, canonical_smiles: str) -> str:
    """Return the cache identity shared by all occurrences of one ligand."""
    return hashlib.sha256(canonical_smiles.encode()).hexdigest()


def protein_key(*, protein_uniprot: str, sequence: str) -> str:
    """Return a stable apo-cache identity for one protein sequence version."""
    payload = "\x1f".join((protein_uniprot.strip(), normalize_protein_sequence(sequence)))
    return hashlib.sha256(payload.encode()).hexdigest()


def split_bucket(
    *,
    assay: str,
    canonical_smiles: str,
    seed: int,
) -> str:
    """Return a deterministic 90/10 compound split inside one assay."""
    digest = hashlib.blake2b(
        f"{seed}:{assay}:{canonical_smiles}".encode(), digest_size=8
    ).digest()
    return "val" if int.from_bytes(digest, "little") % 10 == 0 else "train"


def assign_assay_split(
    records: Iterable[AffinityRecord], *, seed: int
) -> list[AffinityRecord]:
    """Assign train/validation labels without splitting a compound inside an assay."""
    assigned: list[AffinityRecord] = []
    for record in records:
        assigned.append(
            AffinityRecord(
                **{
                    **record.to_dict(),
                    "endpoint": record.endpoint,
                    "split": split_bucket(
                        assay=record.assay_key,
                        canonical_smiles=record.canonical_smiles,
                        seed=seed,
                    ),
                }
            )
        )
    return assigned


def remove_excluded_proteins(
    records: Iterable[AffinityRecord],
    excluded_uniprot: set[str],
) -> list[AffinityRecord]:
    """Remove every record whose target was assigned to the FEP leakage set."""
    return [
        record for record in records if record.protein_uniprot not in excluded_uniprot
    ]


def deduplicate_bindingdb_residual(
    sair_records: Iterable[AffinityRecord],
    bindingdb_records: Iterable[AffinityRecord],
) -> list[AffinityRecord]:
    """Keep only BindingDB records not already represented by the SAIR corpus."""
    sair_keys = {record.source_record_key for record in sair_records}
    residual: list[AffinityRecord] = []
    seen: set[str] = set()
    for record in bindingdb_records:
        if record.source_record_key in sair_keys or record.source_record_key in seen:
            continue
        residual.append(record)
        seen.add(record.source_record_key)
    return residual
