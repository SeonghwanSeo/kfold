"""Source-row normalization for SAIR and BindingDB affinity supervision."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections.abc import Iterable, Mapping
from typing import TextIO

from .data import (
    AffinityRecord,
    Endpoint,
    assay_key,
    canonicalize_smiles,
    p_activity_from_nm,
    p_activity_scale,
    source_record_key,
    system_key,
)
from .sequence import normalize_protein_sequence

_EXACT_NUMBER = re.compile(
    r"^\s*(?:=\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def parse_exact_numeric(value: object) -> float | None:
    """Accept one finite exact number; reject censored, approximate, or ranges."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    match = _EXACT_NUMBER.match(str(value))
    if match is None:
        return None
    numeric = float(match.group(1))
    return numeric if math.isfinite(numeric) else None


def iter_bindingdb_rows(handle: TextIO):
    """Yield source rows without losing values under duplicate BDB headers.

    A dated public mirror repeats its 11-column chain-annotation suffix in the
    header while retaining the canonical 49 values in data rows. ``DictReader``
    keeps the final duplicate key, which makes sequence and UniProt columns
    appear empty. Retain the first normalized occurrence for both that mirror
    and standard BindingDB TSV files.
    """
    reader = csv.reader(handle, delimiter="\t")
    try:
        header = next(reader)
    except StopIteration:
        return
    positions: dict[str, tuple[str, int]] = {}
    for index, name in enumerate(header):
        normalized = " ".join(name.lower().split())
        positions.setdefault(normalized, (name, index))
    for values in reader:
        yield {
            name: values[index] if index < len(values) else ""
            for name, index in positions.values()
        }


def _string(row: Mapping[str, object], *names: str) -> str:
    def normalized_key(value: object) -> str:
        return " ".join(str(value).lower().split())

    lower = {normalized_key(key): value for key, value in row.items()}
    for name in names:
        value = lower.get(normalized_key(name))
        if value is not None and str(value).strip() not in {"", "nan", "None"}:
            return str(value).strip()
    return ""


def _optional_string(row: Mapping[str, object], *names: str) -> str | None:
    value = _string(row, *names)
    return value or None


def _system_id(sequence: str, canonical_smiles: str) -> str:
    return f"sys-{system_key(sequence=sequence, canonical_smiles=canonical_smiles)}"


def normalize_sair_record(
    system_row: Mapping[str, object],
    label_row: Mapping[str, object],
) -> AffinityRecord | None:
    """Convert one sidecar SAIR label into the ranking-table schema.

    SAIR's released label sidecar contains only ``pIC50``/``potency`` and does
    not preserve the original BindingDB endpoint or relation.  It is therefore
    tagged as the explicit ``IC50`` proxy endpoint rather than silently mixed
    with raw BindingDB Ki/Kd/EC50 labels.
    """
    p_activity = parse_exact_numeric(label_row.get("pIC50"))
    if p_activity is None:
        p_activity = parse_exact_numeric(label_row.get("potency"))
    if p_activity is None:
        return None
    protein_uniprot = _string(system_row, "protein_uniprot")
    sequence = _string(system_row, "sequence")
    source_smiles = _string(label_row, "source_smiles")
    canonical_smiles = _string(system_row, "canonical_smiles")
    if not (protein_uniprot and sequence and source_smiles and canonical_smiles):
        return None
    sequence = normalize_protein_sequence(sequence)
    # ``systems.parquet`` is produced from the canonicalized SAIR system field.
    # Re-running RDKit per label is both redundant and a dominant CPU cost at
    # full-corpus scale; raw BindingDB still goes through canonicalization below.
    source = _string(label_row, "source") or "SAIR"
    assay_type = _string(label_row, "assay_type")
    if assay_type.lower() in {"na", "n/a", "none", "null"}:
        assay_type = ""
    description = _string(label_row, "assay_description")
    descriptor = (
        " | ".join(part for part in (assay_type, description) if part) or "unannotated"
    )
    source_id = _optional_string(label_row, "source_record_id", "source_match_key")
    source_document_id = _optional_string(label_row, "source_document_id")
    source_assay_id = None
    # BindingDB rows in the public SAIR sidecar often have neither an assay
    # description nor a source assay ID.  Treating every label for a protein as
    # a single assay would fabricate ranking pairs, so use its source signature
    # as a singleton assay identifier in that case.
    if descriptor == "unannotated":
        source_assay_id = source_id
    fallback_entry_id = ":".join(
        (
            _string(system_row, "sair_entry_id"),
            _string(label_row, "affinity_label_index"),
        )
    )
    entry_id = _string(label_row, "affinity_label_uid") or fallback_entry_id
    if source_assay_id is None and descriptor == "unannotated":
        source_assay_id = f"singleton:{entry_id}"
    endpoint = Endpoint.IC50
    return AffinityRecord(
        record_id=f"sair:{entry_id}",
        origin="SAIR",
        source=source,
        system_id=_system_id(sequence, canonical_smiles),
        protein_uniprot=protein_uniprot,
        sequence=sequence,
        canonical_smiles=canonical_smiles,
        source_smiles=source_smiles,
        endpoint=endpoint,
        p_activity=p_activity,
        assay_descriptor=descriptor,
        assay_key=assay_key(
            source=source,
            protein_uniprot=protein_uniprot,
            endpoint=endpoint,
            descriptor=descriptor,
            source_assay_id=source_assay_id,
        ),
        source_record_key=source_record_key(
            source=source,
            protein_uniprot=protein_uniprot,
            canonical_smiles=canonical_smiles,
            endpoint=endpoint,
            p_activity=p_activity,
            descriptor=descriptor,
            source_record_id=source_id,
        ),
        source_assay_id=source_assay_id,
        source_record_id=source_id,
        source_document_id=source_document_id,
    )


_BINDINGDB_ENDPOINT_COLUMNS: dict[Endpoint, tuple[str, ...]] = {
    Endpoint.KI: ("Ki (nM)", "Ki(nM)", "Ki"),
    Endpoint.KD: ("Kd (nM)", "Kd(nM)", "Kd"),
    Endpoint.IC50: ("IC50 (nM)", "IC50(nM)", "IC50"),
    Endpoint.EC50: ("EC50 (nM)", "EC50(nM)", "EC50"),
}


def normalize_bindingdb_row(row: Mapping[str, object]) -> list[AffinityRecord]:
    """Build exact-numeric Ki/Kd/IC50/EC50 labels from one BindingDB TSV row."""
    protein_uniprot = (
        _string(
            row,
            "UniProt (SwissProt) Primary ID of Target Chain",
            "UniProt ID",
            "UniProt",
        )
        .split(",")[0]
        .strip()
    )
    sequence = _string(
        row,
        "BindingDB Target Chain Sequence",
        "Target Chain Sequence",
        "Target Sequence",
    )
    source_smiles = _string(row, "Ligand SMILES", "SMILES", "Ligand SMILES (Canonical)")
    if not (protein_uniprot and sequence and source_smiles):
        return []
    sequence = normalize_protein_sequence(sequence)
    try:
        canonical_smiles = canonicalize_smiles(source_smiles)
    except ValueError:
        return []
    source = "BindingDB"
    assay_id = _optional_string(row, "Assay ID", "BindingDB Assay ID")
    record_id = _optional_string(
        row,
        "BindingDB Reactant_set_id",
        "BindingDB Reactant Set ID",
        "Reactant_set_id",
    )
    assay_description = _optional_string(
        row, "Assay Description", "Measurement Description"
    )
    source_document_id = _optional_string(row, "PubMed ID", "DOI", "Article DOI")
    # DOI/PubMed are provenance, not evidence that separate records share the
    # same experimental assay.  A raw BindingDB row without an assay identifier
    # is therefore a singleton regression observation.
    descriptor = assay_description or "unannotated"
    records: list[AffinityRecord] = []
    for endpoint, columns in _BINDINGDB_ENDPOINT_COLUMNS.items():
        raw_value = _string(row, *columns)
        value_nm = parse_exact_numeric(raw_value)
        if value_nm is None or value_nm <= 0.0:
            continue
        p_activity = p_activity_from_nm(value_nm)
        source_key = source_record_key(
            source=source,
            protein_uniprot=protein_uniprot,
            canonical_smiles=canonical_smiles,
            endpoint=endpoint,
            p_activity=p_activity,
            descriptor=descriptor,
            source_record_id=record_id,
        )
        record_suffix = record_id or hashlib.sha256(source_key.encode()).hexdigest()[:16]
        resolved_assay_id = assay_id or f"singleton:{record_id or source_key}"
        records.append(
            AffinityRecord(
                record_id=f"bindingdb:{record_suffix}:{endpoint.value}",
                origin="BindingDB-residual",
                source=source,
                system_id=_system_id(sequence, canonical_smiles),
                protein_uniprot=protein_uniprot,
                sequence=sequence,
                canonical_smiles=canonical_smiles,
                source_smiles=source_smiles,
                endpoint=endpoint,
                p_activity=p_activity,
                assay_descriptor=descriptor,
                assay_key=assay_key(
                    source=source,
                    protein_uniprot=protein_uniprot,
                    endpoint=endpoint,
                    descriptor=descriptor,
                    source_assay_id=resolved_assay_id,
                ),
                source_record_key=source_key,
                source_assay_id=resolved_assay_id,
                source_record_id=record_id,
                source_document_id=source_document_id,
            )
        )
    return records


def deduplicate_exact_records(records: Iterable[AffinityRecord]) -> list[AffinityRecord]:
    """Keep the first occurrence of each source-specific normalized record."""
    deduplicated: list[AffinityRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.source_record_key in seen:
            continue
        seen.add(record.source_record_key)
        deduplicated.append(record)
    return deduplicated


def sair_bindingdb_membership_key(record: AffinityRecord) -> str:
    """Conservative SAIR membership key when SAIR lacks raw BDB record IDs.

    The public SAIR sidecar does not retain BindingDB's relation or endpoint
    fields.  Matching exact protein, canonical ligand, and an integerized
    four-decimal p-scale
    avoids re-inserting a BDB measurement already represented by SAIR while
    retaining raw BDB endpoint identity in all residual records that survive.
    """
    payload = "\x1f".join(
        (
            record.protein_uniprot,
            record.canonical_smiles,
            f"p_scale={p_activity_scale(record.p_activity)}",
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def bindingdb_residual_from_sair(
    sair_records: Iterable[AffinityRecord],
    bindingdb_records: Iterable[AffinityRecord],
) -> list[AffinityRecord]:
    """Remove raw BDB records conservatively represented by a SAIR BDB label."""
    sair_membership = {
        sair_bindingdb_membership_key(record)
        for record in sair_records
        if record.source == "BindingDB"
    }
    residual: list[AffinityRecord] = []
    seen: set[str] = set()
    for record in bindingdb_records:
        if record.source_record_key in seen:
            continue
        seen.add(record.source_record_key)
        if sair_bindingdb_membership_key(record) in sair_membership:
            continue
        residual.append(record)
    return residual
