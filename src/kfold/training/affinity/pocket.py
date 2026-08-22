"""Target-level pocket annotations for Boltz-2-style affinity cropping."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POCKET_ANNOTATION_CONTRACT_V1 = "boltz2_target_pocket_v1"
POCKET_EVIDENCE_CONTRACT_V1 = "boltz2_target_pocket_evidence_v1"
AFFINITY_CROP_CONTRACT_V2 = "boltz2_affinity_crop_v2"
DISTOGRAM_POCKET_ANNOTATION_CONTRACT_V1 = "affinity_distogram_target_consensus_pocket_v1"
DISTOGRAM_POCKET_EVIDENCE_CONTRACT_V1 = "affinity_distogram_target_consensus_evidence_v1"
POCKET80K_EVIDENCE_CONTRACT_V1 = "affinity_pocket80k_evidence_v1"
POCKET80K_TARGET_CONSENSUS_CONTRACT_V1 = "affinity_pocket80k_target_consensus_v1"
POCKET80K_TARGET_CROP_CONTRACT_V1 = "target_consensus_100_v1"
POCKET80K_QUERY_ADAPTIVE_DELTA20_CONTRACT_V1 = "query_adaptive_delta20_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, kw_only=True)
class PocketAnnotation:
    """One protein's consensus minimum-distance profile from Algorithm 2 evidence."""

    protein_key: str
    protein_residue_min_distance: np.ndarray
    selected_request_id: str
    evidence_contract_sha256: str

    def __post_init__(self) -> None:
        distances = np.asarray(self.protein_residue_min_distance, dtype=np.float32)
        if distances.ndim != 1 or not len(distances):
            raise ValueError("Pocket annotation distances must be a non-empty vector.")
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError(
                "Pocket annotation distances must be finite and non-negative."
            )
        object.__setattr__(self, "protein_residue_min_distance", distances)


@dataclass(frozen=True, kw_only=True)
class PocketEvidence:
    """One binder-complex structure selected by confidence before consensus."""

    request_id: str
    protein_key: str
    protein_residue_min_distance: np.ndarray
    iptm: float
    structure_sha256: str
    evidence_contract_sha256: str

    def __post_init__(self) -> None:
        distances = np.asarray(self.protein_residue_min_distance, dtype=np.float32)
        if distances.ndim != 1 or not len(distances):
            raise ValueError("Pocket evidence distances must be a non-empty vector.")
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("Pocket evidence distances must be finite and non-negative.")
        if not np.isfinite(self.iptm):
            raise ValueError("Pocket evidence ipTM must be finite.")
        object.__setattr__(self, "protein_residue_min_distance", distances)


@dataclass(frozen=True, kw_only=True)
class DistogramPocketEvidence:
    """One cached train ligand's coordinate-free protein distance profile."""

    system_id: str
    protein_key: str
    canonical_smiles: str
    protein_token_min_expected_distance: np.ndarray

    def __post_init__(self) -> None:
        distances = np.asarray(self.protein_token_min_expected_distance, dtype=np.float32)
        if distances.ndim != 1 or not len(distances):
            raise ValueError("Distogram pocket evidence requires a non-empty vector.")
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError(
                "Distogram pocket distances must be finite and non-negative."
            )
        object.__setattr__(self, "protein_token_min_expected_distance", distances)


@dataclass(frozen=True, kw_only=True)
class Pocket80kEvidence:
    """One coordinate-free 80k distogram profile used for target consensus."""

    system_id: str
    protein_key: str
    canonical_smiles: str
    protein_min_expected_distance: np.ndarray
    protein_mean_normalized_entropy: np.ndarray
    pocket_hlp_15a: float | None
    pocket_residue_count: int

    def __post_init__(self) -> None:
        distances = np.asarray(self.protein_min_expected_distance, dtype=np.float32)
        entropy = np.asarray(self.protein_mean_normalized_entropy, dtype=np.float32)
        if distances.ndim != 1 or not len(distances) or entropy.shape != distances.shape:
            raise ValueError(
                "80k evidence distance and entropy must be matching vectors."
            )
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("80k evidence distances must be finite and non-negative.")
        if not np.isfinite(entropy).all() or (entropy < 0).any() or (entropy > 1).any():
            raise ValueError("80k evidence entropy must be finite and normalized.")
        if self.pocket_residue_count < 0:
            raise ValueError("80k pocket residue count cannot be negative.")
        if self.pocket_hlp_15a is None:
            if self.pocket_residue_count:
                raise ValueError("A non-empty 80k pocket requires finite H_LP.")
        elif not np.isfinite(self.pocket_hlp_15a) or not 0 <= self.pocket_hlp_15a <= 1:
            raise ValueError("80k pocket H_LP must be normalized when present.")
        object.__setattr__(self, "protein_min_expected_distance", distances)
        object.__setattr__(self, "protein_mean_normalized_entropy", entropy)


@dataclass(frozen=True, kw_only=True)
class Pocket80kTargetConsensus:
    """One target's frozen 80k medoid distance profile from ten ligands."""

    protein_key: str
    protein_residue_min_distance: np.ndarray
    medoid_system_id: str
    medoid_canonical_smiles: str
    medoid_mean_overlap: float

    def __post_init__(self) -> None:
        distance = np.asarray(self.protein_residue_min_distance, dtype=np.float32)
        if distance.ndim != 1 or not len(distance):
            raise ValueError("80k medoid distance profile must be a non-empty vector.")
        if not np.isfinite(distance).all() or (distance < 0).any():
            raise ValueError("80k medoid distances must be finite and non-negative.")
        if (
            not np.isfinite(self.medoid_mean_overlap)
            or not 0 <= self.medoid_mean_overlap <= 1
        ):
            raise ValueError("80k medoid overlap must lie in [0, 1].")
        object.__setattr__(self, "protein_residue_min_distance", distance)


def select_best_structure_sample(iptm: np.ndarray) -> tuple[int, float]:
    """Return the deterministic highest-ipTM sample for one complex request."""
    values = np.asarray(iptm, dtype=np.float32)
    if values.ndim != 1 or not len(values):
        raise ValueError("ipTM scores must be a non-empty one-dimensional vector.")
    if not np.isfinite(values).all():
        raise ValueError("ipTM scores must be finite.")
    index = int(np.argmax(values))
    return index, float(values[index])


def protein_residue_min_ligand_distances(
    protein_atom_coords: np.ndarray,
    ligand_atom_coords: np.ndarray,
    residue_atom_starts: np.ndarray,
    residue_atom_ends: np.ndarray,
) -> np.ndarray:
    """Return each protein residue's closest heavy-atom distance to the ligand."""
    protein = np.asarray(protein_atom_coords, dtype=np.float32)
    ligand = np.asarray(ligand_atom_coords, dtype=np.float32)
    starts = np.asarray(residue_atom_starts, dtype=np.int64)
    ends = np.asarray(residue_atom_ends, dtype=np.int64)
    if protein.ndim != 2 or protein.shape[-1] != 3:
        raise ValueError("Protein atom coordinates must have shape [atoms, 3].")
    if ligand.ndim != 2 or ligand.shape[-1] != 3 or not len(ligand):
        raise ValueError("Ligand atom coordinates must have shape [atoms, 3].")
    if starts.ndim != 1 or ends.ndim != 1 or len(starts) != len(ends) or not len(starts):
        raise ValueError("Residue atom bounds must be non-empty matching vectors.")
    if (starts < 0).any() or (ends <= starts).any() or (ends > len(protein)).any():
        raise ValueError("Residue atom bounds are outside the protein atom array.")
    if not np.isfinite(protein).all() or not np.isfinite(ligand).all():
        raise ValueError("Pocket evidence coordinates must be finite.")
    distances = np.empty(len(starts), dtype=np.float32)
    for residue_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        pairwise = protein[start:end, None, :] - ligand[None, :, :]
        distances[residue_index] = np.linalg.norm(pairwise, axis=-1).min()
    return distances


def select_consensus_pocket_evidence(
    evidence: Sequence[PocketEvidence],
    *,
    closest_residues: int = 500,
) -> tuple[PocketEvidence, float]:
    """Implement Boltz-2 Algorithm 2's maximum mean top-pocket overlap rule."""
    if not evidence:
        raise ValueError("Pocket consensus requires at least one evidence row.")
    if closest_residues <= 0:
        raise ValueError("closest_residues must be positive.")
    protein_keys = {item.protein_key for item in evidence}
    lengths = {len(item.protein_residue_min_distance) for item in evidence}
    if len(protein_keys) != 1 or len(lengths) != 1:
        raise ValueError("Pocket consensus evidence must share one protein and length.")
    size = min(closest_residues, next(iter(lengths)))
    closest = [
        set(np.argsort(item.protein_residue_min_distance, kind="stable")[:size].tolist())
        for item in evidence
    ]
    scores = [
        sum(len(left & right) / size for right in closest) / len(closest)
        for left in closest
    ]
    best_index = int(np.argmax(scores))
    return evidence[best_index], float(scores[best_index])


def select_distogram_consensus_pocket(
    evidence: Sequence[DistogramPocketEvidence],
    *,
    closest_tokens: int = 500,
) -> tuple[DistogramPocketEvidence, float]:
    """Select the deterministic nearest-k-overlap medoid for one target.

    Input order never controls ties: evidence is first ordered by stable system
    identity, and equal mean-overlap scores choose the first such identity.
    """
    if not evidence:
        raise ValueError("Distogram pocket consensus requires evidence.")
    if closest_tokens <= 0:
        raise ValueError("closest_tokens must be positive.")
    ordered = sorted(
        evidence,
        key=lambda item: (item.system_id, item.canonical_smiles),
    )
    protein_keys = {item.protein_key for item in ordered}
    lengths = {len(item.protein_token_min_expected_distance) for item in ordered}
    ligand_keys = {item.canonical_smiles for item in ordered}
    if len(protein_keys) != 1 or len(lengths) != 1:
        raise ValueError("Consensus evidence must share one protein and token length.")
    if len(ligand_keys) != len(ordered):
        raise ValueError("Consensus evidence must use distinct train ligands.")
    size = min(closest_tokens, next(iter(lengths)))
    closest = [
        set(
            np.argsort(item.protein_token_min_expected_distance, kind="stable")[
                :size
            ].tolist()
        )
        for item in ordered
    ]
    scores = [
        sum(len(left & right) / size for right in closest) / len(closest)
        for left in closest
    ]
    best_score = max(scores)
    best_index = next(
        index for index, score in enumerate(scores) if np.isclose(score, best_score)
    )
    return ordered[best_index], float(scores[best_index])


def build_pocket80k_target_consensus(
    evidence: Sequence[Pocket80kEvidence],
    *,
    min_evidence_ligands: int = 10,
    closest_residues: int = 500,
) -> Pocket80kTargetConsensus:
    """Freeze the exact all-profile nearest-500 overlap medoid efficiently."""
    if len(evidence) < min_evidence_ligands:
        raise ValueError(
            f"80k consensus requires at least {min_evidence_ligands} ligands."
        )
    if closest_residues <= 0:
        raise ValueError("80k nearest-residue count must be positive.")
    ordered = sorted(evidence, key=lambda item: (item.system_id, item.canonical_smiles))
    if len({item.canonical_smiles for item in ordered}) != len(ordered):
        raise ValueError("80k consensus evidence must use distinct ligands.")
    proteins = {item.protein_key for item in ordered}
    lengths = {len(item.protein_min_expected_distance) for item in ordered}
    if len(proteins) != 1 or len(lengths) != 1:
        raise ValueError("80k consensus evidence must share one protein and length.")
    length = next(iter(lengths))
    nearest_count = min(closest_residues, length)
    nearest = [
        np.argsort(item.protein_min_expected_distance, kind="stable")[:nearest_count]
        for item in ordered
    ]
    residue_frequency = np.zeros(length, dtype=np.int64)
    for indices in nearest:
        residue_frequency[indices] += 1
    overlaps = [
        float(residue_frequency[indices].sum()) / (len(nearest) * nearest_count)
        for indices in nearest
    ]
    best_overlap = max(overlaps)
    medoid_index = next(
        index for index, score in enumerate(overlaps) if np.isclose(score, best_overlap)
    )
    medoid = ordered[medoid_index]
    return Pocket80kTargetConsensus(
        protein_key=medoid.protein_key,
        protein_residue_min_distance=medoid.protein_min_expected_distance.copy(),
        medoid_system_id=medoid.system_id,
        medoid_canonical_smiles=medoid.canonical_smiles,
        medoid_mean_overlap=float(best_overlap),
    )


class PocketAnnotationLookup:
    """Read validated target-level pocket annotations by protein cache identity."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        expected_contract_version: str = POCKET_ANNOTATION_CONTRACT_V1,
    ) -> None:
        annotations: dict[str, PocketAnnotation] = {}
        for row in rows:
            if str(row["pocket_contract_version"]) != expected_contract_version:
                raise ValueError("Pocket manifest has an unsupported contract version.")
            annotation = PocketAnnotation(
                protein_key=str(row["protein_key"]),
                protein_residue_min_distance=np.asarray(
                    row["protein_residue_min_distance"], dtype=np.float32
                ),
                selected_request_id=str(row["selected_request_id"]),
                evidence_contract_sha256=str(row["evidence_contract_sha256"]),
            )
            if annotation.protein_key in annotations:
                raise ValueError(
                    f"Pocket manifest has duplicate protein key {annotation.protein_key}."
                )
            annotations[annotation.protein_key] = annotation
        if not annotations:
            raise ValueError("Pocket manifest has no annotations.")
        self.annotations = annotations

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        expected_contract_version: str = POCKET_ANNOTATION_CONTRACT_V1,
    ) -> PocketAnnotationLookup:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - runtime environment concern.
            raise RuntimeError("Reading pocket annotations requires pyarrow.") from exc
        columns = [
            "protein_key",
            "pocket_contract_version",
            "protein_residue_min_distance",
            "selected_request_id",
            "evidence_contract_sha256",
        ]
        return cls(
            pq.read_table(path, columns=columns).to_pylist(),
            expected_contract_version=expected_contract_version,
        )

    def get(self, protein_key: str, *, protein_tokens: int) -> PocketAnnotation:
        try:
            annotation = self.annotations[protein_key]
        except KeyError as exc:
            raise KeyError(
                f"Pocket manifest has no annotation for protein key {protein_key}."
            ) from exc
        if len(annotation.protein_residue_min_distance) != protein_tokens:
            raise ValueError(
                "Pocket annotation length does not match cached protein token count: "
                f"{len(annotation.protein_residue_min_distance)} != {protein_tokens}."
            )
        return annotation


class Pocket80kTargetConsensusLookup:
    """Validated lookup for the frozen 80k-only target consensus manifest."""

    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        annotations: dict[str, Pocket80kTargetConsensus] = {}
        checkpoint_sha256: str | None = None
        evidence_lineage_sha256: str | None = None
        for row in rows:
            if (
                str(row["pocket_contract_version"])
                != POCKET80K_TARGET_CONSENSUS_CONTRACT_V1
            ):
                raise ValueError("80k target pocket manifest has the wrong contract.")
            protein_key = str(row["protein_key"])
            annotation = Pocket80kTargetConsensus(
                protein_key=protein_key,
                protein_residue_min_distance=np.asarray(
                    row["protein_residue_min_distance"], dtype=np.float32
                ),
                medoid_system_id=str(row["medoid_system_id"]),
                medoid_canonical_smiles=str(row["medoid_canonical_smiles"]),
                medoid_mean_overlap=float(row["medoid_mean_overlap"]),
            )
            if protein_key in annotations:
                raise ValueError(f"Duplicate 80k target pocket {protein_key}.")
            current_checkpoint = str(row["checkpoint_sha256"])
            current_lineage = str(row["evidence_lineage_sha256"])
            if checkpoint_sha256 is None:
                checkpoint_sha256 = current_checkpoint
                evidence_lineage_sha256 = current_lineage
            elif (
                checkpoint_sha256 != current_checkpoint
                or evidence_lineage_sha256 != current_lineage
            ):
                raise ValueError("80k target pocket manifest mixes lineages.")
            annotations[protein_key] = annotation
        if not annotations:
            raise ValueError("80k target pocket manifest is empty.")
        self.annotations = annotations
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.evidence_lineage_sha256 = str(evidence_lineage_sha256)

    @classmethod
    def from_parquet(cls, path: str | Path) -> Pocket80kTargetConsensusLookup:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Reading 80k target pockets requires pyarrow.") from exc
        return cls(pq.read_table(path).to_pylist())

    def get(self, protein_key: str, *, protein_tokens: int) -> Pocket80kTargetConsensus:
        try:
            annotation = self.annotations[protein_key]
        except KeyError as exc:
            raise KeyError(f"80k target pocket lacks protein {protein_key}.") from exc
        if len(annotation.protein_residue_min_distance) != protein_tokens:
            raise ValueError(
                "80k target pocket length disagrees with protein tokens: "
                f"{len(annotation.protein_residue_min_distance)} != {protein_tokens}."
            )
        return annotation
