"""Reusable ligand-apo cache primitives for CCD and ETKDGv3 conformers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import FeatureCacheReader, FeatureCacheWriter
from .data import ligand_key


def ligand_etkdg_seed(canonical_smiles: str) -> int:
    """Derive a reproducible ETKDG seed from normalized ligand identity."""
    return int.from_bytes(
        hashlib.blake2b(canonical_smiles.encode(), digest_size=8).digest(), "little"
    )


@dataclass(frozen=True, kw_only=True)
class LigandApo:
    """One source-order ligand conformer with explicit provenance."""

    canonical_smiles: str
    coords: np.ndarray
    structure_sha256: str
    source: str = "etkdg_v3"
    source_id: str | None = None

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[-1] != 3:
            raise ValueError(
                "Ligand apo coordinates must have shape [atoms, 3], got "
                f"{self.coords.shape}."
            )
        if not np.isfinite(self.coords).all():
            raise ValueError("Ligand apo coordinates must be finite.")
        if self.coords.dtype != np.float32:
            object.__setattr__(self, "coords", self.coords.astype(np.float32))


def generate_ligand_etkdg(canonical_smiles: str) -> LigandApo:
    """Generate the one deterministic ETKDGv3 apo conformer for a ligand."""
    from kfold.data.types.ccd import Component

    component = Component.from_smiles("LIG", canonical_smiles)
    coords = component.get_conformer(
        "etkdg", np.random.default_rng(ligand_etkdg_seed(canonical_smiles))
    )
    if coords is None or not np.isfinite(coords).all():
        raise ValueError("Unable to generate a finite ligand ETKDGv3 conformer.")
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != (component.num_atoms, 3):
        raise ValueError(
            "ETKDG ligand coordinates do not match canonical component atoms: "
            f"{coords.shape} versus {(component.num_atoms, 3)}."
        )
    return LigandApo(
        canonical_smiles=canonical_smiles,
        coords=coords,
        structure_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
        source_id="generated_etkdg_v3",
    )


class LigandApoLookup:
    """Read precomputed CCD/ETKDG ligand apo coordinates from sharded LMDB."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]] | Mapping[str, Mapping[str, object]],
        *,
        cache_root: str,
        copy_rows: bool = True,
    ) -> None:
        if isinstance(rows, Mapping):
            self.rows = (
                {str(key): row for key, row in rows.items()} if copy_rows else rows
            )
        else:
            self.rows = {str(row["ligand_key"]): row for row in rows}
            if len(self.rows) != len(rows):
                raise ValueError("Ligand apo index contains duplicate ligand keys.")
        self.reader = FeatureCacheReader(cache_root)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        cache_root: str,
    ) -> LigandApoLookup:
        import pyarrow.parquet as pq

        return cls(pq.read_table(path).to_pylist(), cache_root=cache_root)

    def get(self, canonical_smiles: str) -> LigandApo:
        key = ligand_key(canonical_smiles=canonical_smiles)
        try:
            row = self.rows[key]
        except KeyError as exc:
            raise KeyError(f"No precomputed ligand apo for key {key}.") from exc
        if str(row["canonical_smiles"]) != canonical_smiles:
            raise ValueError("Ligand apo index key does not match canonical SMILES.")
        arrays = self.reader.get(row)
        source = row.get("ligand_apo_source", row.get("etkdg_source"))
        structure_sha256 = row.get(
            "ligand_apo_structure_sha256", row.get("etkdg_structure_sha256")
        )
        if source is None or structure_sha256 is None:
            raise KeyError("Ligand apo index lacks source or structure digest fields.")
        return LigandApo(
            canonical_smiles=canonical_smiles,
            coords=np.asarray(arrays["etkdg_coords"], dtype=np.float32),
            structure_sha256=str(structure_sha256),
            source=str(source),
            source_id=(
                str(row["ligand_apo_source_id"])
                if row.get("ligand_apo_source_id") is not None
                else None
            ),
        )

    def validate_cache(self) -> None:
        """Ensure every manifest-referenced ligand shard is present before caching."""
        self.reader.validate_shards(self.rows.values())

    def close(self) -> None:
        self.reader.close()


def store_ligand_apo(
    writer: FeatureCacheWriter,
    *,
    apo: LigandApo,
) -> dict[str, object]:
    """Store one resolved ligand apo in the generic sharded cache."""
    key = ligand_key(canonical_smiles=apo.canonical_smiles)
    entry = writer.put(
        system_id=key,
        arrays={"etkdg_coords": apo.coords},
        protein_tokens=0,
        ligand_tokens=len(apo.coords),
        crop_tokens=len(apo.coords),
        ligand_protein_entropy=0.0,
    )
    return {
        "ligand_key": key,
        "canonical_smiles": apo.canonical_smiles,
        "ligand_apo_source": apo.source,
        "ligand_apo_source_id": apo.source_id,
        "ligand_apo_structure_sha256": apo.structure_sha256,
        "cache_shard": entry.shard,
        "cache_key": entry.key,
        "compressed_bytes": entry.compressed_bytes,
        "raw_bytes": entry.raw_bytes,
        "ligand_atoms": len(apo.coords),
    }
