"""Fixed-layout, source-balanced assay sampling for affinity ranking."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Sampler

from .data import AffinityBatchIndex


def _seed_for_batch(seed: int, epoch: int, global_batch_index: int) -> int:
    payload = f"{seed}:{epoch}:{global_batch_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class SourceBalancedAssayBatchSampler(Sampler[list[int]]):
    """Draw four SAIR and four BindingDB-residual assays per GPU batch.

    Each yielded list is always 32 positions ordered as eight contiguous assay
    groups of four records.  ``-1`` marks padding in a small assay group and is
    handled by :class:`CachedAffinityFeatureDataset` / ``collate_affinity_batch``.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        num_batches: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        source_origins: tuple[str, str] = ("SAIR", "BindingDB-residual"),
        assays_per_source: int = 4,
        records_per_assay: int = 4,
    ) -> None:
        if num_batches <= 0:
            raise ValueError("num_batches must be positive.")
        if rank < 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size).")
        if assays_per_source <= 0 or records_per_assay <= 0:
            raise ValueError("Assay and record counts must be positive.")
        self.num_batches = num_batches
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.source_origins = source_origins
        self.assays_per_source = assays_per_source
        self.records_per_assay = records_per_assay
        self.epoch = 0
        self.records = list(records)
        grouped: dict[str, dict[str, list[int]]] = {
            origin: defaultdict(list) for origin in source_origins
        }
        for index, record in enumerate(self.records):
            origin = str(record["origin"])
            if origin not in grouped:
                continue
            grouped[origin][str(record["assay_key"])].append(index)
        self.groups = {
            origin: {assay: tuple(indices) for assay, indices in group.items()}
            for origin, group in grouped.items()
        }
        missing = [origin for origin, groups in self.groups.items() if not groups]
        if missing:
            raise ValueError(
                "Source-balanced affinity sampling requires at least one assay for "
                f"every source; missing {missing}."
            )
        self._keys = {
            origin: tuple(sorted(groups)) for origin, groups in self.groups.items()
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def _sample_assay_records(
        self, rng: np.random.Generator, record_indices: Sequence[int]
    ) -> list[int]:
        """Prefer unique ligands while retaining repeats for the Huber term."""
        by_ligand: dict[str, list[int]] = defaultdict(list)
        for index in record_indices:
            record = self.records[index]
            ligand = str(record.get("canonical_smiles", record["record_id"]))
            by_ligand[ligand].append(index)
        ligand_keys = tuple(sorted(by_ligand))
        take_ligands = min(self.records_per_assay, len(ligand_keys))
        chosen_ligands = rng.choice(
            len(ligand_keys), size=take_ligands, replace=False
        ).tolist()
        selected = [
            int(rng.choice(by_ligand[ligand_keys[int(choice)]]))
            for choice in chosen_ligands
        ]
        if len(selected) < self.records_per_assay:
            remaining = [index for index in record_indices if index not in selected]
            take_repeats = min(self.records_per_assay - len(selected), len(remaining))
            if take_repeats:
                selected.extend(
                    int(index)
                    for index in rng.choice(remaining, size=take_repeats, replace=False)
                )
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        base_batch = self.epoch * self.num_batches * self.world_size
        for local_batch in range(self.num_batches):
            global_batch = base_batch + local_batch * self.world_size + self.rank
            rng = np.random.default_rng(
                _seed_for_batch(self.seed, self.epoch, global_batch)
            )
            batch: list[int] = []
            for origin in self.source_origins:
                keys = self._keys[origin]
                chosen_assays = rng.integers(0, len(keys), size=self.assays_per_source)
                for chosen in chosen_assays:
                    record_indices = self.groups[origin][keys[int(chosen)]]
                    sampled = self._sample_assay_records(rng, record_indices)
                    take = len(sampled)
                    batch.extend(sampled)
                    batch.extend([-1] * (self.records_per_assay - take))
            yield batch


class FixedValidSourceBalancedBucketSampler(Sampler[list[int]]):
    """Draw a fixed number of real labels from each source in one shape bucket.

    SAIR examples reserve rankable assay groups first and then backfill with
    other valid SAIR labels.  BindingDB source-record singletons occupy their
    own slots/groups, so they remain regression-only without wasting tensor
    positions on padding.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        num_batches: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        source_origins: tuple[str, str] = ("SAIR", "BindingDB-residual"),
        labels_per_source: int = 16,
        rankable_assays_per_batch: int = 4,
        records_per_rankable_assay: int = 4,
    ) -> None:
        if num_batches <= 0:
            raise ValueError("num_batches must be positive.")
        if rank < 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size).")
        if labels_per_source <= 0:
            raise ValueError("labels_per_source must be positive.")
        if rankable_assays_per_batch <= 0 or records_per_rankable_assay <= 0:
            raise ValueError("Rankable assay settings must be positive.")
        self.records = list(records)
        self.num_batches = num_batches
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.source_origins = source_origins
        self.labels_per_source = labels_per_source
        self.rankable_assays_per_batch = rankable_assays_per_batch
        self.records_per_rankable_assay = records_per_rankable_assay
        self.epoch = 0

        self._source_indices: dict[int, dict[str, list[int]]] = defaultdict(
            lambda: {origin: [] for origin in source_origins}
        )
        grouped: dict[int, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: {origin: defaultdict(list) for origin in source_origins}
        )
        for index, record in enumerate(self.records):
            try:
                bucket = int(record["shape_bucket"])
            except KeyError as exc:
                raise KeyError(
                    "Fixed-valid affinity sampling requires manifest shape_bucket "
                    "metadata."
                ) from exc
            origin = str(record["origin"])
            if origin not in source_origins:
                continue
            self._source_indices[bucket][origin].append(index)
            grouped[bucket][origin][str(record["assay_key"])].append(index)

        self._rankable_assays: dict[int, tuple[str, ...]] = {}
        eligible: list[int] = []
        for bucket, by_origin in self._source_indices.items():
            if any(
                len(by_origin[origin]) < labels_per_source for origin in source_origins
            ):
                continue
            rankable = tuple(
                sorted(
                    assay
                    for assay, indices in grouped[bucket]["SAIR"].items()
                    if len(
                        {
                            str(self.records[index]["canonical_smiles"])
                            for index in indices
                        }
                    )
                    >= 2
                )
            )
            if not rankable:
                continue
            eligible.append(bucket)
            self._rankable_assays[bucket] = rankable
        self._eligible_buckets = tuple(sorted(eligible))
        if not self._eligible_buckets:
            raise ValueError(
                "No shape bucket can supply the requested valid labels per source "
                "and one rankable SAIR assay."
            )
        self._groups = grouped

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def _sample_assay(
        self,
        rng: np.random.Generator,
        indices: Sequence[int],
        *,
        limit: int,
    ) -> list[int]:
        by_ligand: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            by_ligand[str(self.records[index]["canonical_smiles"])].append(index)
        ligand_keys = tuple(sorted(by_ligand))
        take_ligands = min(limit, len(ligand_keys))
        selected_ligands = rng.choice(
            len(ligand_keys), size=take_ligands, replace=False
        ).tolist()
        selected = [
            int(rng.choice(by_ligand[ligand_keys[int(choice)]]))
            for choice in selected_ligands
        ]
        if len(selected) < limit:
            remaining = [index for index in indices if index not in selected]
            extra = min(limit - len(selected), len(remaining))
            if extra:
                selected.extend(
                    int(index)
                    for index in rng.choice(remaining, size=extra, replace=False)
                )
        return selected

    @staticmethod
    def _extend_to_fixed_count(
        rng: np.random.Generator,
        *,
        selected: list[int],
        candidates: Sequence[int],
        count: int,
    ) -> list[int]:
        available = [index for index in candidates if index not in selected]
        needed = count - len(selected)
        if needed < 0:
            return selected[:count]
        if len(available) < needed:
            raise ValueError("Bucket does not contain enough distinct valid labels.")
        if needed:
            selected.extend(
                int(index) for index in rng.choice(available, size=needed, replace=False)
            )
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        base_batch = self.epoch * self.num_batches * self.world_size
        for local_batch in range(self.num_batches):
            global_batch = base_batch + local_batch * self.world_size + self.rank
            rng = np.random.default_rng(
                _seed_for_batch(self.seed, self.epoch, global_batch)
            )
            bucket = int(rng.choice(self._eligible_buckets))
            sair_candidates = self._source_indices[bucket]["SAIR"]
            bdb_candidates = self._source_indices[bucket]["BindingDB-residual"]
            rankable = self._rankable_assays[bucket]
            rankable_count = min(self.rankable_assays_per_batch, len(rankable))
            selected_assays = rng.choice(
                len(rankable), size=rankable_count, replace=False
            ).tolist()
            sair_selected: list[int] = []
            for choice in selected_assays:
                assay = rankable[int(choice)]
                sair_selected.extend(
                    self._sample_assay(
                        rng,
                        self._groups[bucket]["SAIR"][assay],
                        limit=self.records_per_rankable_assay,
                    )
                )
            sair_selected = list(dict.fromkeys(sair_selected))
            sair_selected = self._extend_to_fixed_count(
                rng,
                selected=sair_selected,
                candidates=sair_candidates,
                count=self.labels_per_source,
            )
            bdb_selected = rng.choice(
                bdb_candidates,
                size=self.labels_per_source,
                replace=False,
            ).tolist()
            yield sair_selected + [int(index) for index in bdb_selected]


class FixedShapeValidationBatchSampler(Sampler[list[int]]):
    """Yield sequential validation batches with one static token bucket each."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        batch_size: int = 32,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if rank < 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size).")
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            try:
                bucket = int(record["shape_bucket"])
            except KeyError as exc:
                raise KeyError(
                    "Fixed-shape validation requires manifest shape_bucket metadata."
                ) from exc
            grouped[bucket].append(index)
        self._batches: list[list[int]] = []
        for bucket in sorted(grouped):
            indices = grouped[bucket][rank::world_size]
            for offset in range(0, len(indices), batch_size):
                batch = indices[offset : offset + batch_size]
                batch.extend([-1] * (batch_size - len(batch)))
                self._batches.append(batch)

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches


class ActivityCliffBucketSampler(Sampler[list[AffinityBatchIndex]]):
    """Boltz-2 Algorithm 4 packed into one fixed physical batch.

    Distinct continuous assays are sampled without replacement with probability
    proportional to their global IQR.  Each assay contributes one logical group
    of records.  The remaining regression-only slots are always drawn from
    BindingDB-residual records in the same crop bucket.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        num_batches: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        batch_size: int = 32,
        logical_group_size: int = 5,
        logical_groups_per_batch: int = 6,
        singleton_regression_slots: int = 2,
    ) -> None:
        if num_batches <= 0:
            raise ValueError("num_batches must be positive.")
        if rank < 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size).")
        if logical_group_size <= 1 or logical_groups_per_batch <= 0:
            raise ValueError("Activity-cliff group settings must be positive.")
        if singleton_regression_slots < 0:
            raise ValueError("singleton_regression_slots must be non-negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if (
            logical_group_size * logical_groups_per_batch + singleton_regression_slots
            != batch_size
        ):
            raise ValueError(
                "Activity-cliff groups and BindingDB slots must exactly fill "
                "the configured physical batch."
            )
        self.records = list(records)
        self.num_batches = num_batches
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.logical_group_size = logical_group_size
        self.logical_groups_per_batch = logical_groups_per_batch
        self.singleton_regression_slots = singleton_regression_slots
        self.epoch = 0

        assay_rows: dict[str, list[int]] = defaultdict(list)
        bucket_assay_rows: dict[int, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        singleton_primary: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            try:
                bucket = int(record["shape_bucket"])
            except KeyError as exc:
                raise KeyError(
                    "Activity-cliff sampling requires manifest shape_bucket metadata."
                ) from exc
            if self._ranking_eligible(record):
                assay = str(record["assay_key"])
                assay_rows[assay].append(index)
                bucket_assay_rows[bucket][assay].append(index)
            elif str(record.get("origin")) == "BindingDB-residual":
                singleton_primary[bucket].append(index)

        self.assay_iqr: dict[str, float] = {}
        for assay, indices in assay_rows.items():
            by_ligand: dict[str, list[float]] = defaultdict(list)
            for index in indices:
                by_ligand[str(self.records[index]["canonical_smiles"])].append(
                    float(self.records[index]["p_activity"])
                )
            values = np.asarray(
                [np.mean(values) for _, values in sorted(by_ligand.items())],
                dtype=np.float64,
            )
            if len(values) < 2:
                self.assay_iqr[assay] = 0.0
            else:
                self.assay_iqr[assay] = float(
                    np.quantile(values, 0.75) - np.quantile(values, 0.25)
                )

        self._activity_strata: dict[int, tuple[str, ...]] = {}
        self._activity_weights: dict[int, np.ndarray] = {}
        self._singleton_primary: dict[int, list[int]] = defaultdict(list)
        eligible: list[int] = []
        for bucket, by_assay in bucket_assay_rows.items():
            strata = tuple(
                sorted(
                    assay
                    for assay, indices in by_assay.items()
                    if len(indices) >= logical_group_size and self.assay_iqr[assay] > 0.0
                )
            )
            self._singleton_primary[bucket] = list(
                dict.fromkeys(singleton_primary[bucket])
            )
            if (
                len(strata) < logical_groups_per_batch
                or len(self._singleton_primary[bucket]) < singleton_regression_slots
            ):
                continue
            weights = np.asarray([self.assay_iqr[assay] for assay in strata])
            self._activity_strata[bucket] = strata
            self._activity_weights[bucket] = weights / weights.sum()
            eligible.append(bucket)
        self._eligible_buckets = tuple(sorted(eligible))
        if not self._eligible_buckets:
            raise ValueError(
                "No crop bucket has activity-cliff groups and the requested "
                "regression-only singleton slots."
            )
        self._bucket_assay_rows = bucket_assay_rows

    @staticmethod
    def _ranking_eligible(record: Mapping[str, object]) -> bool:
        """Use explicit corpus eligibility and never infer BDB ranking pairs."""
        value = record.get("ranking_eligible")
        if value is not None:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes"}
            return bool(value)
        if str(record.get("origin")) == "BindingDB-residual":
            return False
        return "assay_id=singleton:" not in str(record.get("assay_key", ""))

    def _sample_group_records(
        self,
        rng: np.random.Generator,
        indices: Sequence[int],
    ) -> list[int]:
        """Take five unique ligands first, then raw replicates if necessary."""
        by_ligand: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            by_ligand[str(self.records[index]["canonical_smiles"])].append(index)
        ligands = tuple(sorted(by_ligand))
        take_ligands = min(self.logical_group_size, len(ligands))
        selected_ligands = rng.choice(
            len(ligands), size=take_ligands, replace=False
        ).tolist()
        selected = [
            int(rng.choice(by_ligand[ligands[int(choice)]]))
            for choice in selected_ligands
        ]
        if len(selected) < self.logical_group_size:
            remaining = [index for index in indices if index not in selected]
            needed = self.logical_group_size - len(selected)
            if len(remaining) < needed:
                raise ValueError(
                    "Activity-cliff assay does not contain the requested records."
                )
            selected.extend(
                int(index) for index in rng.choice(remaining, size=needed, replace=False)
            )
        return selected

    def _sample_singletons(self, rng: np.random.Generator, bucket: int) -> list[int]:
        primary = self._singleton_primary[bucket]
        if self.singleton_regression_slots == 0:
            return []
        return [
            int(index)
            for index in rng.choice(
                primary,
                size=self.singleton_regression_slots,
                replace=False,
            )
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[AffinityBatchIndex]]:
        base_batch = self.epoch * self.num_batches * self.world_size
        for local_batch in range(self.num_batches):
            global_batch = base_batch + local_batch * self.world_size + self.rank
            rng = np.random.default_rng(
                _seed_for_batch(self.seed, self.epoch, global_batch)
            )
            bucket = int(rng.choice(self._eligible_buckets))
            strata = self._activity_strata[bucket]
            weights = self._activity_weights[bucket]
            batch: list[AffinityBatchIndex] = []
            assay_indices = rng.choice(
                len(strata),
                size=self.logical_groups_per_batch,
                replace=False,
                p=weights,
            )
            for group_id, assay_index in enumerate(assay_indices.tolist()):
                assay = strata[assay_index]
                records = self._bucket_assay_rows[bucket][assay]
                selected = self._sample_group_records(rng, records)
                batch.extend(
                    AffinityBatchIndex(int(index), group_id) for index in selected
                )
            singleton_selected = self._sample_singletons(rng, bucket)
            batch.extend(
                AffinityBatchIndex(int(index), self.logical_groups_per_batch + offset)
                for offset, index in enumerate(singleton_selected)
            )
            yield batch
