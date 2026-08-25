"""Lightning data module for cached assay-balanced affinity head training."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from .cache import AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1, FeatureCacheReader
from .crop import (
    head_crop_token_count,
    shape_bucket_for_tokens,
)
from .dataset import (
    CachedAffinityFeatureDataset,
    collate_affinity_batch,
    collate_affinity_eval,
    materialize_target_compact_batch,
)
from .pocket import (
    AFFINITY_CROP_CONTRACT_V2,
    DISTOGRAM_POCKET_ANNOTATION_CONTRACT_V1,
    POCKET80K_TARGET_CROP_CONTRACT_V1,
    POCKET_ANNOTATION_CONTRACT_V1,
    PocketAnnotationLookup,
)
from .pocket import (
    sha256_file as pocket_manifest_sha256,
)
from .sampler import (
    ActivityCliffBucketSampler,
    FixedShapeValidationBatchSampler,
    FixedValidSourceBalancedBucketSampler,
)
from .schema import (
    AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
    AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1,
    AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1,
    AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
    AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1,
    AFFINITY_CACHE_SCHEMAS,
    AFFINITY_COMPACT_SCHEMAS,
    AFFINITY_DISTOGRAM_CROP_CONTRACT_V2,
    AFFINITY_DISTOGRAM_TARGET_CONSENSUS_CONTRACT_V1,
)


def configure_affinity_data_worker(_: int) -> None:
    """Avoid file-descriptor shared-memory cleanup races in spawned workers."""
    torch.multiprocessing.set_sharing_strategy("file_system")


def affinity_contract_value_matches(actual: object, expected: object) -> bool:
    """Compare serialized contract scalars without rejecting float32 roundoff."""
    if isinstance(expected, float):
        return actual is not None and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-6
        )
    return actual == expected


def read_manifest_rows(path: str | Path) -> list[dict[str, object]]:
    """Read the cache-joined training manifest only when the data module starts."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - execution environment concern.
        raise RuntimeError(
            "Reading the affinity Parquet manifest requires pyarrow. Install it in "
            "the preprocessing/training environment."
        ) from exc
    return pq.read_table(path).to_pylist()


def attach_shape_buckets(
    rows: list[dict[str, object]],
    *,
    shape_buckets: tuple[int, ...],
    max_crop_tokens: int,
    max_protein_crop_tokens: int,
) -> list[dict[str, object]]:
    """Attach deterministic head crop/bucket metadata from cache token counts."""
    attached: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        try:
            head_crop_tokens = (
                int(row["crop_tokens"])
                if row.get("cache_schema") in AFFINITY_COMPACT_SCHEMAS
                else head_crop_token_count(
                    protein_tokens=int(row["protein_tokens"]),
                    ligand_tokens=int(row["ligand_tokens"]),
                    max_tokens=max_crop_tokens,
                    max_protein_tokens=max_protein_crop_tokens,
                )
            )
        except KeyError as exc:
            raise KeyError(
                "Fixed-shape affinity batching requires cache-backed protein_tokens "
                "and ligand_tokens in the manifest."
            ) from exc
        row["head_crop_tokens"] = head_crop_tokens
        row["shape_bucket"] = shape_bucket_for_tokens(
            head_crop_tokens,
            buckets=shape_buckets,
        )
        attached.append(row)
    return attached


class AffinityDataModule(pl.LightningDataModule):
    """Boltz2 continuous-affinity training with static crop-shape buckets."""

    def __init__(
        self,
        *,
        manifest_path: str,
        cache_root: str,
        train_batches_per_epoch: int,
        seed: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        cache_schema: str = AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
        cache_encoding: str | None = None,
        max_crop_tokens: int = 256,
        max_protein_crop_tokens: int = 200,
        shape_buckets: tuple[int, ...] = (256,),
        sampling_mode: str = "boltz2_activity_cliff",
        crop_mode: str = "boltz2_pocket",
        pocket_manifest_path: str | None = None,
        pocket_neighborhood_size: int = 10,
        crop_contract_version: str = AFFINITY_CROP_CONTRACT_V2,
        distogram_pocket_distance_cutoff: float = 15.0,
        distogram_use_entropy_tiebreak: bool = True,
        distogram_entropy_gate: float = 0.7,
        distogram_strong_binder_p_activity: float = 6.0,
        activity_group_size: int = 5,
        activity_groups_per_batch: int = 6,
        singleton_regression_slots: int = 2,
        labels_per_source: int = 16,
        rankable_assays_per_batch: int = 4,
        records_per_rankable_assay: int = 4,
    ) -> None:
        super().__init__()
        if val_batch_size > 32:
            raise ValueError("val_batch_size cannot exceed the fixed 32-slot layout.")
        if sampling_mode not in {"boltz2_activity_cliff", "legacy_fixed_valid"}:
            raise ValueError(f"Unsupported affinity sampling mode: {sampling_mode!r}.")
        if crop_mode not in {
            "boltz2_pocket",
            "distogram_predicted_contact",
            "distogram_target_consensus",
            "distogram_target_consensus_compact",
            "pocket80k_target_compact",
            "pocket80k_query_adaptive_80_20",
            "legacy_predicted_contact",
        }:
            raise ValueError(f"Unsupported affinity crop mode: {crop_mode!r}.")
        if train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive.")
        if (
            sampling_mode == "legacy_fixed_valid"
            and labels_per_source * 2 != train_batch_size
        ):
            raise ValueError(
                "Fixed-valid affinity labels must exactly fill train_batch_size."
            )
        if (
            sampling_mode == "boltz2_activity_cliff"
            and activity_group_size * activity_groups_per_batch
            + singleton_regression_slots
            != train_batch_size
        ):
            raise ValueError(
                "Boltz2 activity-cliff layout must exactly fill train_batch_size."
            )
        if crop_mode == "boltz2_pocket" and pocket_manifest_path is None:
            raise ValueError("Boltz2 pocket crop requires pocket_manifest_path.")
        if (
            crop_mode
            in {
                "distogram_target_consensus",
                "distogram_target_consensus_compact",
            }
            and pocket_manifest_path is None
        ):
            raise ValueError("Distogram target consensus requires pocket_manifest_path.")
        if pocket_neighborhood_size <= 0:
            raise ValueError("pocket_neighborhood_size must be positive.")
        if distogram_pocket_distance_cutoff <= 0:
            raise ValueError("distogram_pocket_distance_cutoff must be positive.")
        if not 0 <= distogram_entropy_gate <= 1:
            raise ValueError("distogram_entropy_gate must be in [0, 1].")
        if cache_schema not in AFFINITY_CACHE_SCHEMAS:
            raise ValueError(
                "Affinity training requires a supported physical cache, "
                f"got {cache_schema!r}."
            )
        if (
            crop_mode
            in {
                "distogram_predicted_contact",
                "distogram_target_consensus",
                "legacy_predicted_contact",
            }
            and cache_schema != AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1
        ):
            raise ValueError(
                "Predicted-distogram training requires the v1 physical cache."
            )
        if (
            crop_mode == "distogram_target_consensus_compact"
            and cache_schema != AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1
        ):
            raise ValueError("Compact target consensus requires its compact schema.")
        if (
            cache_schema == AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1
            and crop_mode != "pocket80k_target_compact"
        ):
            raise ValueError("Compact pocket cache is already cropped.")
        if crop_mode == "pocket80k_target_compact" and cache_schema not in {
            AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1,
            AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
        }:
            raise ValueError(
                "80k target-consensus crop requires its final compact schema."
            )
        if cache_schema == AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2:
            if crop_mode != "pocket80k_target_compact":
                raise ValueError("Direct compact pocket cache is already cropped.")
            if cache_encoding != AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1:
                raise ValueError(
                    "Direct compact affinity training requires raw arraypack data."
                )
        if (
            crop_mode == "pocket80k_query_adaptive_80_20"
            and cache_schema != AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1
        ):
            raise ValueError("80/20 ablation requires its final compact schema.")
        if (
            cache_schema == AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1
            and crop_mode != "pocket80k_query_adaptive_80_20"
        ):
            raise ValueError("80/20 ablation cache is already cropped.")
        if (
            crop_mode == "boltz2_pocket"
            and crop_contract_version != AFFINITY_CROP_CONTRACT_V2
        ):
            raise ValueError("Boltz2 pocket training requires the v2 crop contract.")
        if (
            crop_mode == "distogram_predicted_contact"
            and crop_contract_version != AFFINITY_DISTOGRAM_CROP_CONTRACT_V2
        ):
            raise ValueError(
                "Predicted-distogram training requires its explicit crop contract."
            )
        if (
            crop_mode
            in {"distogram_target_consensus", "distogram_target_consensus_compact"}
            and crop_contract_version != AFFINITY_DISTOGRAM_TARGET_CONSENSUS_CONTRACT_V1
        ):
            raise ValueError(
                "Distogram target consensus requires its explicit crop contract."
            )
        if (
            crop_mode == "pocket80k_target_compact"
            and crop_contract_version != POCKET80K_TARGET_CROP_CONTRACT_V1
        ):
            raise ValueError(
                "80k target-consensus training requires its frozen crop contract."
            )
        if (
            crop_mode == "pocket80k_query_adaptive_80_20"
            and crop_contract_version != "query_adaptive_80_20_v1"
        ):
            raise ValueError("80/20 ablation requires its frozen crop contract.")
        if (
            crop_mode == "distogram_predicted_contact"
            and pocket_manifest_path is not None
        ):
            raise ValueError("Predicted-distogram crop does not use a pocket manifest.")
        if (
            crop_mode
            in {
                "distogram_target_consensus",
                "distogram_target_consensus_compact",
            }
            and pocket_neighborhood_size != 10
        ):
            raise ValueError(
                "Distogram target consensus requires a 10-token neighborhood."
            )
        if crop_mode in {
            "distogram_target_consensus",
            "distogram_target_consensus_compact",
            "pocket80k_target_compact",
            "pocket80k_query_adaptive_80_20",
        } and (
            max_crop_tokens != 256
            or max_protein_crop_tokens != 200
            or distogram_pocket_distance_cutoff != 15.0
            or distogram_entropy_gate != 0.7
            or distogram_strong_binder_p_activity != 6.0
        ):
            raise ValueError(
                "Distogram target consensus requires the v1 256/200, 15 A, "
                "HLP 0.7, and p=6 contract."
            )
        if (
            sampling_mode == "legacy_fixed_valid"
            and crop_mode != "legacy_predicted_contact"
        ):
            raise ValueError(
                "The legacy sampler requires the preserved legacy crop mode."
            )
        if sampling_mode == "boltz2_activity_cliff" and crop_mode not in {
            "boltz2_pocket",
            "distogram_predicted_contact",
            "distogram_target_consensus",
            "distogram_target_consensus_compact",
            "pocket80k_target_compact",
            "pocket80k_query_adaptive_80_20",
        }:
            raise ValueError(
                "The Boltz2 activity-cliff sampler requires a Boltz2 pocket or "
                "predicted-distogram crop."
            )
        self.manifest_path = manifest_path
        self.cache_root = cache_root
        self.train_batches_per_epoch = train_batches_per_epoch
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.cache_schema = cache_schema
        self.cache_encoding = cache_encoding
        self.max_crop_tokens = max_crop_tokens
        self.max_protein_crop_tokens = max_protein_crop_tokens
        self.shape_buckets = tuple(int(bucket) for bucket in shape_buckets)
        self.sampling_mode = sampling_mode
        self.crop_mode = crop_mode
        self.pocket_manifest_path = pocket_manifest_path
        self.pocket_neighborhood_size = pocket_neighborhood_size
        self.crop_contract_version = crop_contract_version
        self.distogram_pocket_distance_cutoff = distogram_pocket_distance_cutoff
        self.distogram_use_entropy_tiebreak = distogram_use_entropy_tiebreak
        self.distogram_entropy_gate = distogram_entropy_gate
        self.distogram_strong_binder_p_activity = distogram_strong_binder_p_activity
        self.activity_group_size = activity_group_size
        self.activity_groups_per_batch = activity_groups_per_batch
        self.singleton_regression_slots = singleton_regression_slots
        self.labels_per_source = labels_per_source
        self.rankable_assays_per_batch = rankable_assays_per_batch
        self.records_per_rankable_assay = records_per_rankable_assay
        self._train_dataset: CachedAffinityFeatureDataset | None = None
        self._val_dataset: CachedAffinityFeatureDataset | None = None
        self._train_sampler: (
            FixedValidSourceBalancedBucketSampler | ActivityCliffBucketSampler | None
        ) = None
        self._pocket_annotations: PocketAnnotationLookup | None = None
        self._cache_reader: FeatureCacheReader | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit", "validate"}:
            return
        rows = read_manifest_rows(self.manifest_path)
        usable = [
            row
            for row in rows
            if row.get("cache_shard") is not None
            and row.get("cache_key") is not None
            and bool(row.get("eligible_for_training", True))
        ]
        schemas = {str(row.get("cache_schema")) for row in usable}
        if schemas != {self.cache_schema}:
            raise ValueError(
                "Affinity data module requires exactly one full-cross cache schema: "
                f"expected {self.cache_schema!r}, found {sorted(schemas)!r}."
            )
        if self.cache_encoding is not None:
            encodings = {str(row.get("cache_encoding")) for row in usable}
            if encodings != {self.cache_encoding}:
                raise ValueError(
                    "Affinity data module requires exactly one physical encoding: "
                    f"expected {self.cache_encoding!r}, found {sorted(encodings)!r}."
                )
        bucketed = attach_shape_buckets(
            usable,
            shape_buckets=self.shape_buckets,
            max_crop_tokens=self.max_crop_tokens,
            max_protein_crop_tokens=self.max_protein_crop_tokens,
        )
        train_rows = [row for row in bucketed if row.get("split") == "train"]
        val_rows = [row for row in bucketed if row.get("split") == "val"]
        if not train_rows:
            raise ValueError("The affinity manifest has no cache-backed train records.")
        if not val_rows:
            raise ValueError(
                "The affinity manifest has no cache-backed validation records."
            )
        if self.crop_mode in {
            "pocket80k_target_compact",
            "pocket80k_query_adaptive_80_20",
        }:
            expected = {
                "crop_mode": self.crop_mode,
                "crop_contract_version": (
                    "query_adaptive_80_20_v1"
                    if self.crop_mode == "pocket80k_query_adaptive_80_20"
                    else POCKET80K_TARGET_CROP_CONTRACT_V1
                ),
                "affinity_crop_max_tokens": 256,
                "affinity_crop_max_protein_tokens": 200,
                "pocket_neighborhood_size": 10,
            }
            for row in bucketed:
                for name, value in expected.items():
                    if row.get(name) != value:
                        raise ValueError(
                            f"80k target-consensus cache contract mismatch for {name!r}."
                        )
                if row.get("pocket_manifest_sha256") is None:
                    raise ValueError(
                        "80k target-consensus row lacks its frozen pocket digest."
                    )
                if row.get("checkpoint_sha256") is None:
                    raise ValueError(
                        "80k target-consensus row lacks its 80k checkpoint digest."
                    )
        elif self.crop_mode in {
            "boltz2_pocket",
            "distogram_target_consensus",
            "distogram_target_consensus_compact",
        }:
            assert self.pocket_manifest_path is not None
            manifest_sha = pocket_manifest_sha256(self.pocket_manifest_path)
            expected_mode = self.crop_mode
            expected_pocket_contract = (
                DISTOGRAM_POCKET_ANNOTATION_CONTRACT_V1
                if self.crop_mode
                in {"distogram_target_consensus", "distogram_target_consensus_compact"}
                else POCKET_ANNOTATION_CONTRACT_V1
            )
            for row in bucketed:
                if row.get("crop_mode") != expected_mode:
                    raise ValueError(
                        "Target-level pocket training rows have the wrong crop mode."
                    )
                if row.get("pocket_manifest_sha256") != manifest_sha:
                    raise ValueError(
                        "Cache rows do not match the supplied pocket annotation manifest."
                    )
                if row.get("pocket_contract_version") != expected_pocket_contract:
                    raise ValueError(
                        "Cache rows do not declare the configured pocket contract."
                    )
                if row.get("crop_contract_version") != self.crop_contract_version:
                    raise ValueError(
                        "Training rows do not match the configured crop contract."
                    )
                if row.get("pocket_neighborhood_size") != self.pocket_neighborhood_size:
                    raise ValueError(
                        "Cache rows do not match the configured pocket neighborhood."
                    )
                if row.get("affinity_crop_max_tokens") != self.max_crop_tokens:
                    raise ValueError("Cache rows do not match max_crop_tokens.")
                if (
                    row.get("affinity_crop_max_protein_tokens")
                    != self.max_protein_crop_tokens
                ):
                    raise ValueError("Cache rows do not match max_protein_crop_tokens.")
                if self.crop_mode in {
                    "distogram_target_consensus",
                    "distogram_target_consensus_compact",
                }:
                    expected_distogram = {
                        "distogram_pocket_distance_cutoff": (
                            self.distogram_pocket_distance_cutoff
                        ),
                        "distogram_entropy_gate": self.distogram_entropy_gate,
                        "distogram_strong_binder_p_activity": (
                            self.distogram_strong_binder_p_activity
                        ),
                        "distogram_entropy_scope": (
                            "protein_min_expected_distance_lt_15A_pl_pairs"
                        ),
                    }
                    for name, value in expected_distogram.items():
                        if not affinity_contract_value_matches(row.get(name), value):
                            raise ValueError(
                                "Target-consensus cache contract mismatch for "
                                f"{name!r}: expected {value!r}, "
                                f"got {row.get(name)!r}."
                            )
            if self.crop_mode != "distogram_target_consensus_compact":
                self._pocket_annotations = PocketAnnotationLookup.from_parquet(
                    self.pocket_manifest_path,
                    expected_contract_version=expected_pocket_contract,
                )
                for row in bucketed:
                    self._pocket_annotations.get(
                        str(row["protein_key"]),
                        protein_tokens=int(row["protein_tokens"]),
                    )
        elif self.crop_mode == "distogram_predicted_contact":
            expected = {
                "crop_mode": self.crop_mode,
                "crop_contract_version": self.crop_contract_version,
                "affinity_crop_max_tokens": self.max_crop_tokens,
                "affinity_crop_max_protein_tokens": self.max_protein_crop_tokens,
                "distogram_pocket_distance_cutoff": (
                    self.distogram_pocket_distance_cutoff
                ),
                "distogram_use_entropy_tiebreak": (self.distogram_use_entropy_tiebreak),
                "distogram_entropy_gate": self.distogram_entropy_gate,
                "distogram_strong_binder_p_activity": (
                    self.distogram_strong_binder_p_activity
                ),
            }
            for row in bucketed:
                for name, value in expected.items():
                    if not affinity_contract_value_matches(row.get(name), value):
                        raise ValueError(
                            "Predicted-distogram cache contract mismatch for "
                            f"{name!r}: expected {value!r}, got {row.get(name)!r}."
                        )
        if self._cache_reader is not None:
            self._cache_reader.close()
        self._cache_reader = FeatureCacheReader(self.cache_root)
        self._train_dataset = CachedAffinityFeatureDataset(
            train_rows,
            cache_root=self.cache_root,
            cache_schema=self.cache_schema,
            max_crop_tokens=self.max_crop_tokens,
            max_protein_crop_tokens=self.max_protein_crop_tokens,
            crop_mode=self.crop_mode,
            pocket_annotations=self._pocket_annotations,
            pocket_neighborhood_size=self.pocket_neighborhood_size,
            distogram_pocket_distance_cutoff=(self.distogram_pocket_distance_cutoff),
            distogram_use_entropy_tiebreak=self.distogram_use_entropy_tiebreak,
            reader=self._cache_reader,
            required_cache_encoding=self.cache_encoding,
        )
        self._val_dataset = CachedAffinityFeatureDataset(
            val_rows,
            cache_root=self.cache_root,
            cache_schema=self.cache_schema,
            max_crop_tokens=self.max_crop_tokens,
            max_protein_crop_tokens=self.max_protein_crop_tokens,
            crop_mode=self.crop_mode,
            pocket_annotations=self._pocket_annotations,
            pocket_neighborhood_size=self.pocket_neighborhood_size,
            distogram_pocket_distance_cutoff=(self.distogram_pocket_distance_cutoff),
            distogram_use_entropy_tiebreak=self.distogram_use_entropy_tiebreak,
            reader=self._cache_reader,
            required_cache_encoding=self.cache_encoding,
        )

    def teardown(self, stage: str | None = None) -> None:
        del stage
        if self._cache_reader is not None:
            self._cache_reader.close()
            self._cache_reader = None

    def train_dataloader(self) -> DataLoader[Any]:
        if self._train_dataset is None:
            raise RuntimeError(
                "Call setup('fit') before requesting affinity training data."
            )
        rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1
        if self.sampling_mode == "boltz2_activity_cliff":
            self._train_sampler = ActivityCliffBucketSampler(
                self._train_dataset.rows,
                num_batches=self.train_batches_per_epoch,
                seed=self.seed,
                rank=rank,
                world_size=world_size,
                batch_size=self.train_batch_size,
                logical_group_size=self.activity_group_size,
                logical_groups_per_batch=self.activity_groups_per_batch,
                singleton_regression_slots=self.singleton_regression_slots,
            )
        else:
            self._train_sampler = FixedValidSourceBalancedBucketSampler(
                self._train_dataset.rows,
                num_batches=self.train_batches_per_epoch,
                seed=self.seed,
                rank=rank,
                world_size=world_size,
                labels_per_source=self.labels_per_source,
                rankable_assays_per_batch=self.rankable_assays_per_batch,
                records_per_rankable_assay=self.records_per_rankable_assay,
            )
        worker_context = (
            {"multiprocessing_context": "spawn"} if self.num_workers > 0 else {}
        )
        return DataLoader(
            self._train_dataset,
            batch_sampler=self._train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=partial(
                collate_affinity_batch,
                batch_size=self.train_batch_size,
                fixed_shape=True,
            ),
            worker_init_fn=(
                configure_affinity_data_worker if self.num_workers > 0 else None
            ),
            **worker_context,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self._val_dataset is None:
            raise RuntimeError(
                "Call setup('fit') before requesting affinity validation data."
            )
        rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1
        worker_context = (
            {"multiprocessing_context": "spawn"} if self.num_workers > 0 else {}
        )
        return DataLoader(
            self._val_dataset,
            batch_sampler=FixedShapeValidationBatchSampler(
                self._val_dataset.rows,
                batch_size=self.val_batch_size,
                rank=rank,
                world_size=world_size,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=partial(collate_affinity_eval, fixed_shape=True),
            worker_init_fn=(
                configure_affinity_data_worker if self.num_workers > 0 else None
            ),
            **worker_context,
        )

    def on_train_epoch_start(self) -> None:
        if self.trainer is not None:
            self.set_train_epoch(self.trainer.current_epoch)

    def on_after_batch_transfer(
        self, batch: dict[str, Any], dataloader_idx: int
    ) -> dict[str, Any]:
        """Materialize compact pairs only after Lightning transfers them to device."""
        del dataloader_idx
        if self.cache_schema not in {
            AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1,
            AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
            AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1,
        }:
            return batch
        return materialize_target_compact_batch(batch)

    def set_train_epoch(self, epoch: int) -> None:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
