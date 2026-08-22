"""Full-cross schema isolation and train-time crop dataset tests."""

import numpy as np
import pytest
import torch

import kfold.constants as C
from kfold.training.affinity.cache import FeatureCacheReader, FeatureCacheWriter
from kfold.training.affinity.dataset import (
    CachedAffinityFeatureDataset,
    CachedAffinityItem,
    collate_affinity_batch,
)
from kfold.training.affinity.pair_storage import (
    AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
    PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS,
    pack_full_cross_pair_storage,
    pack_pair_storage,
)
from kfold.training.affinity.pocket import PocketAnnotationLookup
from kfold.training.affinity.schema import AFFINITY_CACHE_SCHEMA_FULL_CROSS_POCKET_V2


def _full_cross_arrays() -> dict[str, np.ndarray]:
    length = 5
    logits = np.full((length, length, 8), -8.0, dtype=np.float32)
    for protein_index, logit in ((0, 8.0), (1, 6.0), (2, 2.0)):
        for ligand_index in (3, 4):
            logits[protein_index, ligand_index, 0] = logit
            logits[ligand_index, protein_index, 0] = logit
    return {
        "s_inputs": np.arange(length * 4, dtype=np.float32).reshape(length, 4),
        "s_lm": np.arange(length * 4, dtype=np.float32).reshape(length, 4),
        "z": np.arange(length * length * 3, dtype=np.float32).reshape(length, length, 3),
        "token_mask": np.ones(length, dtype=bool),
        "chain_type": np.asarray(
            [
                C.ChainType.PROTEIN.value,
                C.ChainType.PROTEIN.value,
                C.ChainType.PROTEIN.value,
                C.ChainType.LIGAND.value,
                C.ChainType.LIGAND.value,
            ]
        ),
        "distogram_logits": logits,
    }


def _row(entry) -> dict[str, object]:
    return {
        "cache_shard": entry.shard,
        "cache_key": entry.key,
        "cache_schema": AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
        "p_activity": 7.0,
        "assay_key": "assay",
        "canonical_smiles": "CCO",
        "origin": "sair",
        "record_id": "record",
    }


def test_dataset_crops_a_full_cross_record_only_when_loaded(tmp_path) -> None:
    packed = pack_pair_storage(
        _full_cross_arrays(), mode=PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS
    )
    with FeatureCacheWriter(tmp_path, map_size_bytes=2**20) as writer:
        entry = writer.put(
            system_id="system",
            arrays=packed.arrays,
            protein_tokens=3,
            ligand_tokens=2,
            crop_tokens=5,
            source_tokens=5,
            ligand_protein_entropy=0.0,
        )
    dataset = CachedAffinityFeatureDataset(
        [_row(entry)],
        cache_root=str(tmp_path),
        max_crop_tokens=4,
        max_protein_crop_tokens=2,
    )
    item = dataset[0]
    assert item is not None
    assert item.features["crop_indices"].tolist() == [0, 1, 3, 4]
    assert item.features["z"].shape[:2] == (4, 4)


def test_dataset_distogram_mode_reads_the_same_cached_pl_logits(tmp_path) -> None:
    packed = pack_pair_storage(
        _full_cross_arrays(), mode=PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS
    )
    with FeatureCacheWriter(tmp_path, map_size_bytes=2**20) as writer:
        entry = writer.put(
            system_id="system",
            arrays=packed.arrays,
            protein_tokens=3,
            ligand_tokens=2,
            crop_tokens=5,
            source_tokens=5,
            ligand_protein_entropy=0.0,
        )
    dataset = CachedAffinityFeatureDataset(
        [_row(entry)],
        cache_root=str(tmp_path),
        max_crop_tokens=4,
        max_protein_crop_tokens=2,
        crop_mode="distogram_predicted_contact",
    )
    item = dataset[0]
    assert item is not None
    assert item.features["crop_indices"].tolist() == [0, 1, 3, 4]


def test_train_and_validation_datasets_share_one_lmdb_reader(tmp_path) -> None:
    packed = pack_pair_storage(
        _full_cross_arrays(), mode=PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS
    )
    with FeatureCacheWriter(tmp_path, map_size_bytes=2**20) as writer:
        entry = writer.put(
            system_id="system",
            arrays=packed.arrays,
            protein_tokens=3,
            ligand_tokens=2,
            crop_tokens=5,
            source_tokens=5,
            ligand_protein_entropy=0.0,
        )
    reader = FeatureCacheReader(tmp_path)
    try:
        train_dataset = CachedAffinityFeatureDataset(
            [_row(entry)], cache_root=str(tmp_path), reader=reader
        )
        validation_dataset = CachedAffinityFeatureDataset(
            [_row(entry)], cache_root=str(tmp_path), reader=reader
        )
        assert train_dataset._get_reader() is validation_dataset._get_reader()
        assert train_dataset[0] is not None
        assert validation_dataset[0] is not None
    finally:
        reader.close()


def test_dataset_refuses_mixed_or_legacy_schema_rows(tmp_path) -> None:
    row = {
        "cache_shard": "unused",
        "cache_key": "unused",
        "cache_schema": AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
        "p_activity": 7.0,
        "assay_key": "assay",
        "canonical_smiles": "CCO",
        "origin": "sair",
        "record_id": "record",
    }
    legacy = {**row, "cache_schema": "affinity_cropped_legacy"}
    with pytest.raises(ValueError, match="cannot mix cache schemas"):
        CachedAffinityFeatureDataset([row, legacy], cache_root=str(tmp_path))


def test_dataset_uses_target_pocket_annotation_for_v2_crop(tmp_path) -> None:
    packed = pack_full_cross_pair_storage(
        _full_cross_arrays(),
        cache_schema=AFFINITY_CACHE_SCHEMA_FULL_CROSS_POCKET_V2,
    )
    with FeatureCacheWriter(tmp_path, map_size_bytes=2**20) as writer:
        entry = writer.put(
            system_id="system",
            arrays=packed.arrays,
            protein_tokens=3,
            ligand_tokens=2,
            crop_tokens=5,
            source_tokens=5,
            ligand_protein_entropy=0.0,
        )
    rows = [
        {
            "protein_key": "protein",
            "pocket_contract_version": "boltz2_target_pocket_v1",
            "protein_residue_min_distance": [3.0, 0.0, 2.0],
            "selected_request_id": "protein:00",
            "evidence_contract_sha256": "a" * 64,
        }
    ]
    dataset = CachedAffinityFeatureDataset(
        [
            {
                **_row(entry),
                "cache_schema": AFFINITY_CACHE_SCHEMA_FULL_CROSS_POCKET_V2,
                "protein_key": "protein",
                "protein_tokens": 3,
            }
        ],
        cache_root=str(tmp_path),
        cache_schema=AFFINITY_CACHE_SCHEMA_FULL_CROSS_POCKET_V2,
        max_crop_tokens=5,
        max_protein_crop_tokens=3,
        crop_mode="boltz2_pocket",
        pocket_annotations=PocketAnnotationLookup(rows),
    )
    item = dataset[0]
    assert item is not None
    assert item.features["crop_indices"].tolist() == [0, 1, 2, 3, 4]


def test_dataset_uses_pocket_overlay_with_v1_physical_payload(tmp_path) -> None:
    packed = pack_full_cross_pair_storage(
        _full_cross_arrays(),
        cache_schema=AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
    )
    with FeatureCacheWriter(tmp_path, map_size_bytes=2**20) as writer:
        entry = writer.put(
            system_id="system",
            arrays=packed.arrays,
            protein_tokens=3,
            ligand_tokens=2,
            crop_tokens=5,
            source_tokens=5,
            ligand_protein_entropy=0.0,
        )
    rows = [
        {
            "protein_key": "protein",
            "pocket_contract_version": "boltz2_target_pocket_v1",
            "protein_residue_min_distance": [3.0, 0.0, 2.0],
            "selected_request_id": "protein:00",
            "evidence_contract_sha256": "a" * 64,
        }
    ]
    dataset = CachedAffinityFeatureDataset(
        [
            {
                **_row(entry),
                "protein_key": "protein",
                "protein_tokens": 3,
            }
        ],
        cache_root=str(tmp_path),
        cache_schema=AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
        max_crop_tokens=5,
        max_protein_crop_tokens=3,
        crop_mode="boltz2_pocket",
        pocket_annotations=PocketAnnotationLookup(rows),
    )
    item = dataset[0]
    assert item is not None
    assert item.features["crop_indices"].tolist() == [0, 1, 2, 3, 4]


def test_dataset_uses_distogram_target_consensus_with_v1_payload(tmp_path) -> None:
    length = 11
    source = {
        "s_inputs": np.zeros((length, 4), dtype=np.float32),
        "s_lm": np.zeros((length, 4), dtype=np.float32),
        "z": np.zeros((length, length, 3), dtype=np.float32),
        "token_mask": np.ones(length, dtype=bool),
        "chain_type": np.asarray(
            [C.ChainType.PROTEIN.value] * 10 + [C.ChainType.LIGAND.value]
        ),
        "distogram_logits": np.zeros((length, length, 8), dtype=np.float32),
    }
    packed = pack_full_cross_pair_storage(source)
    with FeatureCacheWriter(tmp_path, map_size_bytes=2**20) as writer:
        entry = writer.put(
            system_id="system",
            arrays=packed.arrays,
            protein_tokens=10,
            ligand_tokens=1,
            crop_tokens=11,
            source_tokens=11,
            ligand_protein_entropy=0.0,
        )
    annotations = PocketAnnotationLookup(
        [
            {
                "protein_key": "protein",
                "pocket_contract_version": (
                    "affinity_distogram_target_consensus_pocket_v1"
                ),
                "protein_residue_min_distance": list(range(10)),
                "selected_request_id": "system",
                "evidence_contract_sha256": "a" * 64,
            }
        ],
        expected_contract_version="affinity_distogram_target_consensus_pocket_v1",
    )
    dataset = CachedAffinityFeatureDataset(
        [
            {
                **_row(entry),
                "protein_key": "protein",
                "protein_tokens": 10,
            }
        ],
        cache_root=str(tmp_path),
        cache_schema=AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
        max_crop_tokens=11,
        max_protein_crop_tokens=10,
        crop_mode="distogram_target_consensus",
        pocket_annotations=annotations,
        pocket_neighborhood_size=10,
    )
    item = dataset[0]
    assert item is not None
    assert item.features["crop_indices"].tolist() == list(range(11))


def test_benchmark_query_uses_ten_token_distogram_window(tmp_path) -> None:
    length = 13
    source = {
        "s_inputs": np.zeros((length, 4), dtype=np.float32),
        "s_lm": np.zeros((length, 4), dtype=np.float32),
        "z": np.zeros((length, length, 3), dtype=np.float32),
        "token_mask": np.ones(length, dtype=bool),
        "chain_type": np.asarray(
            [C.ChainType.PROTEIN.value] * 12 + [C.ChainType.LIGAND.value]
        ),
        "distogram_logits": np.zeros((length, length, 8), dtype=np.float32),
    }
    source["distogram_logits"][5, 12, 0] = 10.0
    source["distogram_logits"][12, 5, 0] = 10.0
    packed = pack_full_cross_pair_storage(source)
    with FeatureCacheWriter(tmp_path, map_size_bytes=2**20) as writer:
        entry = writer.put(
            system_id="query",
            arrays=packed.arrays,
            protein_tokens=12,
            ligand_tokens=1,
            crop_tokens=13,
            source_tokens=13,
            ligand_protein_entropy=0.0,
        )
    dataset = CachedAffinityFeatureDataset(
        [{**_row(entry), "protein_tokens": 12}],
        cache_root=str(tmp_path),
        cache_schema=AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
        max_crop_tokens=11,
        max_protein_crop_tokens=10,
        crop_mode="distogram_query_window",
        pocket_neighborhood_size=10,
    )
    item = dataset[0]
    assert item is not None
    assert (
        int((item.features["chain_type"] == C.ChainType.PROTEIN.value).sum().item()) == 10
    )
    assert len(item.features["crop_indices"]) == 11
    assert item.features["crop_indices"][-1] == 12


def test_fixed_shape_collation_uses_the_shared_bucket_not_dynamic_max() -> None:
    def item(record_id: str) -> CachedAffinityItem:
        length = 5
        return CachedAffinityItem(
            features={
                "s_inputs": torch.zeros((length, 4)),
                "s_lm": torch.zeros((length, 4)),
                "z": torch.zeros((length, length, 3)),
                "distogram_features": torch.zeros((length, length, 3)),
                "token_mask": torch.ones(length, dtype=torch.bool),
                "chain_type": torch.tensor(
                    [
                        C.ChainType.PROTEIN.value,
                        C.ChainType.PROTEIN.value,
                        C.ChainType.PROTEIN.value,
                        C.ChainType.LIGAND.value,
                        C.ChainType.LIGAND.value,
                    ]
                ),
            },
            label=7.0,
            assay_key=f"assay-{record_id}",
            canonical_smiles=f"ligand-{record_id}",
            shape_bucket=8,
            origin="SAIR",
            record_id=record_id,
        )

    batch = collate_affinity_batch(
        [item(str(index)) for index in range(64)],
        batch_size=64,
        fixed_shape=True,
    )
    assert batch["s_inputs"].shape == (64, 8, 4)
    assert batch["z"].shape == (64, 8, 8, 3)
    assert batch["valid_mask"].sum() == 64
