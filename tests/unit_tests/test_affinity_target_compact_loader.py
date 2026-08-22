from __future__ import annotations

import numpy as np
import pytest
import torch

import kfold.constants as C
from kfold.training.affinity.cache import (
    AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
    FeatureCacheWriter,
)
from kfold.training.affinity.datamodule import AffinityDataModule
from kfold.training.affinity.dataset import (
    CachedAffinityFeatureDataset,
    collate_affinity_batch,
    materialize_target_compact_batch,
)
from kfold.training.affinity.pocket import POCKET80K_TARGET_CROP_CONTRACT_V1
from kfold.training.affinity.schema import (
    AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
)


def bf16_bits(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    return tensor.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16).copy()


def compact_payload() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    length = 5
    token_mask = np.ones(length, dtype=bool)
    chain_type = np.asarray(
        [C.ChainType.PROTEIN.value] * 3 + [C.ChainType.LIGAND.value] * 2,
        dtype=np.int64,
    )
    protein = chain_type == C.ChainType.PROTEIN.value
    ligand = chain_type == C.ChainType.LIGAND.value
    pair_mask = (
        (protein[:, None] & ligand[None, :])
        | (ligand[:, None] & protein[None, :])
        | (ligand[:, None] & ligand[None, :])
    )
    pair_indices = np.stack(np.nonzero(pair_mask), axis=-1).astype(np.uint16)
    return {
        "cache_schema": np.asarray(AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2),
        "s_inputs_bf16": bf16_bits(rng.normal(size=(length, 4))),
        "s_lm_bf16": bf16_bits(rng.normal(size=(length, 4))),
        "token_mask": token_mask,
        "chain_type": chain_type,
        "crop_indices": np.arange(length, dtype=np.int32),
        "pair_indices": pair_indices,
        "z_pair_values_bf16": bf16_bits(rng.normal(size=(len(pair_indices), 6))),
        "distogram_feature_values_bf16": bf16_bits(
            rng.normal(size=(len(pair_indices), 3))
        ),
    }


def write_item(tmp_path, monkeypatch: pytest.MonkeyPatch):
    payload = compact_payload()
    with FeatureCacheWriter(
        tmp_path,
        map_size_bytes=2**20,
        value_encoding=AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
    ) as writer:
        entry = writer.put(
            system_id="system",
            arrays=payload,
            protein_tokens=3,
            ligand_tokens=2,
            crop_tokens=5,
            source_tokens=5,
            ligand_protein_entropy=0.0,
        )
    row = {
        "cache_shard": entry.shard,
        "cache_key": entry.key,
        "cache_schema": AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
        "cache_encoding": AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
        "p_activity": 7.0,
        "assay_key": "assay",
        "canonical_smiles": "CCO",
        "origin": "SAIR",
        "record_id": "record",
        "shape_bucket": 256,
    }
    monkeypatch.setattr(
        "kfold.training.affinity.dataset.unpack_cross_only_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("target compact loader must not unpack on CPU")
        ),
    )
    dataset = CachedAffinityFeatureDataset(
        [row],
        cache_root=str(tmp_path),
        cache_schema=AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
        crop_mode="pocket80k_target_compact",
        required_cache_encoding=AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
    )
    item = dataset[0]
    assert item is not None
    return item, payload


def test_dataset_returns_encoded_sparse_bf16_without_dense_cpu_pair(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, payload = write_item(tmp_path, monkeypatch)
    assert "z" not in item.features
    assert "distogram_features" not in item.features
    assert item.features["s_inputs_bf16"].dtype == torch.bfloat16
    assert item.features["s_lm_bf16"].dtype == torch.bfloat16
    assert item.features["z_pair_values_bf16"].dtype == torch.bfloat16
    assert item.features["distogram_feature_values_bf16"].dtype == torch.bfloat16
    assert item.features["pair_indices"].shape == payload["pair_indices"].shape


def test_sparse_collate_and_materialization_match_dense_reference(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, _ = write_item(tmp_path, monkeypatch)
    batch = collate_affinity_batch([item, item], batch_size=2, fixed_shape=True)
    assert batch["sparse_compact"] is True
    assert "z" not in batch
    assert batch["s_inputs_bf16"].shape == (2, 256, 4)
    assert batch["s_inputs_bf16"].dtype == torch.bfloat16

    dense = materialize_target_compact_batch(batch)
    assert dense["z"].shape == (2, 256, 256, 6)
    assert dense["distogram_features"].shape == (2, 256, 256, 3)
    assert dense["z"].dtype == torch.bfloat16
    assert dense["distogram_features"].dtype == torch.bfloat16
    reference_z = torch.zeros_like(dense["z"])
    reference_distogram = torch.zeros_like(dense["distogram_features"])
    sample, left, right = batch["pair_indices"].unbind(dim=-1)
    reference_z[sample, left, right] = batch["z_pair_values_bf16"]
    reference_distogram[sample, left, right] = batch["distogram_feature_values_bf16"]
    assert torch.equal(dense["z"], reference_z)
    assert torch.equal(dense["distogram_features"], reference_distogram)
    assert dense["protein_mask"][:, :3].all()
    assert dense["ligand_mask"][:, 3:5].all()


def test_datamodule_materializes_target_batch_after_transfer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, _ = write_item(tmp_path, monkeypatch)
    batch = collate_affinity_batch([item], batch_size=1, fixed_shape=True)
    module = AffinityDataModule(
        manifest_path="unused.parquet",
        cache_root="unused",
        train_batches_per_epoch=1,
        seed=1,
        train_batch_size=1,
        activity_group_size=1,
        activity_groups_per_batch=1,
        singleton_regression_slots=0,
        cache_schema=AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
        cache_encoding=AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
        crop_mode="pocket80k_target_compact",
        pocket_manifest_path=None,
        crop_contract_version=POCKET80K_TARGET_CROP_CONTRACT_V1,
    )
    dense = module.on_after_batch_transfer(batch, 0)
    assert dense["z"].shape == (1, 256, 256, 6)
    assert "pair_indices" not in dense


def test_dataset_rejects_duplicate_sparse_pairs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = compact_payload()
    payload["pair_indices"][1] = payload["pair_indices"][0]
    with FeatureCacheWriter(
        tmp_path,
        map_size_bytes=2**20,
        value_encoding=AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
    ) as writer:
        entry = writer.put(
            system_id="bad",
            arrays=payload,
            protein_tokens=3,
            ligand_tokens=2,
            crop_tokens=5,
            source_tokens=5,
            ligand_protein_entropy=0.0,
        )
    dataset = CachedAffinityFeatureDataset(
        [
            {
                "cache_shard": entry.shard,
                "cache_key": entry.key,
                "cache_schema": AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
                "cache_encoding": AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
                "p_activity": 7.0,
                "assay_key": "assay",
                "canonical_smiles": "CCO",
                "origin": "SAIR",
                "record_id": "record",
            }
        ],
        cache_root=str(tmp_path),
        cache_schema=AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
        crop_mode="pocket80k_target_compact",
        required_cache_encoding=AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
    )
    with pytest.raises(ValueError, match="must be unique"):
        dataset[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_target_materialization_stays_bf16_on_gpu(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, _ = write_item(tmp_path, monkeypatch)
    batch = collate_affinity_batch([item], batch_size=1, fixed_shape=True)
    transferred = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    dense = materialize_target_compact_batch(transferred)
    assert dense["z"].is_cuda
    assert dense["z"].dtype == torch.bfloat16
