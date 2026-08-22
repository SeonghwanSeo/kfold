import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from kfold.training.affinity.datamodule import (
    AffinityDataModule,
    affinity_contract_value_matches,
    attach_shape_buckets,
    configure_affinity_data_worker,
)


def test_contract_float_comparison_accepts_parquet_float32_roundoff() -> None:
    assert affinity_contract_value_matches(0.699999988079071, 0.7)
    assert not affinity_contract_value_matches(0.69, 0.7)
    assert affinity_contract_value_matches(
        "protein_min_expected_distance_lt_15A_pl_pairs",
        "protein_min_expected_distance_lt_15A_pl_pairs",
    )


class SharingStrategyDataset(Dataset[str]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> str:
        assert index == 0
        return torch.multiprocessing.get_sharing_strategy()


def test_attach_shape_buckets_uses_cache_token_metadata() -> None:
    rows = [
        {"protein_tokens": 100, "ligand_tokens": 20},
        {"protein_tokens": 156, "ligand_tokens": 12},
    ]
    attached = attach_shape_buckets(
        rows,
        shape_buckets=(64, 128, 192, 256),
        max_crop_tokens=256,
        max_protein_crop_tokens=200,
    )
    assert [row["head_crop_tokens"] for row in attached] == [120, 168]
    assert [row["shape_bucket"] for row in attached] == [128, 192]


def test_attach_shape_buckets_requires_cache_token_metadata() -> None:
    with pytest.raises(KeyError, match="protein_tokens"):
        attach_shape_buckets(
            [{"ligand_tokens": 20}],
            shape_buckets=(64, 128),
            max_crop_tokens=256,
            max_protein_crop_tokens=200,
        )


def test_fixed_valid_data_module_requires_32_real_labels() -> None:
    with pytest.raises(ValueError, match="fill train_batch_size"):
        AffinityDataModule(
            manifest_path="unused.parquet",
            cache_root="unused-cache",
            train_batches_per_epoch=1,
            seed=1,
            sampling_mode="legacy_fixed_valid",
            crop_mode="legacy_predicted_contact",
            labels_per_source=15,
        )


def test_boltz2_crop_accepts_v1_physical_full_cross_cache() -> None:
    module = AffinityDataModule(
        manifest_path="unused.parquet",
        cache_root="unused-cache",
        train_batches_per_epoch=1,
        seed=1,
        cache_schema="affinity_full_cross_v1",
        crop_mode="boltz2_pocket",
        sampling_mode="boltz2_activity_cliff",
        pocket_manifest_path="unused-pocket.parquet",
        crop_contract_version="boltz2_affinity_crop_v2",
    )
    assert module.cache_schema == "affinity_full_cross_v1"


def test_boltz2_sampler_accepts_cpu_distogram_crop() -> None:
    module = AffinityDataModule(
        manifest_path="unused.parquet",
        cache_root="unused-cache",
        train_batches_per_epoch=1,
        seed=1,
        cache_schema="affinity_full_cross_v1",
        crop_mode="distogram_predicted_contact",
        sampling_mode="boltz2_activity_cliff",
        pocket_manifest_path=None,
        crop_contract_version="affinity_distogram_crop_v2",
    )
    assert module.crop_mode == "distogram_predicted_contact"


def test_boltz2_sampler_accepts_b64_twelve_assay_layout() -> None:
    module = AffinityDataModule(
        manifest_path="unused.parquet",
        cache_root="unused-cache",
        train_batches_per_epoch=1,
        seed=1,
        train_batch_size=64,
        activity_group_size=5,
        activity_groups_per_batch=12,
        singleton_regression_slots=4,
        cache_schema="affinity_full_cross_v1",
        crop_mode="distogram_predicted_contact",
        sampling_mode="boltz2_activity_cliff",
        crop_contract_version="affinity_distogram_crop_v2",
    )
    assert module.train_batch_size == 64


def test_distogram_crop_refuses_pocket_manifest() -> None:
    with pytest.raises(ValueError, match="does not use a pocket manifest"):
        AffinityDataModule(
            manifest_path="unused.parquet",
            cache_root="unused-cache",
            train_batches_per_epoch=1,
            seed=1,
            cache_schema="affinity_full_cross_v1",
            crop_mode="distogram_predicted_contact",
            sampling_mode="boltz2_activity_cliff",
            pocket_manifest_path="unused-pocket.parquet",
            crop_contract_version="affinity_distogram_crop_v2",
        )


def test_target_consensus_requires_explicit_contract_and_pocket() -> None:
    module = AffinityDataModule(
        manifest_path="unused.parquet",
        cache_root="unused-cache",
        train_batches_per_epoch=1,
        seed=1,
        cache_schema="affinity_full_cross_v1",
        crop_mode="distogram_target_consensus",
        sampling_mode="boltz2_activity_cliff",
        pocket_manifest_path="consensus.parquet",
        crop_contract_version="affinity_distogram_target_consensus_v1",
        pocket_neighborhood_size=10,
    )
    assert module.crop_mode == "distogram_target_consensus"

    with pytest.raises(ValueError, match="explicit crop contract"):
        AffinityDataModule(
            manifest_path="unused.parquet",
            cache_root="unused-cache",
            train_batches_per_epoch=1,
            seed=1,
            cache_schema="affinity_full_cross_v1",
            crop_mode="distogram_target_consensus",
            sampling_mode="boltz2_activity_cliff",
            pocket_manifest_path="consensus.parquet",
            crop_contract_version="affinity_distogram_crop_v2",
        )


def test_data_module_updates_real_batch_sampler_epoch() -> None:
    module = AffinityDataModule(
        manifest_path="unused.parquet",
        cache_root="unused-cache",
        train_batches_per_epoch=1,
        seed=1,
        sampling_mode="legacy_fixed_valid",
        crop_mode="legacy_predicted_contact",
        cache_schema="affinity_full_cross_v1",
    )

    class Sampler:
        epoch = -1

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    sampler = Sampler()
    module._train_sampler = sampler  # type: ignore[assignment]
    module.set_train_epoch(3)
    assert sampler.epoch == 3


def test_affinity_worker_uses_file_system_tensor_sharing(monkeypatch) -> None:
    observed: list[str] = []
    monkeypatch.setattr(
        "kfold.training.affinity.datamodule.torch.multiprocessing.set_sharing_strategy",
        observed.append,
    )
    configure_affinity_data_worker(0)
    assert observed == ["file_system"]


def test_spawned_affinity_worker_uses_file_system_tensor_sharing() -> None:
    loader = DataLoader(
        SharingStrategyDataset(),
        num_workers=1,
        multiprocessing_context="spawn",
        worker_init_fn=configure_affinity_data_worker,
    )
    assert next(iter(loader)) == ["file_system"]
