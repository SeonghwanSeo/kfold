import pytest

from kfold.training.affinity.sampler import (
    ActivityCliffBucketSampler,
    FixedShapeValidationBatchSampler,
    FixedValidSourceBalancedBucketSampler,
    SourceBalancedAssayBatchSampler,
)


def test_source_balanced_sampler_keeps_fixed_8x4_layout() -> None:
    records = []
    for origin in ("SAIR", "BindingDB-residual"):
        for assay_index in range(5):
            for record_index in range(assay_index % 4 + 1):
                records.append(
                    {
                        "origin": origin,
                        "assay_key": f"{origin}:assay-{assay_index}",
                        "record_id": f"{origin}:{assay_index}:{record_index}",
                    }
                )
    sampler = SourceBalancedAssayBatchSampler(records, num_batches=1, seed=1)
    batch = next(iter(sampler))
    assert len(batch) == 32
    for group_index in range(8):
        group = batch[group_index * 4 : (group_index + 1) * 4]
        non_padding = [index for index in group if index >= 0]
        assert non_padding
        expected_origin = "SAIR" if group_index < 4 else "BindingDB-residual"
        assert all(records[index]["origin"] == expected_origin for index in non_padding)


def test_sampler_prefers_distinct_ligands_before_replicates() -> None:
    records = []
    for origin in ("SAIR", "BindingDB-residual"):
        for ligand in ("A", "B", "C"):
            for replicate in range(2):
                records.append(
                    {
                        "origin": origin,
                        "assay_key": f"{origin}:assay",
                        "canonical_smiles": ligand,
                        "record_id": f"{origin}:{ligand}:{replicate}",
                    }
                )
    sampler = SourceBalancedAssayBatchSampler(
        records,
        num_batches=1,
        seed=7,
        assays_per_source=1,
        records_per_assay=3,
    )
    batch = next(iter(sampler))
    for group in (batch[:3], batch[3:]):
        assert {records[index]["canonical_smiles"] for index in group} == {"A", "B", "C"}


def _fixed_bucket_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for assay in range(4):
        for ligand in range(4):
            records.append(
                {
                    "origin": "SAIR",
                    "assay_key": f"sair-rank-{assay}",
                    "canonical_smiles": f"S{assay}-{ligand}",
                    "record_id": f"sair-{assay}-{ligand}",
                    "shape_bucket": 128,
                }
            )
    for label in range(20):
        records.append(
            {
                "origin": "BindingDB-residual",
                "assay_key": f"bdb-singleton-{label}",
                "canonical_smiles": f"B{label}",
                "record_id": f"bdb-{label}",
                "shape_bucket": 128,
            }
        )
    return records


def test_fixed_valid_sampler_emits_16_real_labels_per_source() -> None:
    records = _fixed_bucket_records()
    sampler = FixedValidSourceBalancedBucketSampler(
        records,
        num_batches=1,
        seed=11,
    )
    batch = next(iter(sampler))
    assert len(batch) == 32
    assert len(set(batch)) == 32
    assert all(index >= 0 for index in batch)
    assert all(records[index]["origin"] == "SAIR" for index in batch[:16])
    assert all(records[index]["origin"] == "BindingDB-residual" for index in batch[16:])
    assert {records[index]["shape_bucket"] for index in batch} == {128}


def test_fixed_shape_validation_sampler_pads_only_the_last_bucket_batch() -> None:
    records = [
        {"shape_bucket": 128},
        {"shape_bucket": 128},
        *[{"shape_bucket": 160} for _ in range(33)],
    ]
    sampler = FixedShapeValidationBatchSampler(records, batch_size=32)
    batches = list(sampler)
    assert [len(batch) for batch in batches] == [32, 32, 32]
    assert batches[0][:2] == [0, 1]
    assert all(index == -1 for index in batches[0][2:])
    assert all(index >= 0 for index in batches[1])
    assert batches[2][0] == 34
    assert all(index == -1 for index in batches[2][1:])


def test_fixed_shape_validation_sampler_shards_records_without_duplication() -> None:
    records = [
        *[{"shape_bucket": 128} for _ in range(35)],
        *[{"shape_bucket": 160} for _ in range(17)],
    ]
    batches = [
        batch
        for rank in range(4)
        for batch in FixedShapeValidationBatchSampler(
            records,
            batch_size=32,
            rank=rank,
            world_size=4,
        )
    ]
    observed = [index for batch in batches for index in batch if index >= 0]
    assert sorted(observed) == list(range(len(records)))
    assert len(observed) == len(set(observed))
    assert all(len(batch) == 32 for batch in batches)


def test_activity_cliff_sampler_emits_twelve_assays_and_four_bdb_records() -> None:
    records: list[dict[str, object]] = []
    for assay in range(12):
        for ligand in range(5):
            records.append(
                {
                    "origin": "SAIR",
                    "assay_key": f"activity-{assay}",
                    "canonical_smiles": f"A{assay}-{ligand}",
                    "record_id": f"activity-{assay}-{ligand}",
                    "p_activity": float(ligand),
                    "shape_bucket": 128,
                }
            )
    for singleton in range(4):
        records.append(
            {
                "origin": "BindingDB-residual",
                "assay_key": f"singleton-{singleton}",
                "canonical_smiles": f"B{singleton}",
                "record_id": f"singleton-{singleton}",
                "p_activity": 7.0,
                "shape_bucket": 128,
            }
        )
    sampler = ActivityCliffBucketSampler(
        records,
        num_batches=1,
        seed=3,
        batch_size=64,
        logical_groups_per_batch=12,
        singleton_regression_slots=4,
    )
    batch = next(iter(sampler))
    assert len(batch) == 64
    selected_assays = set()
    for group_id in range(12):
        group = [item for item in batch if item.logical_group_id == group_id]
        assert len(group) == 5
        assert len({records[item.record_index]["assay_key"] for item in group}) == 1
        assert {records[item.record_index]["origin"] for item in group} == {"SAIR"}
        selected_assays.add(records[group[0].record_index]["assay_key"])
    assert len(selected_assays) == 12
    singleton_ids = [item.logical_group_id for item in batch[-4:]]
    assert singleton_ids == [12, 13, 14, 15]
    assert {records[item.record_index]["origin"] for item in batch[-4:]} == {
        "BindingDB-residual"
    }


def test_activity_cliff_sampler_requires_bindingdb_slots() -> None:
    records = []
    for assay in range(6):
        for ligand in range(5):
            records.append(
                {
                    "origin": "SAIR",
                    "assay_key": f"activity-{assay}",
                    "canonical_smiles": f"A{assay}-{ligand}",
                    "record_id": f"activity-{assay}-{ligand}",
                    "p_activity": float(ligand),
                    "shape_bucket": 256,
                    "ranking_eligible": True,
                }
            )
    with pytest.raises(ValueError, match="No crop bucket"):
        ActivityCliffBucketSampler(records, num_batches=1, seed=3)
