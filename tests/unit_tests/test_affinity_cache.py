import numpy as np
import pytest

from kfold.training.affinity.cache import (
    AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
    AFFINITY_CACHE_ENCODING_ARRAYPACK_ZSTD_V1,
    CacheWriteRequest,
    FeatureCacheReader,
    FeatureCacheWriter,
    deserialize_cache_payload,
    deserialize_npz,
    detect_cache_encoding,
    reconstruct_precision,
    serialize_cache_payload,
    serialize_npz,
)


def test_npz_cache_round_trip_and_precision_reconstruction(tmp_path) -> None:
    arrays = {
        "z": np.arange(32, dtype=np.float32).reshape(2, 2, 8),
        "mask": np.array([True, False]),
    }
    assert np.array_equal(deserialize_npz(serialize_npz(arrays))["z"], arrays["z"])
    with FeatureCacheWriter(
        tmp_path, records_per_shard=2, map_size_bytes=2**20
    ) as writer:
        entry = writer.put(
            system_id="sys-1",
            arrays=arrays,
            protein_tokens=1,
            ligand_tokens=1,
            crop_tokens=2,
            ligand_protein_entropy=0.5,
        )
    with FeatureCacheReader(tmp_path) as reader:
        assert reader.max_open_envs == 128
        loaded = reader.get(entry)
    assert np.array_equal(loaded["z"], arrays["z"])
    for mode in ("fp16", "int8", "nf4"):
        restored = reconstruct_precision(arrays["z"], mode)
        assert restored.shape == arrays["z"].shape
        assert np.isfinite(restored).all()


def test_reader_preflight_reports_missing_cache_shards(tmp_path) -> None:
    with FeatureCacheReader(tmp_path) as reader:
        with pytest.raises(FileNotFoundError, match="missing 1 referenced shard"):
            reader.validate_shards([{"cache_shard": "missing.lmdb"}])


def test_reader_bounds_open_shards_with_lru_eviction(tmp_path) -> None:
    arrays = {"values": np.arange(8, dtype=np.float32)}
    with FeatureCacheWriter(
        tmp_path, records_per_shard=1, map_size_bytes=2**20
    ) as writer:
        entries = [
            writer.put(
                system_id=f"sys-{index}",
                arrays=arrays,
                protein_tokens=1,
                ligand_tokens=1,
                crop_tokens=2,
                ligand_protein_entropy=0.0,
            )
            for index in range(3)
        ]

    with FeatureCacheReader(tmp_path, max_open_envs=2) as reader:
        assert np.array_equal(reader.get(entries[0])["values"], arrays["values"])
        assert np.array_equal(reader.get(entries[1])["values"], arrays["values"])
        assert list(reader._envs) == [entries[0].shard, entries[1].shard]

        # Refresh shard 0, then opening shard 2 must evict shard 1.
        reader.get(entries[0])
        reader.get(entries[2])
        assert list(reader._envs) == [entries[0].shard, entries[2].shard]
        assert len(reader._envs) == 2

        # An evicted shard remains readable and re-enters at the MRU end.
        assert np.array_equal(reader.get(entries[1])["values"], arrays["values"])
        assert list(reader._envs) == [entries[2].shard, entries[1].shard]


def test_reader_rejects_nonpositive_open_shard_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_open_envs must be positive"):
        FeatureCacheReader(tmp_path, max_open_envs=0)


def test_writer_commits_precompressed_batch_payloads(tmp_path) -> None:
    arrays = {"values": np.arange(8, dtype=np.float32)}
    request = CacheWriteRequest(
        system_id="sys-serialized",
        arrays=arrays,
        protein_tokens=1,
        ligand_tokens=1,
        crop_tokens=2,
        ligand_protein_entropy=0.0,
    )
    with FeatureCacheWriter(tmp_path, map_size_bytes=2**20) as writer:
        entry = writer.put_many_serialized([(request, serialize_npz(arrays))])[0]
    with FeatureCacheReader(tmp_path) as reader:
        loaded = reader.get(entry)
    assert np.array_equal(loaded["values"], arrays["values"])


@pytest.mark.parametrize(
    "encoding",
    [
        AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
        AFFINITY_CACHE_ENCODING_ARRAYPACK_ZSTD_V1,
    ],
)
def test_arraypack_round_trip_preserves_named_arrays(encoding: str) -> None:
    arrays = {
        "bf16_bits": np.arange(18, dtype=np.uint16).reshape(3, 6),
        "mask": np.asarray([True, False, True]),
        "indices": np.asarray([[0, 2], [2, 1]], dtype=np.uint16),
        "schema": np.asarray("affinity_pocket80k_target_consensus_100_v2"),
        "scalar": np.asarray(7, dtype=np.int64),
    }
    payload = serialize_cache_payload(arrays, encoding=encoding)
    assert detect_cache_encoding(payload) == encoding
    decoded, detected = deserialize_cache_payload(payload)
    assert detected == encoding
    assert set(decoded) == set(arrays)
    for name, expected in arrays.items():
        assert decoded[name].dtype == expected.dtype
        assert np.array_equal(decoded[name], expected)


def test_arraypack_reader_enforces_manifest_encoding(tmp_path) -> None:
    arrays = {"values": np.arange(8, dtype=np.uint16)}
    with FeatureCacheWriter(
        tmp_path,
        map_size_bytes=2**20,
        value_encoding=AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
    ) as writer:
        entry = writer.put(
            system_id="direct",
            arrays=arrays,
            protein_tokens=1,
            ligand_tokens=1,
            crop_tokens=2,
            ligand_protein_entropy=0.0,
        )
    with FeatureCacheReader(tmp_path) as reader:
        loaded = reader.get(entry)
        assert np.array_equal(loaded["values"], arrays["values"])
        with pytest.raises(ValueError, match="does not match its manifest"):
            reader.get(
                {
                    "cache_shard": entry.shard,
                    "cache_key": entry.key,
                    "cache_encoding": AFFINITY_CACHE_ENCODING_ARRAYPACK_ZSTD_V1,
                }
            )


@pytest.mark.parametrize("encoding", ["raw", "zstd"])
def test_arraypack_rejects_truncated_payload(encoding: str) -> None:
    selected = (
        AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1
        if encoding == "raw"
        else AFFINITY_CACHE_ENCODING_ARRAYPACK_ZSTD_V1
    )
    payload = serialize_cache_payload(
        {"values": np.arange(8, dtype=np.float32)}, encoding=selected
    )
    with pytest.raises(ValueError):
        deserialize_cache_payload(payload[:-3])
