"""Sharded feature storage for the frozen affinity backbone.

Legacy caches use compressed NPZ values.  Final pocket-cropped affinity caches
use one deterministic arraypack value per system: Zstd on shared storage and
raw contiguous bytes in node-local training replicas.
"""

from __future__ import annotations

import io
import json
import os
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

AFFINITY_CACHE_ENCODING_NPZ_V1 = "affinity_npz_compressed_v1"
AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1 = "affinity_arraypack_raw_v1"
AFFINITY_CACHE_ENCODING_ARRAYPACK_ZSTD_V1 = "affinity_arraypack_zstd_v1"
AFFINITY_CACHE_ENCODINGS = frozenset(
    {
        AFFINITY_CACHE_ENCODING_NPZ_V1,
        AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1,
        AFFINITY_CACHE_ENCODING_ARRAYPACK_ZSTD_V1,
    }
)

_ARRAYPACK_RAW_MAGIC = b"KFAAR1\0\0"
_ARRAYPACK_ZSTD_MAGIC = b"KFAAZ1\0\0"
_ARRAYPACK_PREFIX = struct.Struct("<IQ")
_ARRAYPACK_ZSTD_SIZE = struct.Struct("<Q")
_ARRAYPACK_ALIGNMENT = 8


def _zstandard_module():
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - exercised in cache jobs.
        raise RuntimeError(
            "Zstd affinity arraypack values require the optional 'zstandard' "
            "training dependency."
        ) from exc
    return zstandard


def _align(value: int, alignment: int = _ARRAYPACK_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def serialize_arraypack_raw(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Pack named contiguous arrays without compression or pickle metadata."""
    entries: list[dict[str, object]] = []
    data = bytearray()
    for name in sorted(arrays):
        if not isinstance(name, str) or not name:
            raise ValueError("Arraypack field names must be non-empty strings.")
        array = np.asarray(arrays[name])
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if array.dtype.hasobject:
            raise ValueError(f"Arraypack field {name!r} cannot contain objects.")
        aligned = _align(len(data))
        if aligned > len(data):
            data.extend(b"\0" * (aligned - len(data)))
        payload = array.tobytes(order="C")
        entries.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "offset": len(data),
                "nbytes": len(payload),
            }
        )
        data.extend(payload)
    header = json.dumps(
        {"schema_version": "affinity_arraypack_v1", "arrays": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prefix = _ARRAYPACK_PREFIX.pack(len(header), len(data))
    data_start = _align(len(_ARRAYPACK_RAW_MAGIC) + len(prefix) + len(header))
    padding = data_start - len(_ARRAYPACK_RAW_MAGIC) - len(prefix) - len(header)
    return _ARRAYPACK_RAW_MAGIC + prefix + header + b"\0" * padding + bytes(data)


def _deserialize_arraypack_raw(payload: bytes | bytearray) -> dict[str, np.ndarray]:
    if not payload.startswith(_ARRAYPACK_RAW_MAGIC):
        raise ValueError("Affinity arraypack raw magic is missing.")
    prefix_start = len(_ARRAYPACK_RAW_MAGIC)
    prefix_end = prefix_start + _ARRAYPACK_PREFIX.size
    if len(payload) < prefix_end:
        raise ValueError("Affinity arraypack prefix is truncated.")
    header_size, data_size = _ARRAYPACK_PREFIX.unpack(payload[prefix_start:prefix_end])
    header_end = prefix_end + header_size
    data_start = _align(header_end)
    if header_end > len(payload) or data_start + data_size != len(payload):
        raise ValueError("Affinity arraypack size metadata is inconsistent.")
    try:
        header = json.loads(bytes(payload[prefix_end:header_end]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Affinity arraypack header is invalid.") from exc
    if header.get("schema_version") != "affinity_arraypack_v1":
        raise ValueError("Unsupported affinity arraypack header version.")
    entries = header.get("arrays")
    if not isinstance(entries, list):
        raise ValueError("Affinity arraypack header lacks an array table.")
    # LMDB returns immutable bytes.  One bytearray copy makes every NumPy view
    # writable for torch.from_numpy while retaining a single shared backing buffer.
    storage = payload if isinstance(payload, bytearray) else bytearray(payload)
    arrays: dict[str, np.ndarray] = {}
    occupied: list[tuple[int, int]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Affinity arraypack entries must be objects.")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in arrays:
            raise ValueError("Affinity arraypack field names must be unique.")
        try:
            dtype = np.dtype(str(entry["dtype"]))
            shape = tuple(int(value) for value in entry["shape"])
            offset = int(entry["offset"])
            nbytes = int(entry["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Affinity arraypack metadata is invalid for {name!r}."
            ) from exc
        if (
            dtype.hasobject
            or any(value < 0 for value in shape)
            or offset < 0
            or nbytes < 0
        ):
            raise ValueError(f"Affinity arraypack metadata is unsafe for {name!r}.")
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes != expected or offset + nbytes > data_size:
            raise ValueError(f"Affinity arraypack byte extent is invalid for {name!r}.")
        extent = (offset, offset + nbytes)
        if any(extent[0] < right and left < extent[1] for left, right in occupied):
            raise ValueError("Affinity arraypack array extents overlap.")
        occupied.append(extent)
        arrays[name] = np.frombuffer(
            storage,
            dtype=dtype,
            count=int(np.prod(shape, dtype=np.int64)),
            offset=data_start + offset,
        ).reshape(shape)
    return arrays


def serialize_arraypack_zstd(
    arrays: Mapping[str, np.ndarray], *, level: int = 3
) -> bytes:
    """Compress one raw arraypack as a single Zstd frame for durable storage."""
    raw = serialize_arraypack_raw(arrays)
    body = raw[len(_ARRAYPACK_RAW_MAGIC) :]
    compressed = _zstandard_module().ZstdCompressor(level=level).compress(body)
    return _ARRAYPACK_ZSTD_MAGIC + _ARRAYPACK_ZSTD_SIZE.pack(len(body)) + compressed


def deserialize_arraypack_zstd(payload: bytes) -> dict[str, np.ndarray]:
    if not payload.startswith(_ARRAYPACK_ZSTD_MAGIC):
        raise ValueError("Affinity arraypack Zstd magic is missing.")
    size_start = len(_ARRAYPACK_ZSTD_MAGIC)
    size_end = size_start + _ARRAYPACK_ZSTD_SIZE.size
    if len(payload) <= size_end:
        raise ValueError("Affinity arraypack Zstd payload is truncated.")
    (expected_size,) = _ARRAYPACK_ZSTD_SIZE.unpack(payload[size_start:size_end])
    try:
        body = (
            _zstandard_module()
            .ZstdDecompressor()
            .decompress(payload[size_end:], max_output_size=expected_size)
        )
    except _zstandard_module().ZstdError as exc:
        raise ValueError("Affinity arraypack Zstd frame is invalid.") from exc
    if len(body) != expected_size:
        raise ValueError("Affinity arraypack Zstd decoded size is inconsistent.")
    return _deserialize_arraypack_raw(_ARRAYPACK_RAW_MAGIC + body)


def detect_cache_encoding(payload: bytes) -> str:
    if payload.startswith(_ARRAYPACK_RAW_MAGIC):
        return AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1
    if payload.startswith(_ARRAYPACK_ZSTD_MAGIC):
        return AFFINITY_CACHE_ENCODING_ARRAYPACK_ZSTD_V1
    if payload.startswith(b"PK"):
        return AFFINITY_CACHE_ENCODING_NPZ_V1
    raise ValueError("Affinity cache value has an unknown physical encoding.")


def serialize_cache_payload(arrays: Mapping[str, np.ndarray], *, encoding: str) -> bytes:
    if encoding == AFFINITY_CACHE_ENCODING_NPZ_V1:
        return serialize_npz(arrays)
    if encoding == AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1:
        return serialize_arraypack_raw(arrays)
    if encoding == AFFINITY_CACHE_ENCODING_ARRAYPACK_ZSTD_V1:
        return serialize_arraypack_zstd(arrays)
    raise ValueError(f"Unsupported affinity cache encoding: {encoding!r}.")


def deserialize_cache_payload(payload: bytes) -> tuple[dict[str, np.ndarray], str]:
    encoding = detect_cache_encoding(payload)
    if encoding == AFFINITY_CACHE_ENCODING_NPZ_V1:
        return deserialize_npz(payload), encoding
    if encoding == AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1:
        return _deserialize_arraypack_raw(payload), encoding
    return deserialize_arraypack_zstd(payload), encoding


def _lmdb_module():
    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover - exercised in cache jobs.
        raise RuntimeError(
            "The affinity feature cache requires the optional 'lmdb' training "
            "dependency. Install the project's train extra first."
        ) from exc
    return lmdb


def serialize_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize contiguous NumPy arrays into a compressed, portable payload."""
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **{key: np.ascontiguousarray(value) for key, value in arrays.items()},
    )
    return buffer.getvalue()


def deserialize_npz(payload: bytes) -> dict[str, np.ndarray]:
    """Load a compressed cache value without allowing pickle execution."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


@dataclass(frozen=True, kw_only=True)
class CacheIndexEntry:
    """Location and basic accounting metadata for one cached system."""

    system_id: str
    shard: str
    key: str
    compressed_bytes: int
    raw_bytes: int
    protein_tokens: int
    ligand_tokens: int
    crop_tokens: int
    ligand_protein_entropy: float
    cache_encoding: str
    source_tokens: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class CacheWriteRequest:
    """One immutable cache value prepared before a batch LMDB transaction."""

    system_id: str
    arrays: Mapping[str, np.ndarray]
    protein_tokens: int
    ligand_tokens: int
    crop_tokens: int
    ligand_protein_entropy: float
    source_tokens: int | None = None


class FeatureCacheWriter:
    """Append immutable system feature records to predictable LMDB shards."""

    def __init__(
        self,
        root: str | Path,
        *,
        records_per_shard: int = 2_000,
        map_size_bytes: int = 128 * 1024**3,
        shard_prefix: str = "features",
        resume: bool = False,
        value_encoding: str = AFFINITY_CACHE_ENCODING_NPZ_V1,
    ) -> None:
        if records_per_shard <= 0:
            raise ValueError("records_per_shard must be positive.")
        if map_size_bytes <= 0:
            raise ValueError("map_size_bytes must be positive.")
        if value_encoding not in AFFINITY_CACHE_ENCODINGS:
            raise ValueError(f"Unsupported affinity cache encoding: {value_encoding!r}.")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_per_shard = records_per_shard
        self.map_size_bytes = map_size_bytes
        self.shard_prefix = shard_prefix
        self.resume = resume
        self.value_encoding = value_encoding
        self._env: Any | None = None
        self._shard_index: int | None = None
        self._records_in_shard = 0

    def _shard_name(self, index: int) -> str:
        return f"{self.shard_prefix}-{index:05d}.lmdb"

    def _existing_shard_indices(self) -> list[int]:
        indices = []
        for path in self.root.glob(f"{self.shard_prefix}-*.lmdb"):
            stem = path.name[len(self.shard_prefix) + 1 : -len(".lmdb")]
            if stem.isdigit():
                indices.append(int(stem))
        return sorted(indices)

    def _open_shard(self, shard_index: int) -> None:
        self.close()
        lmdb = _lmdb_module()
        shard_path = self.root / self._shard_name(shard_index)
        self._env = lmdb.open(
            str(shard_path),
            subdir=False,
            map_size=self.map_size_bytes,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
            max_readers=512,
        )
        self._shard_index = shard_index
        with self._env.begin() as transaction:
            self._records_in_shard = int(transaction.stat()["entries"])

    def _ensure_shard(self, *, required_records: int = 1) -> None:
        if required_records <= 0:
            raise ValueError("required_records must be positive.")
        if required_records > self.records_per_shard:
            raise ValueError(
                "A single atomic write batch exceeds records_per_shard: "
                f"{required_records} > {self.records_per_shard}."
            )
        if self._env is None:
            if self._shard_index is not None:
                # ``discard`` closes the environment to reopen shards for
                # writing; resume where this writer left off, not at shard 0.
                self._open_shard(self._shard_index)
            else:
                existing = self._existing_shard_indices() if self.resume else []
                self._open_shard(existing[-1] if existing else 0)
        if self._records_in_shard + required_records > self.records_per_shard:
            assert self._shard_index is not None
            self._open_shard(self._shard_index + 1)

    @staticmethod
    def _read_keys(env: Any) -> set[str]:
        with env.begin() as transaction:
            return {key.decode("utf-8") for key, _ in transaction.cursor()}

    def keys(self) -> set[str]:
        """Return every system ID already present across this partition's shards.

        LMDB refuses a second open of the same environment inside one process,
        so the shard this writer currently holds is read through the live
        handle rather than reopened.
        """
        lmdb = _lmdb_module()
        found: set[str] = set()
        for index in self._existing_shard_indices():
            if self._env is not None and index == self._shard_index:
                found |= self._read_keys(self._env)
                continue
            env = lmdb.open(
                str(self.root / self._shard_name(index)),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
            )
            try:
                found |= self._read_keys(env)
            finally:
                env.close()
        return found

    def discard(self, system_ids: Iterable[str]) -> int:
        """Delete orphaned records so a resumed run can rewrite them exactly.

        A kill between the LMDB commit and the index append leaves a value with
        no index row.  Rather than carry an unindexed record forever, the
        resume path removes it and lets the normal loop regenerate both.
        """
        targets = {str(system_id) for system_id in system_ids}
        if not targets:
            return 0
        self.close()
        lmdb = _lmdb_module()
        removed = 0
        for index in self._existing_shard_indices():
            path = self.root / self._shard_name(index)
            env = lmdb.open(
                str(path),
                subdir=False,
                map_size=self.map_size_bytes,
                readonly=False,
                lock=True,
                readahead=False,
                meminit=False,
                max_readers=512,
            )
            try:
                with env.begin(write=True) as transaction:
                    for system_id in targets:
                        if transaction.delete(system_id.encode("utf-8")):
                            removed += 1
                env.sync()
            finally:
                env.close()
        return removed

    def put(
        self,
        *,
        system_id: str,
        arrays: Mapping[str, np.ndarray],
        protein_tokens: int,
        ligand_tokens: int,
        crop_tokens: int,
        ligand_protein_entropy: float,
        source_tokens: int | None = None,
    ) -> CacheIndexEntry:
        """Write one cache value, refusing accidental overwrite of an existing ID."""
        return self.put_many(
            [
                CacheWriteRequest(
                    system_id=system_id,
                    arrays=arrays,
                    protein_tokens=protein_tokens,
                    ligand_tokens=ligand_tokens,
                    crop_tokens=crop_tokens,
                    ligand_protein_entropy=ligand_protein_entropy,
                    source_tokens=source_tokens,
                )
            ]
        )[0]

    def put_many(self, requests: Iterable[CacheWriteRequest]) -> list[CacheIndexEntry]:
        """Commit one realized GPU batch in a single LMDB transaction.

        The caller serializes all records before this method returns, then
        appends the corresponding JSONL index in one fsync.  A crash between
        the two leaves only complete LMDB values, which resume reconciliation
        can delete and regenerate as a batch.
        """
        return self.put_many_serialized(
            (
                request,
                serialize_cache_payload(request.arrays, encoding=self.value_encoding),
            )
            for request in requests
        )

    def put_many_serialized(
        self,
        serialized: Iterable[tuple[CacheWriteRequest, bytes]],
    ) -> list[CacheIndexEntry]:
        """Commit one batch whose CPU compression completed before the writer."""
        prepared = list(serialized)
        if not prepared:
            return []
        requests = [request for request, _ in prepared]
        system_ids = [request.system_id for request in requests]
        if len(system_ids) != len(set(system_ids)):
            raise ValueError("An atomic cache batch contains duplicate system IDs.")
        self._ensure_shard(required_records=len(requests))
        assert self._env is not None and self._shard_index is not None
        with self._env.begin(write=True) as transaction:
            for request, payload in prepared:
                inserted = transaction.put(
                    request.system_id.encode("utf-8"), payload, overwrite=False
                )
                if not inserted:
                    raise FileExistsError(
                        f"System {request.system_id!r} already exists in active shard."
                    )
        self._env.sync()
        self._records_in_shard += len(requests)
        shard = self._shard_name(self._shard_index)
        return [
            CacheIndexEntry(
                system_id=request.system_id,
                shard=shard,
                key=request.system_id,
                compressed_bytes=len(payload),
                raw_bytes=sum(
                    np.asarray(value).nbytes for value in request.arrays.values()
                ),
                protein_tokens=request.protein_tokens,
                ligand_tokens=request.ligand_tokens,
                crop_tokens=request.crop_tokens,
                ligand_protein_entropy=float(request.ligand_protein_entropy),
                cache_encoding=self.value_encoding,
                source_tokens=request.source_tokens,
            )
            for request, payload in prepared
        ]

    def close(self) -> None:
        if self._env is not None:
            self._env.sync()
            self._env.close()
        self._env = None

    def __enter__(self) -> FeatureCacheWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class FeatureCacheReader:
    """Read named values from a sharded frozen-feature cache."""

    def __init__(self, root: str | Path, *, max_open_envs: int = 128) -> None:
        if max_open_envs <= 0:
            raise ValueError("max_open_envs must be positive.")
        self.root = Path(root)
        self.max_open_envs = max_open_envs
        self._envs: OrderedDict[str, Any] = OrderedDict()
        self._env_lock = threading.Lock()

    def get(self, entry: CacheIndexEntry | Mapping[str, object]) -> dict[str, np.ndarray]:
        expected_encoding: str | None
        if isinstance(entry, Mapping):
            shard_value = entry.get("cache_shard") or entry.get("shard")
            key_value = entry.get("cache_key") or entry.get("key")
            if shard_value is None or key_value is None:
                raise KeyError(
                    "Cache records require shard/key or cache_shard/cache_key."
                )
            shard = str(shard_value)
            key = str(key_value)
            encoding_value = entry.get("cache_encoding")
            expected_encoding = (
                str(encoding_value) if encoding_value is not None else None
            )
        else:
            shard = entry.shard
            key = entry.key
            expected_encoding = entry.cache_encoding
        # LMDB environments cannot be closed while another thread is using a
        # transaction from them.  Keep LRU lookup, eviction, and the short
        # value-copying transaction under one lock; NPZ decompression remains
        # outside the critical section.
        with self._env_lock:
            env = self._envs.get(shard)
            if env is None:
                lmdb = _lmdb_module()
                path = self.root / shard
                if not path.is_file():
                    raise FileNotFoundError(f"Cache shard does not exist: {path}")
                env = lmdb.open(
                    str(path),
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=512,
                )
                self._envs[shard] = env
                while len(self._envs) > self.max_open_envs:
                    _, stale_env = self._envs.popitem(last=False)
                    stale_env.close()
            else:
                self._envs.move_to_end(shard)
            with env.begin(buffers=False) as transaction:
                payload = transaction.get(key.encode("utf-8"))
        if payload is None:
            raise KeyError(f"Cache key {key!r} is absent from shard {shard!r}.")
        arrays, encoding = deserialize_cache_payload(payload)
        if expected_encoding is not None and encoding != expected_encoding:
            raise ValueError(
                "Affinity cache encoding does not match its manifest row: "
                f"{encoding!r} != {expected_encoding!r}."
            )
        return arrays

    def validate_shards(
        self, entries: Iterable[CacheIndexEntry | Mapping[str, object]]
    ) -> None:
        """Fail before work starts when a referenced cache shard is absent.

        A missing apo-cache root is a configuration/preparation failure, not a
        sequence of record-level failures.  Checking the distinct shard paths
        once avoids filling the feature-cache rejection journal with every
        system that happened to reference the absent root.
        """
        shards: set[str] = set()
        for entry in entries:
            if isinstance(entry, Mapping):
                shard_value = entry.get("cache_shard") or entry.get("shard")
            else:
                shard_value = entry.shard
            if shard_value is None:
                raise KeyError("Cache records require shard or cache_shard.")
            shards.add(str(shard_value))
        missing = sorted(shard for shard in shards if not (self.root / shard).is_file())
        if missing:
            preview = ", ".join(missing[:3])
            suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
            raise FileNotFoundError(
                f"Cache root {self.root} is missing {len(missing)} referenced shard(s): "
                f"{preview}{suffix}"
            )

    def close(self) -> None:
        with self._env_lock:
            for env in self._envs.values():
                env.close()
            self._envs.clear()

    def __enter__(self) -> FeatureCacheReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def write_cache_index_jsonl(entries: Iterable[CacheIndexEntry], path: str | Path) -> None:
    """Write a small append-only index for recovery before Parquet materialization."""
    output = Path(path)
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")


def append_cache_index_rows(rows: Iterable[Mapping[str, object]], handle: Any) -> None:
    """Append index rows and flush them to disk before the next batch starts.

    The Parquet index is only materialized when a partition finishes, so this
    JSONL is what makes a killed run resumable without recomputing features.
    """
    for row in rows:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def read_cache_index_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a partial JSONL index, ignoring a truncated final line."""
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A crash can truncate the last record; drop it and let the
                # LMDB reconciliation regenerate that system.
                break
    return rows


def reconcile_resume_state(
    writer: FeatureCacheWriter,
    index_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str], int]:
    """Return the index rows, completed IDs, and orphaned records removed.

    Truth is the intersection of the LMDB keys and the JSONL index.  Rows whose
    value is missing are dropped; values whose row is missing are deleted so
    the schedule regenerates them.
    """
    stored = writer.keys()
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in index_rows:
        system_id = str(row.get("system_id", ""))
        if system_id in stored and system_id not in seen:
            seen.add(system_id)
            kept.append(row)
    orphaned = stored - seen
    removed = writer.discard(orphaned)
    return kept, seen, removed


def read_cache_bandwidth(
    reader: FeatureCacheReader,
    entries: Iterable[CacheIndexEntry | Mapping[str, object]],
    *,
    max_entries: int | None = None,
) -> dict[str, float | int]:
    """Measure end-to-end LMDB + NPZ read throughput on a bounded sample."""
    selected = list(entries)
    if max_entries is not None:
        selected = selected[:max_entries]
    started = time.perf_counter()
    total_bytes = 0
    for entry in selected:
        arrays = reader.get(entry)
        total_bytes += sum(value.nbytes for value in arrays.values())
    elapsed = time.perf_counter() - started
    return {
        "records": len(selected),
        "decoded_bytes": total_bytes,
        "seconds": elapsed,
        "decoded_bytes_per_second": total_bytes / elapsed if elapsed > 0 else 0.0,
    }


_NF4_CODEBOOK = np.asarray(
    [
        -1.0,
        -0.6961928,
        -0.52507305,
        -0.3949175,
        -0.28444138,
        -0.18477343,
        -0.09105004,
        0.0,
        0.0795803,
        0.1609302,
        0.2461123,
        0.33791524,
        0.44070983,
        0.562617,
        0.72295684,
        1.0,
    ],
    dtype=np.float32,
)


def _channel_scale(values: np.ndarray) -> np.ndarray:
    axes = tuple(range(values.ndim - 1))
    scale = np.max(np.abs(values), axis=axes, keepdims=True).astype(np.float32)
    return np.maximum(scale, np.finfo(np.float32).tiny)


def reconstruct_precision(values: np.ndarray, mode: str) -> np.ndarray:
    """Reconstruct a floating tensor after the candidate cache precision mode.

    INT8 and NF4 use a per-last-channel absolute-max scale.  This matches the
    pair-channel use case more closely than a single tensor-wide scale and is
    intentionally only an audit path; the initial short cache remains FP32.
    """
    source = np.asarray(values, dtype=np.float32)
    if mode == "fp32":
        return source.copy()
    if mode == "fp16":
        return source.astype(np.float16).astype(np.float32)
    scale = _channel_scale(source)
    normalized = np.clip(source / scale, -1.0, 1.0)
    if mode == "int8":
        quantized = np.rint(normalized * 127.0).astype(np.int8)
        return quantized.astype(np.float32) * (scale / 127.0)
    if mode == "nf4":
        distances = np.abs(normalized[..., None] - _NF4_CODEBOOK)
        code = distances.argmin(axis=-1)
        return _NF4_CODEBOOK[code] * scale
    raise ValueError(f"Unsupported precision audit mode: {mode!r}")


def reconstruction_metrics(
    reference: np.ndarray, reconstruction: np.ndarray
) -> dict[str, float]:
    """Report compact numeric fidelity metrics for one reconstructed feature."""
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    reconstruction = np.asarray(reconstruction, dtype=np.float32).reshape(-1)
    if reference.shape != reconstruction.shape:
        raise ValueError("Reference and reconstruction must have equal shape.")
    error = reconstruction - reference
    reference_norm = float(np.linalg.norm(reference))
    reconstruction_norm = float(np.linalg.norm(reconstruction))
    denom = max(reference_norm * reconstruction_norm, np.finfo(np.float32).tiny)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "relative_rmse": float(
            np.linalg.norm(error) / max(reference_norm, np.finfo(np.float32).tiny)
        ),
        "cosine": float(np.dot(reference, reconstruction) / denom),
    }
