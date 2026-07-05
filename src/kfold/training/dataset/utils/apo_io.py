"""Serialization helpers for apo LMDB records."""

from __future__ import annotations

import io

import numpy as np

import kfold.constants as C

CHAIN_TYPE_NAMES: dict[C.ChainType, str] = {
    C.ChainType.PROTEIN: "protein",
    C.ChainType.DNA: "dna",
    C.ChainType.RNA: "rna",
}


def chain_type_to_name(chain_type: C.ChainType | int | str) -> str:
    """Normalize a chain type to the lowercase lookup/LMDB name."""
    if isinstance(chain_type, str):
        return chain_type.lower()
    return CHAIN_TYPE_NAMES[C.ChainType(chain_type)]


def decode_string_array(value: np.ndarray) -> str:
    """Decode a scalar or character-array string stored in npz."""
    if value.ndim == 0:
        return str(value.astype(str).item())
    return "".join(value.astype(str).tolist())


def encode_sequence(sequence: str) -> np.ndarray:
    """Encode a biological sequence as a one-character string array."""
    return np.array(list(sequence), dtype=np.dtype("S1"))


def pack_apo_record(
    sequence: str,
    coords: np.ndarray,
    chain_type: C.ChainType | int | str,
) -> bytes:
    """Pack one monomer apo record as npz bytes."""
    with io.BytesIO() as buffer:
        np.savez_compressed(
            buffer,
            seq=encode_sequence(sequence),
            coords=coords,
            chain_type=np.array(chain_type_to_name(chain_type)),
        )
        return buffer.getvalue()


def unpack_apo_record(value: bytes) -> dict:
    """Unpack one monomer apo record from npz bytes."""
    with io.BytesIO(value) as byte_stream:
        with np.load(byte_stream) as data:
            record = {
                "seq": decode_string_array(data["seq"]),
                "coords": data["coords"].copy(),
            }
            if "chain_type" in data:
                record["chain_type"] = decode_string_array(data["chain_type"])
            return record


def pack_prior_stack_record(
    sequence: str,
    coords: np.ndarray,
    chain_type: C.ChainType | int | str,
    sample_names: list[str],
    ptm: np.ndarray | list[float] | None = None,
    avg_plddt: np.ndarray | list[float] | None = None,
) -> bytes:
    """Pack stacked prior samples as npz bytes."""
    arrays: dict[str, np.ndarray] = {
        "seq": encode_sequence(sequence),
        "coords": coords.astype(np.float32, copy=False),
        "chain_type": np.array(chain_type_to_name(chain_type)),
        "sample_names": np.array(sample_names),
    }
    if ptm is not None:
        arrays["ptm"] = np.asarray(ptm, dtype=np.float32)
    if avg_plddt is not None:
        arrays["avg_plddt"] = np.asarray(avg_plddt, dtype=np.float32)

    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()


def unpack_prior_stack_record(value: bytes) -> dict:
    """Unpack stacked prior samples from npz bytes."""
    with io.BytesIO(value) as byte_stream:
        with np.load(byte_stream) as data:
            record = {
                "seq": decode_string_array(data["seq"]),
                "coords": data["coords"].copy(),
                "chain_type": decode_string_array(data["chain_type"]),
                "sample_names": data["sample_names"].astype(str).tolist(),
            }
            if "ptm" in data:
                record["ptm"] = data["ptm"].copy()
            if "avg_plddt" in data:
                record["avg_plddt"] = data["avg_plddt"].copy()
            return record


def pack_apo_multimer_record(chains: dict[int, dict]) -> bytes:
    """Pack a multimer apo record as npz bytes.

    The public schema is one LMDB value per complex sample.  The value stores
    chain boundaries explicitly so runtime code can extract the chains that
    survived sub-complex sampling/cropping.
    """
    chain_ids = sorted(chains)
    arrays: dict[str, np.ndarray] = {
        "format": np.array("apo_multimer_npz_v1"),
        "chain_ids": np.array(chain_ids),
    }
    for i, chain_id in enumerate(chain_ids):
        chain = chains[chain_id]
        arrays[f"seq_{i}"] = encode_sequence(chain["seq"])
        arrays[f"coords_{i}"] = chain["coords"]
        arrays[f"chain_type_{i}"] = np.array(chain_type_to_name(chain["chain_type"]))

    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()


def unpack_apo_multimer_record(value: bytes) -> dict[int, dict]:
    """Unpack one multimer apo record from npz bytes."""
    with io.BytesIO(value) as byte_stream:
        with np.load(byte_stream) as data:
            chain_ids = [int(chain_id) for chain_id in data["chain_ids"].tolist()]
            chains: dict[int, dict] = {}
            for i, chain_id in enumerate(chain_ids):
                chains[chain_id] = {
                    "seq": decode_string_array(data[f"seq_{i}"]),
                    "coords": data[f"coords_{i}"].copy(),
                    "chain_type": decode_string_array(data[f"chain_type_{i}"]),
                }
            return chains


def pack_prior_multimer_stack_record(
    chains: dict[int, dict],
    sample_names: list[str],
    ptm: np.ndarray | list[float] | None = None,
    avg_plddt: np.ndarray | list[float] | None = None,
) -> bytes:
    """Pack stacked multimer prior samples as npz bytes."""
    chain_ids = sorted(chains)
    arrays: dict[str, np.ndarray] = {
        "format": np.array("prior_multimer_stack_npz_v1"),
        "chain_ids": np.array(chain_ids),
        "sample_names": np.array(sample_names),
    }
    for i, chain_id in enumerate(chain_ids):
        chain = chains[chain_id]
        arrays[f"seq_{i}"] = encode_sequence(chain["seq"])
        arrays[f"coords_{i}"] = chain["coords"].astype(np.float32, copy=False)
        arrays[f"chain_type_{i}"] = np.array(chain_type_to_name(chain["chain_type"]))
    if ptm is not None:
        arrays["ptm"] = np.asarray(ptm, dtype=np.float32)
    if avg_plddt is not None:
        arrays["avg_plddt"] = np.asarray(avg_plddt, dtype=np.float32)

    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()


def unpack_prior_multimer_stack_record(value: bytes) -> dict:
    """Unpack one stacked multimer prior record from npz bytes."""
    with io.BytesIO(value) as byte_stream:
        with np.load(byte_stream) as data:
            chain_ids = [int(chain_id) for chain_id in data["chain_ids"].tolist()]
            chains: dict[int, dict] = {}
            for i, chain_id in enumerate(chain_ids):
                chains[chain_id] = {
                    "seq": decode_string_array(data[f"seq_{i}"]),
                    "coords": data[f"coords_{i}"].copy(),
                    "chain_type": decode_string_array(data[f"chain_type_{i}"]),
                }
            record = {
                "chains": chains,
                "sample_names": data["sample_names"].astype(str).tolist(),
            }
            if "ptm" in data:
                record["ptm"] = data["ptm"].copy()
            if "avg_plddt" in data:
                record["avg_plddt"] = data["avg_plddt"].copy()
            return record


def pack_apo_multimer_token_record(chains: dict[int, np.ndarray]) -> bytes:
    """Pack multimer apo structure tokens keyed by internal asym_id."""
    chain_ids = sorted(chains)
    arrays: dict[str, np.ndarray] = {
        "format": np.array("apo_multimer_token_npz_v1"),
        "chain_ids": np.array(chain_ids),
    }
    for i, chain_id in enumerate(chain_ids):
        arrays[f"tokens_{i}"] = chains[chain_id].astype(np.int16, copy=False)

    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()


def unpack_apo_multimer_token_record(value: bytes) -> dict[int, np.ndarray]:
    """Unpack multimer apo structure tokens keyed by internal asym_id."""
    with io.BytesIO(value) as byte_stream:
        with np.load(byte_stream) as data:
            chain_ids = [int(chain_id) for chain_id in data["chain_ids"].tolist()]
            return {
                chain_id: data[f"tokens_{i}"].astype(np.int16, copy=True)
                for i, chain_id in enumerate(chain_ids)
            }
