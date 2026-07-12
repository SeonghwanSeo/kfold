"""Build the AFDB heterodimer distillation dataset from AFDB-M CIF files.

The output layout matches ``HeterodimerDistillationDataset`` in
``src/kfold/training/dataset/datasets/dimer.py``:

    AFDB-heterodimer/
        structure.lmdb
        manifest.msgpack
        manifest.json
        sequences/all_sequences.fasta
        sequences/sequence.fasta
        sequences/unique_protein_sequences.fasta
        sequences/sequence_mapping.tsv

Each LMDB value is a compressed NPZ with:

    0.sequence, 0.coordinates, 0.b_factors
    1.sequence, 1.coordinates, 1.b_factors
"""

import argparse
import functools
import io
import json
import logging
import multiprocessing
import os
import pathlib
import shutil

import gemmi
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

import kfold.constants as C
from kfold.data.pipelines import cif_factory

DATASET_NAME = "AFDB-heterodimer"

SUCCESS = 0
FILTERED = 1
FAILED = 2

ChainEntry = tuple[gemmi.Entity, gemmi.ResidueSpan]
ChainPayload = tuple[str, np.ndarray, np.ndarray]
ParseResult = tuple[
    int,
    str | None,
    bytes | None,
    dict | None,
    list[dict] | None,
    str | None,
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cif_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing AFDB-M CIF or CIF.GZ files.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help=f"Root processed dataset directory. {DATASET_NAME}/ is appended.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="AFDB-Multimer",
        help="Prediction model name stored in metadata.",
    )
    parser.add_argument(
        "--map_size_gb",
        type=int,
        default=512,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def sample_id_from_path(path: pathlib.Path) -> str:
    name = path.name
    for suffix in (".cif.gz", ".mmcif.gz", ".cif", ".mmcif"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def normalize_protein_residue_name(name: str) -> str:
    name = name.upper()
    if name in C.residue.PROTEIN_RESIDUES_STR_SET:
        return name
    return "UNK"


def protein_three_to_one(name: str) -> str:
    return C.residue.PROTEIN_THREE_TO_ONE[normalize_protein_residue_name(name)]


def extract_chain_payload(
    raw_chain: gemmi.ResidueSpan,
    full_sequence: list[str],
) -> ChainPayload:
    ccd_sequence = [normalize_protein_residue_name(res) for res in full_sequence]
    sequence = "".join(protein_three_to_one(res) for res in ccd_sequence)

    label_seq_to_residue = {}
    for residue in raw_chain:
        if residue.label_seq is None:
            continue
        label_seq_to_residue[residue.label_seq] = residue

    coords = []
    b_factors = []
    for residue_index, res_name in enumerate(ccd_sequence, start=1):
        residue = label_seq_to_residue.get(residue_index)
        name_to_atom = {}
        if residue is not None:
            name_to_atom = {atom.name.upper(): atom for atom in residue}

        for atom_name in C.atom.residue_atoms[res_name]:
            atom = name_to_atom.get(atom_name)
            if atom is None:
                coords.append((np.nan, np.nan, np.nan))
                b_factors.append(np.nan)
                continue

            pos = atom.pos
            coord = (pos.x, pos.y, pos.z)
            coords.append(coord)
            b_factors.append(min(atom.b_iso, 100.0))

    coords_array = np.asarray(coords, dtype=np.float32)
    b_factor_array = np.asarray(b_factors, dtype=np.float32)
    return sequence, coords_array, b_factor_array


def collect_protein_entries(raw_struct: gemmi.Structure) -> list[ChainEntry]:
    chain_spans = {span.subchain_id(): span for span in raw_struct[0].subchains()}
    protein_entries = []
    for entity in raw_struct.entities:
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue
        if entity.polymer_type != gemmi.PolymerType.PeptideL:
            continue

        for subchain_id in entity.subchains:
            raw_chain = chain_spans.get(subchain_id)
            if raw_chain is not None:
                protein_entries.append((entity, raw_chain))

    protein_entries.sort(key=lambda item: item[1].subchain_id())
    return protein_entries


def pack_structure_npz(payloads: list[ChainPayload]) -> tuple[bytes, float | None]:
    npz_arrays = {}
    finite_bfactors = []

    for chain_i, (seq, coords, b_factors) in enumerate(payloads):
        npz_arrays[f"{chain_i}.sequence"] = np.array(seq, dtype="S")
        npz_arrays[f"{chain_i}.coordinates"] = coords
        npz_arrays[f"{chain_i}.b_factors"] = b_factors
        finite_bfactors.append(b_factors[np.isfinite(b_factors)])

    plddt_values = np.concatenate(finite_bfactors)
    plddt = float(plddt_values.mean()) if plddt_values.size > 0 else None

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **npz_arrays)
    return buffer.getvalue(), plddt


def build_metadata(
    sample_id: str,
    model: str,
    plddt: float | None,
    payloads: list[ChainPayload],
) -> dict:
    chains = []
    for asym_id, (seq, coords, _) in enumerate(payloads, start=1):
        chains.append(
            {
                "name": chr(ord("A") + asym_id - 1),
                "type": C.ChainType.PROTEIN.value,
                "entity_id": asym_id,
                "asym_id": asym_id,
                "sym_id": 1,
                "num_residues": len(seq),
                "num_atoms": coords.shape[0],
                "num_tokens": len(seq),
            }
        )

    return {
        "id": sample_id,
        "source": "pred",
        "pred": {"model": model, "plddt": plddt},
        "chains": chains,
        "interfaces": [{"asym_ids": [1, 2]}],
    }


def build_sequence_rows(sample_id: str, payloads: list[ChainPayload]) -> list[dict]:
    return [
        {
            "entry_id": sample_id,
            "chain_name": f"protein_{chain_i}",
            "chain_type": "protein",
            "sequence": seq,
        }
        for chain_i, (seq, _, _) in enumerate(payloads)
    ]


def simplify_structure(
    sample_id: str,
    raw_struct: gemmi.Structure,
    model: str,
) -> tuple[str, bytes, dict, list[dict]] | None:
    protein_entries = collect_protein_entries(raw_struct)
    if len(protein_entries) != 2:
        return None

    payloads = [
        extract_chain_payload(raw_chain, entity.full_sequence)
        for entity, raw_chain in protein_entries
    ]

    seqs = [payload[0] for payload in payloads]
    if seqs[0] == seqs[1]:
        return None

    value, plddt = pack_structure_npz(payloads)
    metadata = build_metadata(sample_id, model, plddt, payloads)
    sequence_rows = build_sequence_rows(sample_id, payloads)
    return sample_id, value, metadata, sequence_rows


def parse_cif(
    cif_path: pathlib.Path,
    model: str,
) -> ParseResult:
    sample_id = sample_id_from_path(cif_path)
    doc = gemmi.cif.read(str(cif_path))
    raw_struct = gemmi.make_structure_from_block(doc[0])
    cif_factory.clean_up_gemmi_structure(raw_struct)

    parsed = simplify_structure(sample_id, raw_struct, model)
    if parsed is None:
        return FILTERED, None, None, None, None, None

    key, value, metadata_dict, sequence_rows = parsed
    return SUCCESS, key, value, metadata_dict, sequence_rows, None


def worker_fn(cif_path: pathlib.Path, model: str) -> ParseResult:
    try:
        return parse_cif(cif_path, model)
    except Exception as exc:
        return FAILED, None, None, None, None, f"{cif_path}: {exc}"


def sequence_output_paths(dataset_dir: pathlib.Path) -> list[pathlib.Path]:
    seq_dir = dataset_dir / "sequences"
    return [
        seq_dir / "all_sequences.fasta",
        seq_dir / "sequence.fasta",
        seq_dir / "unique_protein_sequences.fasta",
        seq_dir / "sequence_mapping.tsv",
    ]


def find_cif_paths(cif_dir: pathlib.Path) -> list[pathlib.Path]:
    patterns = ("*.cif.gz", "*.mmcif.gz", "*.cif", "*.mmcif")
    paths = {path for pattern in patterns for path in cif_dir.glob(pattern)}
    return sorted(paths)


def output_paths(dataset_dir: pathlib.Path) -> list[pathlib.Path]:
    return [
        dataset_dir / "structure.lmdb",
        dataset_dir / "manifest.msgpack",
        dataset_dir / "manifest.json",
        dataset_dir / "errors.log",
        *sequence_output_paths(dataset_dir),
    ]


def prepare_output_dir(path: pathlib.Path, overwrite: bool) -> None:
    if not overwrite:
        for existing_path in output_paths(path):
            if existing_path.exists():
                raise FileExistsError(f"{existing_path} exists. Use --overwrite.")

    if overwrite:
        for output_path in output_paths(path):
            if output_path.is_dir():
                shutil.rmtree(output_path)
            elif output_path.exists():
                output_path.unlink()

    path.mkdir(parents=True, exist_ok=True)


def write_fasta(records: list[tuple[str, str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for name, seq in records:
            f.write(f">{name}\n{seq}\n")


def write_sequence_files(dataset_dir: pathlib.Path, sequence_rows: list[dict]) -> None:
    seq_dir = dataset_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    sequence_rows = sorted(
        sequence_rows,
        key=lambda row: (row["entry_id"], row["chain_name"]),
    )
    all_records = [
        (f"{row['entry_id']}|{row['chain_name']}", row["sequence"])
        for row in sequence_rows
    ]
    write_fasta(all_records, seq_dir / "all_sequences.fasta")
    write_fasta(all_records, seq_dir / "sequence.fasta")

    unique_by_sequence: dict[str, str] = {}
    for row in sequence_rows:
        seq = row["sequence"]
        if seq not in unique_by_sequence:
            unique_by_sequence[seq] = f"uniq_protein_{len(unique_by_sequence) + 1}"
        row["unique_id"] = unique_by_sequence[seq]

    unique_records = [(unique_id, seq) for seq, unique_id in unique_by_sequence.items()]
    write_fasta(unique_records, seq_dir / "unique_protein_sequences.fasta")

    columns = ["entry_id", "chain_name", "chain_type", "unique_id", "sequence"]
    with (seq_dir / "sequence_mapping.tsv").open("w") as f:
        f.write("\t".join(columns) + "\n")
        for row in sequence_rows:
            f.write("\t".join(str(row[col]) for col in columns) + "\n")


def write_manifest_files(dataset_dir: pathlib.Path, metadata_dicts: list[dict]) -> None:
    with (dataset_dir / "manifest.msgpack").open("wb") as f:
        msgpack.pack(metadata_dicts, f)
    with (dataset_dir / "manifest.json").open("w") as f:
        json.dump(metadata_dicts, f, indent=2)


def main():
    args = parse_args()
    cif_paths = find_cif_paths(args.cif_dir)
    if not cif_paths:
        raise FileNotFoundError(f"No CIF files found under {args.cif_dir}")

    data_dir = args.data_dir / DATASET_NAME
    prepare_output_dir(data_dir, args.overwrite)

    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=args.map_size_gb * 1024**3)
    txn = env.begin(write=True)
    metadata_dicts: list[dict] = []
    sequence_rows: list[dict] = []
    status_counts = {SUCCESS: 0, FILTERED: 0, FAILED: 0}
    errors: list[str] = []

    worker = functools.partial(worker_fn, model=args.model)

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = pool.imap_unordered(worker, cif_paths, chunksize=10)
        for status, key, value, metadata_dict, seq_rows, error in tqdm(
            results, total=len(cif_paths), desc="Processing AFDB heterodimers"
        ):
            status_counts[status] += 1
            if status == SUCCESS:
                assert key is not None
                assert value is not None
                assert metadata_dict is not None
                assert seq_rows is not None
                txn.put(key.encode("utf-8"), value)
                metadata_dicts.append(metadata_dict)
                sequence_rows.extend(seq_rows)
                if len(metadata_dicts) % 10_000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            elif error is not None:
                errors.append(error)

    txn.commit()
    env.close()

    metadata_dicts.sort(key=lambda item: item["id"])
    write_manifest_files(data_dir, metadata_dicts)
    write_sequence_files(data_dir, sequence_rows)

    print(f"Total CIF files: {len(cif_paths)}")
    print(f"Successful: {status_counts[SUCCESS]}")
    print(f"Filtered: {status_counts[FILTERED]}")
    print(f"Failed: {status_counts[FAILED]}")
    print(f"Wrote: {lmdb_path}")
    print(f"Wrote: {data_dir / 'manifest.msgpack'}")
    print(f"Wrote: {data_dir / 'manifest.json'}")
    print(f"Wrote: {data_dir / 'sequences'}")
    if errors:
        error_path = data_dir / "errors.log"
        error_path.write_text("\n".join(errors) + "\n")
        print(f"Wrote: {error_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
