"""Create disordered PDB manifest by reusing RCSB train cluster IDs."""

import argparse
import json
import multiprocessing as mp
import pathlib
from collections import defaultdict

import msgpack
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.fasta import read_fasta, write_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory containing disordered_pdb and rcsb-train.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of NPZ parser workers.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report cluster-id extraction statistics without writing manifest files.",
    )
    return parser.parse_args()


def load_rcsb_sequences(path: pathlib.Path) -> dict[str, dict[str, dict[int, str]]]:
    sequences: dict[str, dict[str, dict[int, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for header, sequence in read_fasta(path):
        entry_id, entity_id_str, chain_type = header.split("|")
        sequences[entry_id.lower()][chain_type.lower()][int(entity_id_str)] = sequence
    return {entry_id: dict(by_type) for entry_id, by_type in sequences.items()}


def load_rcsb_cluster_ids(
    manifest_path: pathlib.Path,
) -> dict[str, dict[int, str]]:
    with manifest_path.open("rb") as f:
        manifest = msgpack.unpack(f, raw=False)
    out: dict[str, dict[int, str]] = {}
    for metadata_dict in manifest:
        metadata = Metadata.from_dict(metadata_dict)
        entry_clusters: dict[int, str] = {}
        for chain in metadata.chains:
            if chain.cluster_id is not None:
                entry_clusters[chain.entity_id] = chain.cluster_id
        out[metadata.id.lower()] = entry_clusters
    return out


def chain_type_name(ctype: C.ChainType) -> str:
    if ctype.is_protein:
        return "protein"
    if ctype.is_dna:
        return "dna"
    if ctype.is_rna:
        return "rna"
    return "ligand"


def parse_npz(npz_path: pathlib.Path) -> tuple[str, Metadata, dict[int, tuple[str, str]]]:
    struct = RefStructure.load_npz(npz_path)
    entity_sequences: dict[int, tuple[str, str]] = {}
    for chain in struct.chains:
        if chain.entity_id in entity_sequences:
            continue
        if chain.ctype.is_polymer:
            seq = chain.get_sequence(map_to_standard=True)
        else:
            seq = "-".join(chain.get_ccd_sequence())
        entity_sequences[chain.entity_id] = (chain_type_name(chain.ctype), seq)
    return struct.id, struct.metadata, entity_sequences


def choose_rcsb_entity_id(
    entry_id: str,
    entity_id: int,
    chain_type: str,
    sequence: str,
    rcsb_sequences: dict[str, dict[str, dict[int, str]]],
    rcsb_cluster_ids: dict[str, dict[int, str]],
) -> int | None:
    entry_sequences = rcsb_sequences.get(entry_id, {}).get(chain_type, {})
    cluster_ids = rcsb_cluster_ids.get(entry_id, {})
    if entry_sequences.get(entity_id) == sequence and entity_id in cluster_ids:
        return entity_id
    for rcsb_entity_id, rcsb_sequence in entry_sequences.items():
        if rcsb_sequence == sequence and rcsb_entity_id in cluster_ids:
            return rcsb_entity_id
    return None


def apply_cluster_ids(
    metadata: Metadata,
    entity_sequences: dict[int, tuple[str, str]],
    rcsb_sequences: dict[str, dict[str, dict[int, str]]],
    rcsb_cluster_ids: dict[str, dict[int, str]],
) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    entry_id = metadata.id.lower()
    entity_cluster_ids: dict[int, str] = {}
    asym_to_entity_id: dict[int, int] = {}

    for chain in metadata.chains:
        asym_to_entity_id[chain.asym_id] = chain.entity_id
        if chain.entity_id in entity_cluster_ids:
            chain.cluster_id = entity_cluster_ids[chain.entity_id]
            continue
        chain_type, sequence = entity_sequences[chain.entity_id]
        rcsb_entity_id = choose_rcsb_entity_id(
            entry_id,
            chain.entity_id,
            chain_type,
            sequence,
            rcsb_sequences,
            rcsb_cluster_ids,
        )
        if rcsb_entity_id is None:
            cluster_id = f"disordered_{entry_id}_{chain.entity_id}"
            stats[f"fallback:{chain_type}"] += 1
        else:
            cluster_id = rcsb_cluster_ids[entry_id][rcsb_entity_id]
            stats[f"matched:{chain_type}"] += 1
        entity_cluster_ids[chain.entity_id] = cluster_id
        chain.cluster_id = cluster_id

    for interface in metadata.interfaces:
        entity_id_1 = asym_to_entity_id[interface.asym_ids[0]]
        entity_id_2 = asym_to_entity_id[interface.asym_ids[1]]
        interface.cluster_id = "|".join(
            sorted([entity_cluster_ids[entity_id_1], entity_cluster_ids[entity_id_2]])
        )

    return dict(stats)


def main() -> None:
    args = parse_args()
    disordered_root = args.data_dir / "disordered_pdb"
    rcsb_root = args.data_dir / "rcsb-train"

    rcsb_sequences = load_rcsb_sequences(rcsb_root / "sequences" / "all_sequences.fasta")
    rcsb_cluster_ids = load_rcsb_cluster_ids(rcsb_root / "manifest.msgpack")

    npz_files = sorted((disordered_root / "npz").rglob("*.npz"))
    with mp.Pool(processes=args.num_workers) as pool:
        parsed = list(
            tqdm(
                pool.imap_unordered(parse_npz, npz_files, chunksize=16),
                total=len(npz_files),
                desc="Reading disordered NPZ",
            )
        )

    stats: dict[str, int] = defaultdict(int)
    metadata_by_id: dict[str, Metadata] = {}
    all_sequences: list[tuple[str, str, str, str]] = []
    for entry_id, metadata, entity_sequences in parsed:
        entry_stats = apply_cluster_ids(
            metadata,
            entity_sequences,
            rcsb_sequences,
            rcsb_cluster_ids,
        )
        metadata_by_id[entry_id] = metadata
        for entity_id, (chain_type, sequence) in entity_sequences.items():
            all_sequences.append((entry_id, str(entity_id), chain_type, sequence))
        for key, value in entry_stats.items():
            stats[key] += value

    metadatas = [metadata_by_id[entry_id] for entry_id in sorted(metadata_by_id)]
    metadata_dicts = [metadata.to_dict() for metadata in metadatas]

    print("Manifest extraction complete:")
    print(f"  entries: {len(metadata_dicts)}")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")

    if args.dry_run:
        print("Dry run: manifest and sequence files were not written.")
        return

    seq_dir = disordered_root / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(
        [
            (f"{entry_id}|{entity_id}|{chain_type}", sequence)
            for entry_id, entity_id, chain_type, sequence in sorted(all_sequences)
        ],
        seq_dir / "all_sequences.fasta",
    )
    uniq_proteins = sorted(
        {
            sequence
            for _, _, chain_type, sequence in all_sequences
            if chain_type == "protein"
        },
        key=lambda seq: (len(seq), seq),
    )
    write_fasta(
        [(f"uniq_protein_{i + 1}", sequence) for i, sequence in enumerate(uniq_proteins)],
        seq_dir / "uniq_sequences.fasta",
    )

    with (disordered_root / "manifest.json").open("w") as f:
        json.dump(metadata_dicts, f, indent=2)
    with (disordered_root / "manifest.msgpack").open("wb") as f:
        msgpack.pack(metadata_dicts, f)


if __name__ == "__main__":
    main()
