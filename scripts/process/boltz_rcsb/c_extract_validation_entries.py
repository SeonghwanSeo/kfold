import io
from pathlib import Path

import lmdb
from tqdm import tqdm

from kfold.data.structure import TokenizedStructure
from kfold.training.folding.dataset.datamodule import load_manifest

LMDB_PATH = Path("/cache/wykim_lab/kfold_data/kfold_rcsb_processed_v251120.lmdb/")
MANIFEST_PATH = Path("/cache/wykim_lab/kfold_data/manifests/af3_manifest.pkl")
BOLTZ1_SPLIT_PATH = Path("./assets/splits/boltz1/")
NEW_SPLIT_PATH = Path("./assets/splits/kfold_v251213/")


if __name__ == "__main__":
    manifest = load_manifest(MANIFEST_PATH)
    with open(BOLTZ1_SPLIT_PATH / "validation_ids.txt") as f:
        val_ids = set(line.strip().lower() for line in f.readlines())

    val_records = [rec for rec in manifest if rec.id in val_ids]
    print(f"Number of validation records: {len(val_records)}")

    dna_rna_entries: list[str] = []
    multi_polymer_entries: list[str] = []
    single_protein_entries: list[str] = []
    single_protein_ligand_entries: list[str] = []
    single_protein_ion_entries: list[str] = []
    excluded_entries: list[str] = []

    # For priority
    num_chains: dict[str, int] = {}
    num_residues: dict[str, int] = {}

    env = lmdb.open(
        str(LMDB_PATH),
        map_size=1024**4,  # 1 TB
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    with env.begin(write=False) as txn:
        for record in tqdm(val_records):
            byte_data = txn.get(record.id.encode("utf-8"))
            with io.BytesIO(byte_data) as byte_stream:
                struct = TokenizedStructure.load_npz(byte_stream)

            num_chains[record.id] = struct.num_chains
            num_residues[record.id] = struct.num_residues

            if struct.chain.is_dna.any() or struct.chain.is_rna.any():
                # DNA/RNA present
                dna_rna_entries.append(record.id)
                continue

            if struct.chain.is_ligand.all():
                # Non-polymer only
                excluded_entries.append(record.id)
                continue

            is_polymer = (
                struct.chain.is_protein | struct.chain.is_dna | struct.chain.is_rna
            )
            num_polymer_chains = is_polymer.sum().item()
            if num_polymer_chains > 1:
                # Multi-polymer
                multi_polymer_entries.append(record.id)
                continue

            if struct.num_chains == 1 and struct.chain.is_protein[0]:
                # Single protein only
                single_protein_entries.append(record.id)
                continue

            # get all ccd
            ccd_list = []
            for i in range(struct.num_chains):
                if struct.chain.is_protein[i]:
                    continue
                res_i = struct.chain.residue_start[i]
                ccd = str(struct.residue.name[res_i]).strip()
                ccd_list.append(ccd)
            assert len(ccd_list) >= 1
            if any(len(ccd) > 2 for ccd in ccd_list):
                single_protein_ligand_entries.append(record.id)
            else:
                single_protein_ion_entries.append(record.id)
    env.close()

    print("DNA/RNA entries:", len(dna_rna_entries))
    print("Multi-polymer entries:", len(multi_polymer_entries))
    print("Single protein entries:", len(single_protein_entries))
    print("Single protein with ligand entries:", len(single_protein_ligand_entries))
    print("Single protein with ion entries:", len(single_protein_ion_entries))
    print("Excluded entries:", len(excluded_entries))
    print()

    print("Extracting new validation split...")
    print("Original validation split size:", len(val_records))
    new_split = dna_rna_entries + multi_polymer_entries + single_protein_ligand_entries
    print("Validation split size (complex entries):", len(new_split))

    # Add additional entries until we reach 384 entries
    target_size = 384

    remaining_entries = single_protein_ion_entries

    # filter with residue cutoff
    remaining_entries = [
        entry for entry in remaining_entries if num_residues[entry] >= 256
    ]
    print(
        "Remaining single protein with ion entries after residue cutoff:",
        len(remaining_entries),
    )

    # Sort by number of chains (descending)
    remaining_entries.sort(key=lambda x: num_chains[x], reverse=True)
    for entry in remaining_entries:
        if len(new_split) >= target_size:
            break
        new_split.append(entry)

    print("Final validation split size:", len(new_split))
    new_split = sorted(new_split)
    NEW_SPLIT_PATH.mkdir(parents=True, exist_ok=True)
    new_split_path = NEW_SPLIT_PATH / "validation_ids.txt"
    with open(new_split_path, "w") as f:
        for entry in new_split:
            f.write(f"{entry}\n")
