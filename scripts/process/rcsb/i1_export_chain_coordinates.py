"""Export chain-level coordinates from the processed RCSB train/val datasets.

Polymer chains are written as residue-major NPZ files. Non-polymer chains are
written as pickled RDKit molecules with an experimental coordinate conformer.
Each PDB directory also contains metadata.json and interfaces.json.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import multiprocessing
import os
import pickle
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.structure import Chain, RefStructure

FORMAT_VERSION = 1
DEFAULT_SOURCE_ROOT = Path("/cache/wykim_lab/kfold_data/v260701_af3")
DEFAULT_OUTPUT_ROOT = Path("/cache/wykim_lab/icl_shwan/share/rcsb-coords")
SPLIT_DIRS = {"train": "rcsb-train", "val": "rcsb-val"}

_WORKER_LMDB: lmdb.Environment | None = None
_WORKER_LMDB_PATH: Path | None = None
_WORKER_OUTPUT_DIR: Path | None = None
_WORKER_SPLIT: str | None = None
_WORKER_OVERWRITE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export residue-level RCSB coordinates one chain at a time."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Processed K-Fold root containing dataset/rcsb-{train,val}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root containing train/ and val/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_DIRS),
        default=list(SPLIT_DIRS),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(32, len(os.sched_getaffinity(0))),
    )
    parser.add_argument(
        "--pdb-id",
        action="append",
        dest="pdb_ids",
        help="Export only this PDB ID. May be passed more than once.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        help="Export at most this many entries from each selected split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite entries that already have a complete output record.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed PDB instead of recording the error.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.max_entries is not None and args.max_entries < 1:
        parser.error("--max-entries must be at least 1")
    return args


def atom_representation(ctype: C.ChainType) -> tuple[str, int]:
    if ctype.is_protein:
        return "atom14", max(
            len(C.atom.RESIDUE_ATOMS[res]) for res in C.residue.PROTEIN_RESIDUES
        )
    if ctype.is_dna:
        return "atom22", max(
            len(C.atom.RESIDUE_ATOMS[res]) for res in C.residue.DNA_RESIDUES
        )
    if ctype.is_rna:
        return "atom23", max(
            len(C.atom.RESIDUE_ATOMS[res]) for res in C.residue.RNA_RESIDUES
        )
    raise ValueError(f"Non-polymer chain has no atom representation: {ctype}")


def normalize_residue_name(ccd_name: str, ctype: C.ChainType) -> C.ResidueName:
    """Map a polymer CCD name to its standard parent residue type."""
    return C.residue.get_residue_name_with_unk(
        ccd_name,
        ctype,
        map_ambiguous_to_standard=True,
    )


def polymer_npz_dict(chain: Chain) -> dict[str, np.ndarray]:
    """Convert one polymer chain to padded residue-major coordinates."""
    if not chain.is_polymer:
        raise ValueError("polymer_npz_dict requires a polymer chain")

    representation, num_atom_slots = atom_representation(chain.ctype)
    num_residues = chain.num_residues
    coords = np.full((num_residues, num_atom_slots, 3), np.nan, dtype=np.float32)
    bfactor = np.full((num_residues, num_atom_slots), np.nan, dtype=np.float16)
    atom_exists = np.zeros((num_residues, num_atom_slots), dtype=bool)
    atom_mask = np.zeros((num_residues, num_atom_slots), dtype=bool)
    residue_type = np.empty((num_residues,), dtype=np.uint8)
    residue_names: list[str] = []

    for res_i, source_name in enumerate(chain.residue.name.tolist()):
        normalized = normalize_residue_name(source_name, chain.ctype)
        target_atoms = C.atom.RESIDUE_ATOMS[normalized]
        residue_type[res_i] = normalized.value
        residue_names.append(normalized.name)
        atom_exists[res_i, : len(target_atoms)] = True

        source_slice = chain.residue.get_atom_slice(res_i + 1)
        source_indices = {
            name: source_slice.start + offset
            for offset, name in enumerate(chain.atom.name[source_slice].tolist())
        }
        for atom_slot, atom_name in enumerate(target_atoms):
            source_atom_i = source_indices.get(atom_name.value)
            if source_atom_i is None:
                continue
            atom_coords = chain.atom.coords[source_atom_i]
            coords[res_i, atom_slot] = atom_coords
            bfactor[res_i, atom_slot] = chain.atom.bfactor[source_atom_i]
            atom_mask[res_i, atom_slot] = bool(np.isfinite(atom_coords).all())

    return {
        "format_version": np.array(FORMAT_VERSION, dtype=np.uint8),
        "representation": np.array(representation),
        "chain_type": np.array(chain.chain_type, dtype=np.uint8),
        "entity_id": np.array(chain.entity_id, dtype=np.uint16),
        "asym_id": np.array(chain.asym_id, dtype=np.uint16),
        "sym_id": np.array(chain.sym_id, dtype=np.uint16),
        "residue_index": np.arange(1, num_residues + 1, dtype=np.uint32),
        "residue_type": residue_type,
        "residue_name": np.asarray(residue_names, dtype="<U3"),
        "source_residue_name": chain.residue.name.astype("<U6", copy=True),
        "coords": coords,
        "atom_exists": atom_exists,
        "atom_mask": atom_mask,
        "bfactor": bfactor,
    }


def ligand_to_mol(chain: Chain, pdb_id: str) -> Chem.Mol:
    """Build an RDKit molecule in the exact atom order stored by K-Fold."""
    if not chain.is_nonpolymer:
        raise ValueError("ligand_to_mol requires a non-polymer chain")

    editable = Chem.RWMol()
    atom_indices: dict[tuple[int, str], int] = {}
    for res_i in range(chain.num_residues):
        atom_slice = chain.residue.get_atom_slice(res_i + 1)
        residue_name = str(chain.residue.name[res_i])
        for source_atom_i in range(atom_slice.start, atom_slice.stop):
            atom = Chem.Atom(int(chain.atom.element[source_atom_i]))
            atom.SetFormalCharge(int(chain.atom.charge[source_atom_i]))
            atom.SetNoImplicit(True)
            atom_name = str(chain.atom.name[source_atom_i])
            atom.SetProp("atom_name", atom_name)
            atom.SetProp("residue_name", residue_name)
            atom.SetIntProp("residue_index", res_i + 1)
            mol_atom_i = editable.AddAtom(atom)
            atom_indices[(res_i + 1, atom_name)] = mol_atom_i

    for bond_i in range(chain.num_bonds):
        key1 = (
            int(chain.bond.residue_index[bond_i, 0]),
            str(chain.bond.atom_name[bond_i, 0]),
        )
        key2 = (
            int(chain.bond.residue_index[bond_i, 1]),
            str(chain.bond.atom_name[bond_i, 1]),
        )
        atom_i = atom_indices[key1]
        atom_j = atom_indices[key2]
        bond_type = Chem.BondType.values[int(chain.bond.bond_type[bond_i])]
        editable.AddBond(atom_i, atom_j, bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            editable.GetAtomWithIdx(atom_i).SetIsAromatic(True)
            editable.GetAtomWithIdx(atom_j).SetIsAromatic(True)

    mol = editable.GetMol()
    conformer = Chem.Conformer(chain.num_atoms)
    conformer.Set3D(True)
    for atom_i, (x, y, z) in enumerate(chain.atom.coords):
        conformer.SetAtomPosition(atom_i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conformer, assignId=True)

    mol.SetProp("pdb_id", pdb_id)
    mol.SetIntProp("asym_id", chain.asym_id)
    mol.SetIntProp("entity_id", chain.entity_id)
    mol.SetIntProp("sym_id", chain.sym_id)
    mol.SetProp("chain_type", chain.ctype.name.lower())
    mol.SetProp("subchain_type", chain.subtype.name.lower())
    if chain.smiles is not None:
        mol.SetProp("source_smiles", chain.smiles)
    return mol


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temp_path = Path(f.name)
    try:
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Chem.Mol):
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pkl", delete=False) as f:
        temp_path = Path(f.name)
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".json",
        delete=False,
    ) as f:
        temp_path = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def chain_output_filename(pdb_id: str, chain_type: int, asym_id: int) -> str:
    extension = ".npz" if C.ChainType(chain_type).is_polymer else ".pkl"
    return f"{pdb_id}_{asym_id}{extension}"


def entry_is_complete(entry_dir: Path, record: dict[str, Any]) -> bool:
    metadata_path = entry_dir / "metadata.json"
    interfaces_path = entry_dir / "interfaces.json"
    if not metadata_path.is_file() or not interfaces_path.is_file():
        return False
    expected_files = [
        chain_output_filename(record["id"].lower(), chain["type"], chain["asym_id"])
        for chain in record["chains"]
    ]
    if not all((entry_dir / filename).is_file() for filename in expected_files):
        return False
    try:
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("format_version") == FORMAT_VERSION
        and metadata.get("complete") is True
        and metadata.get("pdb_id") == record["id"].lower()
    )


def _connection_dict(connection: Any) -> dict[str, Any]:
    return {
        "asym_ids": list(connection.asym_id),
        "residue_indices": list(connection.residue_index),
        "atom_names": list(connection.atom_names),
    }


def export_entry(
    structure: RefStructure,
    record: dict[str, Any],
    output_split_dir: Path,
    split: str,
    overwrite: bool = False,
) -> tuple[str, int, int]:
    """Export a structure and return status, polymer count, ligand count."""
    pdb_id = record["id"].lower()
    entry_dir = output_split_dir / pdb_id
    if not overwrite and entry_is_complete(entry_dir, record):
        return "skipped", 0, 0

    structure_by_asym_id = {chain.asym_id: chain for chain in structure.chains}
    metadata_asym_ids = {int(chain["asym_id"]) for chain in record["chains"]}
    if metadata_asym_ids != set(structure_by_asym_id):
        raise ValueError(
            f"{pdb_id}: manifest/structure asym_id mismatch: "
            f"{sorted(metadata_asym_ids)} != {sorted(structure_by_asym_id)}"
        )

    output_chains: list[dict[str, Any]] = []
    polymer_count = 0
    ligand_count = 0
    for chain_record in record["chains"]:
        asym_id = int(chain_record["asym_id"])
        chain = structure_by_asym_id[asym_id]
        filename = chain_output_filename(pdb_id, chain.chain_type, asym_id)
        output_path = entry_dir / filename
        output_record = copy.deepcopy(chain_record)
        output_record["chain_type_name"] = chain.ctype.name.lower()
        output_record["subchain_type_name"] = chain.subtype.name.lower()
        output_record["file"] = filename

        if chain.is_polymer:
            representation, _ = atom_representation(chain.ctype)
            _atomic_npz(output_path, polymer_npz_dict(chain))
            output_record["storage"] = "npz"
            output_record["representation"] = representation
            polymer_count += 1
        else:
            _atomic_pickle(output_path, ligand_to_mol(chain, pdb_id))
            output_record["storage"] = "rdkit_pickle"
            ligand_count += 1
        output_chains.append(output_record)

    interfaces = []
    chain_file_by_asym_id = {
        int(chain["asym_id"]): chain["file"] for chain in output_chains
    }
    for interface in record.get("interfaces", []):
        output_interface = copy.deepcopy(interface)
        asym_ids = [int(value) for value in interface["asym_ids"]]
        output_interface["asym_ids"] = asym_ids
        output_interface["chain_files"] = [
            chain_file_by_asym_id[asym_id] for asym_id in asym_ids
        ]
        interfaces.append(output_interface)

    interface_record = {
        "format_version": FORMAT_VERSION,
        "pdb_id": pdb_id,
        "interfaces": interfaces,
    }
    _atomic_json(entry_dir / "interfaces.json", interface_record)

    metadata = copy.deepcopy(record)
    metadata["format_version"] = FORMAT_VERSION
    metadata["complete"] = True
    metadata["split"] = split
    metadata["pdb_id"] = pdb_id
    metadata["chains"] = output_chains
    metadata["covalent_connections"] = [
        _connection_dict(connection) for connection in structure.connections
    ]
    _atomic_json(entry_dir / "metadata.json", metadata)
    return "exported", polymer_count, ligand_count


def _init_worker(
    lmdb_path: str,
    output_split_dir: str,
    split: str,
    overwrite: bool,
) -> None:
    global _WORKER_LMDB
    global _WORKER_LMDB_PATH, _WORKER_OUTPUT_DIR, _WORKER_SPLIT, _WORKER_OVERWRITE
    if _WORKER_LMDB is not None:
        _WORKER_LMDB.close()
        _WORKER_LMDB = None
    _WORKER_LMDB_PATH = Path(lmdb_path)
    _WORKER_OUTPUT_DIR = Path(output_split_dir)
    _WORKER_SPLIT = split
    _WORKER_OVERWRITE = overwrite


def _get_worker_lmdb() -> lmdb.Environment:
    global _WORKER_LMDB
    if _WORKER_LMDB is None:
        assert _WORKER_LMDB_PATH is not None
        _WORKER_LMDB = lmdb.open(
            str(_WORKER_LMDB_PATH),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    return _WORKER_LMDB


def _worker(record: dict[str, Any]) -> dict[str, Any]:
    pdb_id = record["id"].lower()
    try:
        assert _WORKER_OUTPUT_DIR is not None and _WORKER_SPLIT is not None
        entry_dir = _WORKER_OUTPUT_DIR / pdb_id
        if not _WORKER_OVERWRITE and entry_is_complete(entry_dir, record):
            return {
                "pdb_id": pdb_id,
                "status": "skipped",
                "polymer_chains": 0,
                "ligand_chains": 0,
            }

        env = _get_worker_lmdb()
        with env.begin() as txn:
            value = txn.get(pdb_id.encode())
        if value is None:
            raise KeyError(f"PDB ID missing from structure.lmdb: {pdb_id}")
        structure = RefStructure.load_npz(io.BytesIO(value))
        status, polymers, ligands = export_entry(
            structure,
            record,
            _WORKER_OUTPUT_DIR,
            _WORKER_SPLIT,
            overwrite=_WORKER_OVERWRITE,
        )
        return {
            "pdb_id": pdb_id,
            "status": status,
            "polymer_chains": polymers,
            "ligand_chains": ligands,
        }
    except Exception:
        return {
            "pdb_id": pdb_id,
            "status": "failed",
            "error": traceback.format_exc(),
            "polymer_chains": 0,
            "ligand_chains": 0,
        }


def _residue_atom_table(residues: tuple[C.ResidueName, ...], width: int) -> str:
    header = "| residue | " + " | ".join(str(i) for i in range(width)) + " |"
    separator = "|---|" + "---:|" * width
    rows = [header, separator]
    for residue in residues:
        atoms = [atom.value for atom in C.atom.RESIDUE_ATOMS[residue]]
        atoms.extend(["-"] * (width - len(atoms)))
        rows.append(f"| {residue.name} | " + " | ".join(atoms) + " |")
    return "\n".join(rows)


def readme_text() -> str:
    protein_width = atom_representation(C.ChainType.PROTEIN)[1]
    dna_width = atom_representation(C.ChainType.DNA)[1]
    rna_width = atom_representation(C.ChainType.RNA)[1]
    return f"""# RCSB chain coordinates

Generated from K-Fold's processed `rcsb-train` and `rcsb-val` datasets. The
first biological assembly, alternative-conformation cleanup, water removal,
and structure filtering were performed by the upstream K-Fold RCSB pipeline.

## Layout

```text
<root>/<split>/<pdb>/<pdb>_<asym_id>.npz  # protein, DNA, or RNA
<root>/<split>/<pdb>/<pdb>_<asym_id>.pkl  # non-polymer RDKit Chem.Mol
<root>/<split>/<pdb>/metadata.json
<root>/<split>/<pdb>/interfaces.json
```

`split` is `train` or `val`; PDB IDs are lowercase. The numeric K-Fold
`asym_id` is used as the chain ID. Author and label chain IDs remain available
in `metadata.json`.

## Polymer NPZ

All coordinates are float32 and use Angstrom units. Protein, DNA, and RNA use
shapes `[N_residue, 14, 3]`, `[N_residue, 22, 3]`, and
`[N_residue, 23, 3]`, respectively. These sizes and orders come directly from
`kfold.constants.atom.RESIDUE_ATOMS`.

- `coords`: padded residue-major coordinates; missing values are `NaN`.
- `atom_exists`: slots defined by the normalized residue template.
- `atom_mask`: slots whose experimental coordinates are finite.
- `bfactor`: B-factors aligned to `coords`; missing values are `NaN`.
- `residue_index`: 1-based K-Fold/label sequence index.
- `residue_type`: integer `kfold.constants.ResidueName` value.
- `residue_name`: normalized standard residue name.
- `source_residue_name`: original CCD residue name before PTM normalization.
- `chain_type`, `entity_id`, `asym_id`, `sym_id`: scalar identifiers.
- `representation`: `atom14`, `atom22`, or `atom23`.

Modified polymer residues are forced to one residue token. K-Fold's CCD
one-letter mapping selects the standard parent restype. Only atom names present
in that parent template are copied; modification-only atoms are discarded. An
unmapped residue becomes `UNK`, `DN`, or `N` according to polymer type.

### Protein atom14 order

{_residue_atom_table(C.residue.PROTEIN_RESIDUES, protein_width)}

### DNA atom22 order

{_residue_atom_table(C.residue.DNA_RESIDUES, dna_width)}

### RNA atom23 order

{_residue_atom_table(C.residue.RNA_RESIDUES, rna_width)}

## Non-polymer pickle

Each `.pkl` contains an RDKit `Chem.Mol`, not a wrapper dictionary. Atom order
matches the source K-Fold chain. The molecule contains one 3D conformer with
the experimental coordinates. Atom properties include `atom_name`,
`residue_name`, and 1-based `residue_index`. Molecule properties include the
PDB and chain identifiers. Hydrogens and waters were removed upstream.

Connections to other chains cannot be represented inside a chain-local RDKit
molecule; they are listed in `metadata.json` under `covalent_connections`.

## Metadata and interfaces

`metadata.json` preserves the source manifest record, records the generated
file for every `asym_id`, and adds any covalent connections from the structure.
`interfaces.json` preserves source interface annotations and adds the two chain
filenames. Interfaces are not recomputed during export.
"""


def write_readme(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    readme_path = output_root / "README.md"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_root,
        suffix=".md",
        delete=False,
    ) as f:
        temp_path = Path(f.name)
        f.write(readme_text())
    try:
        os.replace(temp_path, readme_path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        manifest = msgpack.unpack(f, raw=False)
    if not isinstance(manifest, list):
        raise TypeError(f"Expected list manifest at {path}, got {type(manifest)}")
    return manifest


def run_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    source_dir = args.source_root / "dataset" / SPLIT_DIRS[split]
    lmdb_path = source_dir / "structure.lmdb"
    manifest_path = source_dir / "manifest.msgpack"
    if not lmdb_path.is_dir():
        raise FileNotFoundError(lmdb_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = load_manifest(manifest_path)
    source_manifest_count = len(manifest)
    if args.pdb_ids:
        selected_ids = {pdb_id.lower() for pdb_id in args.pdb_ids}
        manifest = [record for record in manifest if record["id"].lower() in selected_ids]
        found_ids = {record["id"].lower() for record in manifest}
        missing_ids = selected_ids - found_ids
        if missing_ids:
            raise KeyError(f"PDB IDs missing from {split}: {sorted(missing_ids)}")
    if args.max_entries is not None:
        manifest = manifest[: args.max_entries]

    output_split_dir = args.output_root / split
    output_split_dir.mkdir(parents=True, exist_ok=True)
    initargs = (str(lmdb_path), str(output_split_dir), split, args.overwrite)
    statuses: Counter[str] = Counter()
    polymer_chains = 0
    ligand_chains = 0
    errors = []

    if args.num_workers == 1:
        _init_worker(*initargs)
        results = map(_worker, manifest)
        pool = None
    else:
        pool = multiprocessing.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=initargs,
        )
        results = pool.imap_unordered(_worker, manifest, chunksize=8)

    try:
        for result in tqdm(results, total=len(manifest), desc=f"Exporting {split}"):
            status = result["status"]
            statuses[status] += 1
            polymer_chains += result["polymer_chains"]
            ligand_chains += result["ligand_chains"]
            if status == "failed":
                errors.append({"pdb_id": result["pdb_id"], "error": result["error"]})
                if args.fail_fast:
                    raise RuntimeError(result["error"])
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    summary = {
        "format_version": FORMAT_VERSION,
        "split": split,
        "source_dir": str(source_dir),
        "source_manifest_entries": source_manifest_count,
        "selected_entries": len(manifest),
        "statuses": dict(sorted(statuses.items())),
        "exported_polymer_chains": polymer_chains,
        "exported_ligand_chains": ligand_chains,
        "errors": errors,
    }
    _atomic_json(output_split_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    write_readme(args.output_root)
    failed = 0
    for split in args.splits:
        summary = run_split(args, split)
        print(json.dumps(summary, indent=2))
        failed += summary["statuses"].get("failed", 0)
    if failed:
        raise SystemExit(f"Export completed with {failed} failed PDB entries")


if __name__ == "__main__":
    main()
