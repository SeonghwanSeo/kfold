"""Prediction tasks and explicit chain grouping for prepared structure datasets."""

import csv
from dataclasses import dataclass
from pathlib import Path

import msgpack

import kfold.constants as C
from kfold.data.utils.io.fasta import read_fasta

MAX_APO_LENGTH = 1280
GROUP_COLUMNS = ["group_id", "entry_id", "asym_ids"]


@dataclass(frozen=True)
class ApoTask:
    name: str
    sequences: tuple[str, ...]
    kind: str
    # Monomers can share predictions across entries/entities. Multimer tasks
    # refer to exactly one physical chain group with user-defined identifiers.
    targets: tuple[tuple[str, int], ...] = ()
    entry_id: str = ""
    asym_ids: tuple[int, ...] = ()

    @property
    def length(self) -> int:
        return sum(map(len, self.sequences))


def load_prepared_sequences(dataset_dir: Path) -> tuple[dict, dict]:
    with (dataset_dir / "manifest.msgpack").open("rb") as stream:
        manifest = msgpack.unpack(stream, raw=False)
    entries = {record["id"]: record for record in manifest}
    sequences = {}
    for header, sequence in read_fasta(dataset_dir / "sequences/all_sequences.fasta"):
        entry_id, entity_id, kind = header.split("|")
        if kind == "protein" and entry_id in entries:
            key = (entry_id, int(entity_id))
            if key in sequences:
                raise ValueError(f"Duplicate protein entity: {header}")
            sequences[key] = sequence
    for entry_id, entry in entries.items():
        for chain in entry["chains"]:
            if (
                chain["type"] == C.ChainType.PROTEIN.value
                and (entry_id, chain["entity_id"]) not in sequences
            ):
                raise ValueError(
                    f"Missing prepared sequence: {entry_id}/{chain['entity_id']}"
                )
    return entries, sequences


def protein_tasks(dataset_dir: Path) -> list[ApoTask]:
    """Build entity-linked tasks for LMDB construction."""
    _, sequences = load_prepared_sequences(dataset_dir)
    names_by_sequence = {
        task.sequences[0]: task.name for task in protein_prediction_tasks(dataset_dir)
    }
    targets_by_sequence = {}
    for target, sequence in sorted(sequences.items()):
        if not sequence:
            raise ValueError(f"Empty protein sequence: {target}")
        targets_by_sequence.setdefault(sequence, []).append(target)
    return [
        ApoTask(
            name=names_by_sequence[sequence],
            sequences=(sequence,),
            kind="protein",
            targets=tuple(targets),
        )
        for sequence, targets in sorted(targets_by_sequence.items())
    ]


def protein_prediction_tasks(dataset_dir: Path) -> list[ApoTask]:
    """Predict every unique FASTA sequence, without reading dataset metadata."""
    fasta_path = dataset_dir / "sequences/unique_protein_sequences.fasta"
    names_by_sequence = {}
    seen_names = set()
    for header, sequence in read_fasta(fasta_path):
        name = header.split()[0] if header.split() else ""
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Invalid FASTA ID: {header!r}")
        if name in seen_names:
            raise ValueError(f"Duplicate FASTA ID: {name}")
        seen_names.add(name)
        if not sequence:
            raise ValueError(f"Empty protein sequence: {header}")
        names_by_sequence.setdefault(sequence, name)
    return [
        ApoTask(name=name, sequences=(sequence,), kind="protein")
        for sequence, name in sorted(names_by_sequence.items())
    ]


def multimer_tasks(dataset_dir: Path, groups_csv: Path) -> list[ApoTask]:
    entries, sequences = load_prepared_sequences(dataset_dir)
    tasks = []
    names = set()
    used_chains = set()
    with groups_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != GROUP_COLUMNS:
            raise ValueError(f"Group CSV columns must be {GROUP_COLUMNS}")
        for row in reader:
            name = row["group_id"].strip()
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError(f"Invalid group ID: {name!r}")
            if name in names:
                raise ValueError(f"Duplicate group ID: {name}")
            names.add(name)
            entry_id = row["entry_id"].strip()
            asym_ids = tuple(int(value) for value in row["asym_ids"].split(";"))
            if len(asym_ids) != 2 or len(set(asym_ids)) != 2:
                raise ValueError(f"AtlasFold-m requires two distinct chains: {name}")
            chains = {c["asym_id"]: c for c in entries[entry_id]["chains"]}
            selected = [chains[aid] for aid in asym_ids]
            if any(c["type"] != C.ChainType.PROTEIN.value for c in selected):
                raise ValueError(f"Multimer group must contain proteins: {name}")
            if selected[0]["entity_id"] == selected[1]["entity_id"]:
                raise ValueError(
                    f"Repeated entities are unsupported for multimer apo: {name}"
                )
            for aid in asym_ids:
                key = (entry_id, aid)
                if key in used_chains:
                    raise ValueError(f"Overlapping multimer groups for chain {key}")
                used_chains.add(key)
            tasks.append(
                ApoTask(
                    name=name,
                    kind="protein-multimer",
                    entry_id=entry_id,
                    asym_ids=asym_ids,
                    sequences=tuple(
                        sequences[(entry_id, c["entity_id"])] for c in selected
                    ),
                )
            )
    return tasks


def source_name(kind: str, seed: int) -> str:
    model = "atlasfold" if kind == "protein" else "atlasfold-m"
    return f"{model}-seed{seed}"


def prediction_dir(dataset_dir: Path, task: ApoTask, seed: int) -> Path:
    return dataset_dir / "apo" / task.kind / source_name(task.kind, seed) / task.name


def eligible_tasks(
    tasks: list[ApoTask], max_length: int = MAX_APO_LENGTH
) -> list[ApoTask]:
    return [task for task in tasks if 0 < task.length <= max_length]
