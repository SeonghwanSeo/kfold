# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export sequences with the final entity IDs from prepared RefStructures."""

from pathlib import Path

from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.fasta import write_fasta


def structure_sequences(structure: RefStructure) -> list[tuple[str, str]]:
    entities = {}
    for chain in structure.chains:
        if chain.entity_id in entities:
            continue
        sequence = (
            chain.get_sequence(map_to_standard=True)
            if chain.ctype.is_polymer
            else "-".join(chain.get_ccd_sequence())
        )
        header = f"{structure.id}|{chain.entity_id}|{chain.ctype.name.lower()}"
        entities[chain.entity_id] = (header, sequence)
    return [entities[eid] for eid in sorted(entities)]


def save_sequences(sequences: list[tuple[str, str]], dataset_dir: Path):
    sequences = sorted(sequences)
    directory = dataset_dir / "sequences"
    directory.mkdir(parents=True, exist_ok=True)
    write_fasta(sequences, directory / "all_sequences.fasta")
    unique = {}
    for header, sequence in sequences:
        if header.rsplit("|", 1)[1] == "protein":
            unique.setdefault(sequence, header)
    write_fasta(
        [(header, seq) for seq, header in unique.items()],
        directory / "unique_protein_sequences.fasta",
    )
