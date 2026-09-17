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

"""Strict JSON/YAML query schema. See docs/inference.md for the input format."""

import dataclasses
import json
import os
import re
from collections.abc import Set
from pathlib import Path
from typing import ClassVar

import yaml
from rdkit import Chem

import kfold.constants as C

RESIDUE_CODES = {
    C.ChainType.PROTEIN: C.residue.PROTEIN_ONE_TO_THREE,
    C.ChainType.RNA: C.residue.RNA_ONE_TO_THREE,
    C.ChainType.DNA: C.residue.DNA_ONE_TO_THREE,
}


def _check_fields(data: dict, required: set[str], optional: set[str]) -> None:
    """Reject malformed mappings, unknown fields, missing fields, and explicit nulls."""
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ValueError("Expected a mapping with string keys.")
    if missing := required - data.keys():
        raise ValueError(f"Missing required fields: {sorted(missing)}.")
    if unknown := data.keys() - required - optional:
        raise ValueError(f"Unknown fields: {sorted(unknown)}.")
    if any(value is None for value in data.values()):
        raise ValueError("Omit optional fields instead of setting them to null.")


def _validate_ids(ids: list[str]) -> None:
    if not isinstance(ids, list) or not ids:
        raise ValueError("'id' must be a non-empty list of chain IDs.")
    for chain_id in ids:
        if not isinstance(chain_id, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*", chain_id
        ):
            raise ValueError(
                f"Invalid chain ID: {chain_id!r}. "
                "Use letters, digits, and underscores; start with a letter."
            )
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate chain IDs: {ids}.")


def _validate_ccd(code: str) -> None:
    if not isinstance(code, str) or not re.fullmatch(r"[A-Z0-9]+", code):
        raise ValueError(f"Invalid CCD code: {code!r}. Use uppercase letters and digits.")


@dataclasses.dataclass(kw_only=True)
class Modification:
    residue_index: int
    ccd: str

    def __post_init__(self) -> None:
        if type(self.residue_index) is not int or self.residue_index < 1:
            raise ValueError(
                "Modification 'residue_index' must be a positive, 1-based integer."
            )
        _validate_ccd(self.ccd)


def _validate_polymer(
    sequence: str, modifications: list[Modification], ctype: C.ChainType
) -> None:
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Polymer sequence must be a non-empty string.")
    if invalid := set(sequence) - RESIDUE_CODES[ctype].keys():
        raise ValueError(f"Invalid residues for {ctype.name}: {sorted(invalid)}.")
    if not isinstance(modifications, list):
        raise ValueError("'modifications' must be a list of Modification objects.")
    residue_indices = set()
    for modification in modifications:
        if not isinstance(modification, Modification):
            raise ValueError("'modifications' must contain Modification objects.")
        if modification.residue_index > len(sequence):
            raise ValueError(
                f"Modification residue index {modification.residue_index} exceeds "
                f"sequence length {len(sequence)}."
            )
        if modification.residue_index in residue_indices:
            raise ValueError(
                f"Duplicate modification residue index: {modification.residue_index}."
            )
        residue_indices.add(modification.residue_index)


def _ccd_sequence(
    sequence: str, modifications: list[Modification], ctype: C.ChainType
) -> list[str]:
    codes = [RESIDUE_CODES[ctype][residue] for residue in sequence]
    for modification in modifications:
        codes[modification.residue_index - 1] = modification.ccd
    return codes


def _validate_structures(apo: list[Path] | None, prior: list[Path] | None) -> None:
    """Python objects hold absolute Paths; parsing resolves serialized strings."""
    if prior is not None and apo is None:
        raise ValueError("'prior' requires provided 'apo' structures.")
    for field, paths in (("apo", apo), ("prior", prior)):
        if paths is None:
            continue
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"'{field}' must be a non-empty list of absolute Paths.")
        for path in paths:
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"'{field}' must contain absolute Paths, got: {path!r}.")
            if not path.is_file():
                raise FileNotFoundError(
                    f"'{field}' structure file does not exist: {path}."
                )
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError(f"Duplicate structure path in '{field}'.")


@dataclasses.dataclass(kw_only=True)
class BaseSequence:
    id: list[str]
    description: str | None = None

    def __post_init__(self) -> None:
        _validate_ids(self.id)
        if self.description is not None and not isinstance(self.description, str):
            raise ValueError("'description' must be a string.")


@dataclasses.dataclass(kw_only=True)
class PolymerSequence(BaseSequence):
    ctype: ClassVar[C.ChainType]
    kind: ClassVar[str]
    sequence: str
    modifications: list[Modification] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_polymer(self.sequence, self.modifications, self.ctype)

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def ccd_sequence(self) -> list[str]:
        return _ccd_sequence(self.sequence, self.modifications, self.ctype)


@dataclasses.dataclass(kw_only=True)
class ProteinSequence(PolymerSequence):
    ctype: ClassVar[C.ChainType] = C.ChainType.PROTEIN
    kind: ClassVar[str] = "protein"
    apo: list[Path] | None = None
    prior: list[Path] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_structures(self.apo, self.prior)


@dataclasses.dataclass(kw_only=True)
class DNASequence(PolymerSequence):
    ctype: ClassVar[C.ChainType] = C.ChainType.DNA
    kind: ClassVar[str] = "dna"


@dataclasses.dataclass(kw_only=True)
class RNASequence(PolymerSequence):
    ctype: ClassVar[C.ChainType] = C.ChainType.RNA
    kind: ClassVar[str] = "rna"


@dataclasses.dataclass(kw_only=True)
class LigandSequence(BaseSequence):
    ctype: ClassVar[C.ChainType] = C.ChainType.LIGAND
    kind: ClassVar[str] = "ligand"
    smiles: str | None = None
    ccd: list[str] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.smiles is None) == (self.ccd is None):
            raise ValueError("Ligand must specify exactly one of 'smiles' or 'ccd'.")
        if self.smiles is not None:
            if not isinstance(self.smiles, str) or not self.smiles.strip():
                raise ValueError("Ligand 'smiles' must be a non-empty string.")
            if Chem.MolFromSmiles(self.smiles) is None:
                raise ValueError("Invalid ligand SMILES.")
        else:
            if not isinstance(self.ccd, list) or not self.ccd:
                raise ValueError("Ligand 'ccd' must be a non-empty list of CCD codes.")
            for code in self.ccd:
                _validate_ccd(code)

    def __len__(self) -> int:
        return 1 if self.smiles is not None else len(self.ccd)


@dataclasses.dataclass(kw_only=True)
class ProteinPair:
    """Two protein components sharing an apo frame; each ID pair is one copy."""

    kind: ClassVar[str] = "protein_pair"
    ctype: ClassVar[C.ChainType] = C.ChainType.PROTEIN
    id: list[list[str]]
    sequence1: str
    sequence2: str
    modifications1: list[Modification] = dataclasses.field(default_factory=list)
    modifications2: list[Modification] = dataclasses.field(default_factory=list)
    apo: list[Path] | None = None
    prior: list[Path] | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, list) or not self.id:
            raise ValueError(
                "Protein pair 'id' must be a non-empty list of two-ID lists."
            )
        for pair in self.id:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Each protein pair copy requires exactly two chain IDs.")
        _validate_ids([chain_id for pair in self.id for chain_id in pair])
        _validate_polymer(self.sequence1, self.modifications1, self.ctype)
        _validate_polymer(self.sequence2, self.modifications2, self.ctype)
        _validate_structures(self.apo, self.prior)
        if self.description is not None and not isinstance(self.description, str):
            raise ValueError("'description' must be a string.")

    def __len__(self) -> int:
        return len(self.sequence1) + len(self.sequence2)

    @property
    def ccd_sequence1(self) -> list[str]:
        return _ccd_sequence(self.sequence1, self.modifications1, self.ctype)

    @property
    def ccd_sequence2(self) -> list[str]:
        return _ccd_sequence(self.sequence2, self.modifications2, self.ctype)


Sequence = ProteinSequence | DNASequence | RNASequence | LigandSequence | ProteinPair


def _parse_sequence(entry: dict, base_dir: Path) -> Sequence:
    # Select the sequence type and validate its allowed fields.
    if not isinstance(entry, dict) or len(entry) != 1:
        raise ValueError("Each sequence entry must contain exactly one sequence type.")
    kind, data = next(iter(entry.items()))
    common = {"description"}
    match kind:
        case "protein" | "dna" | "rna":
            cls = {"protein": ProteinSequence, "dna": DNASequence, "rna": RNASequence}[
                kind
            ]
            optional = common | {"modifications"}
            if kind == "protein":
                optional |= {"apo", "prior"}
            _check_fields(data, {"id", "sequence"}, optional)
        case "protein_pair":
            cls = ProteinPair
            _check_fields(
                data,
                {"id", "sequence1", "sequence2"},
                common | {"modifications1", "modifications2", "apo", "prior"},
            )
        case "ligand":
            cls = LigandSequence
            _check_fields(data, {"id"}, common | {"smiles", "ccd"})
        case _:
            raise ValueError(f"Unsupported sequence type: {kind!r}.")
    # Normalize chain IDs and ligand CCD codes into their internal list forms.
    fields = dict(data)
    if kind == "ligand" and isinstance(fields.get("ccd"), str):
        fields["ccd"] = [fields["ccd"]]
    if kind == "protein_pair":
        if isinstance(fields["id"], list) and all(
            isinstance(chain_id, str) for chain_id in fields["id"]
        ):
            fields["id"] = [fields["id"]]
    elif isinstance(fields["id"], str):
        fields["id"] = [fields["id"]]
    # Parse residue modifications into validated objects.
    for field in ("modifications", "modifications1", "modifications2"):
        if field not in fields:
            continue
        if not isinstance(fields[field], list):
            raise ValueError(f"'{field}' must be a list of residue_index/ccd mappings.")
        modifications = []
        for modification in fields[field]:
            _check_fields(modification, {"residue_index", "ccd"}, set())
            modifications.append(Modification(**modification))
        fields[field] = modifications
    # Resolve provided structure paths relative to the query file.
    for field in ("apo", "prior"):
        if field not in fields:
            continue
        paths = fields[field]
        if (
            not isinstance(paths, list)
            or not paths
            or any(not isinstance(path, str) or not path for path in paths)
        ):
            raise ValueError(f"'{field}' must be a non-empty list of path strings.")
        fields[field] = [(base_dir / path).resolve() for path in paths]
    return cls(**fields)


@dataclasses.dataclass(kw_only=True)
class Bond:
    atom1: tuple[str, int, str]  # chain ID, 1-based residue index, atom name
    atom2: tuple[str, int, str]

    def __post_init__(self) -> None:
        for atom in (self.atom1, self.atom2):
            if not isinstance(atom, tuple) or len(atom) != 3:
                raise ValueError(
                    "Bond atoms must be (chain_id, residue_index, atom_name) tuples."
                )
            chain_id, residue_index, atom_name = atom
            _validate_ids([chain_id])
            if type(residue_index) is not int or residue_index < 1:
                raise ValueError(
                    "Bond residue index must be a positive, 1-based integer."
                )
            if (
                not isinstance(atom_name, str)
                or not atom_name
                or any(char.isspace() for char in atom_name)
            ):
                raise ValueError(
                    "Bond atom name must be a non-empty string without whitespace."
                )
        if self.atom1 == self.atom2:
            raise ValueError("A bond must connect two different atoms.")


@dataclasses.dataclass(kw_only=True)
class Query:
    name: str
    sequences: list[Sequence]
    bonds: list[Bond] = dataclasses.field(default_factory=list)

    def validate_ccd_codes(self, valid_codes: Set[str]) -> None:
        """Check that explicit ligand and modification codes exist in the CCD.

        Parameters
        ----------
        valid_codes : Set[str]
            Component codes available in the chemical component dictionary.

        Raises
        ------
        ValueError
            An explicit component code is absent from the CCD.
        """
        ccd_codes = set()
        for entry in self.sequences:
            if isinstance(entry, PolymerSequence):
                ccd_codes.update(modification.ccd for modification in entry.modifications)
            elif isinstance(entry, LigandSequence) and entry.ccd is not None:
                ccd_codes.update(entry.ccd)
            elif isinstance(entry, ProteinPair):
                ccd_codes.update(
                    modification.ccd for modification in entry.modifications1
                )
                ccd_codes.update(
                    modification.ccd for modification in entry.modifications2
                )
        if not ccd_codes.issubset(valid_codes):
            missing = ccd_codes - valid_codes
            raise ValueError(
                f"Query {self.name}: {', '.join(sorted(missing))} are missing from CCD."
            )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name
        ):
            raise ValueError(
                "Query name must start with a letter or digit and contain only "
                "letters, digits, underscores, dots, and hyphens."
            )
        if not isinstance(self.sequences, list) or not self.sequences:
            raise ValueError("'sequences' must be a non-empty list.")
        # Validate sequence entries and collect unique chain IDs and lengths.
        chain_lengths = {}
        for entry in self.sequences:
            if not isinstance(entry, Sequence):
                raise ValueError(f"Unsupported sequence object: {type(entry).__name__}.")
            if isinstance(entry, ProteinPair):
                chains = [
                    (chain_id, length)
                    for pair in entry.id
                    for chain_id, length in zip(
                        pair, (len(entry.sequence1), len(entry.sequence2)), strict=True
                    )
                ]
            else:
                chains = [(chain_id, len(entry)) for chain_id in entry.id]
            for chain_id, length in chains:
                if chain_id in chain_lengths:
                    raise ValueError(f"Duplicate chain ID: {chain_id!r}.")
                chain_lengths[chain_id] = length
        # Validate bond references against the collected chains and residue ranges.
        if not isinstance(self.bonds, list):
            raise ValueError("'bonds' must be a list of Bond objects.")
        seen_bonds = set()
        for bond in self.bonds:
            if not isinstance(bond, Bond):
                raise ValueError("'bonds' must contain Bond objects.")
            for chain_id, residue_index, _ in (bond.atom1, bond.atom2):
                if chain_id not in chain_lengths:
                    raise ValueError(f"Bond references unknown chain ID: {chain_id!r}.")
                if residue_index > chain_lengths[chain_id]:
                    raise ValueError(
                        f"Bond residue index {residue_index} exceeds chain "
                        f"{chain_id!r} length {chain_lengths[chain_id]}."
                    )
            key = frozenset((bond.atom1, bond.atom2))
            if key in seen_bonds:
                raise ValueError(f"Duplicate bond: {bond}.")
            seen_bonds.add(key)

    @property
    def protein_entries(self) -> list[ProteinSequence | ProteinPair]:
        """Protein entries in query order, matching the prepared apo/prior lists."""
        return [
            entry
            for entry in self.sequences
            if isinstance(entry, (ProteinSequence, ProteinPair))
        ]

    @property
    def priority(self) -> tuple[int, str]:
        """Process smaller complexes first, breaking ties by query name."""
        return sum(len(entry) * len(entry.id) for entry in self.sequences), self.name

    @classmethod
    def load(cls, path: str | Path) -> "Query":
        path = Path(path)
        with path.open() as handle:
            if path.suffix == ".json":
                data = json.load(handle)
            elif path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(handle)
            else:
                raise ValueError(f"Query file must end in .json, .yaml, or .yml: {path}.")
        return cls.from_dict(data, base_dir=path.parent)

    @classmethod
    def from_dict(cls, data: dict, *, base_dir: str | Path) -> "Query":
        """Parse a query dictionary, resolving structure paths relative to base_dir."""
        _check_fields(data, {"name", "sequences"}, {"bonds"})
        # Parse sequence entries, retaining their positions in validation errors.
        if not isinstance(data["sequences"], list):
            raise ValueError("'sequences' must be a list.")
        sequences = []
        for index, entry in enumerate(data["sequences"]):
            try:
                sequences.append(_parse_sequence(entry, Path(base_dir)))
            except (ValueError, FileNotFoundError) as error:
                raise type(error)(f"sequences[{index}]: {error}") from error
        # Parse atom-reference pairs before validating the complete query.
        raw_bonds = data.get("bonds", [])
        if not isinstance(raw_bonds, list):
            raise ValueError("'bonds' must be a list of atom-reference pairs.")
        bonds = []
        for atoms in raw_bonds:
            if (
                not isinstance(atoms, list)
                or len(atoms) != 2
                or any(not isinstance(atom, list) for atom in atoms)
            ):
                raise ValueError(
                    "Each bond must contain two "
                    "[chain_id, residue_index, atom_name] lists."
                )
            bonds.append(Bond(atom1=tuple(atoms[0]), atom2=tuple(atoms[1])))
        return cls(name=data["name"], sequences=sequences, bonds=bonds)

    def to_dict(self, *, relative_to: str | Path | None = None) -> dict:
        """Return a query dictionary with absolute paths unless relative_to is set."""
        # Serialize sequence entries and express structure paths for the destination.
        entries = []
        for entry in self.sequences:
            fields = {
                key: value
                for key, value in dataclasses.asdict(entry).items()
                if value is not None and value != []
            }
            for field in ("apo", "prior"):
                if field in fields:
                    fields[field] = [
                        os.path.relpath(path, relative_to)
                        if relative_to is not None
                        else str(path)
                        for path in fields[field]
                    ]
            entries.append({entry.kind: fields})
        data = {"name": self.name, "sequences": entries}
        if self.bonds:
            data["bonds"] = [[list(bond.atom1), list(bond.atom2)] for bond in self.bonds]
        return data

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = self.to_dict(relative_to=path.parent)
        if path.suffix == ".json":
            text = json.dumps(data, indent=2) + "\n"
        elif path.suffix in {".yaml", ".yml"}:
            text = yaml.safe_dump(data, sort_keys=False)
        else:
            raise ValueError(f"Query file must end in .json, .yaml, or .yml: {path}.")
        path.write_text(text)
