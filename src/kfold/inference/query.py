"""Parsing input JSON/YAML files for co-folding tasks.

The input file should be a json or yaml file with the following format:

```yaml
name: <optional_name_of_the_job>
sequences:
  - protein:
      id: ["A", "B"]
      sequence: "GKMS..."
      description: "Example protein chain" (optional)
      apo: [
        "protein_apo_1.pdb",
        "protein_apo_2.pdb",
        ...
      ]
      prior: [
        "protein_prior_1.pdb",
        "protein_prior_2.pdb",
        ...
      ] (optional)
      modifications:
        "1": "6OG"
        "4": "SEP"
  - dna:
      id: "C"
      sequence: "ACGT..."
  - ligand:
      id: "D"
      ccd: ["GLY", "TYR"] # multi-residue ligand

multimer_sequences:
  - protein:
      id: ["H:L", "M:N"]
      sequence: "GKMS:ACDEFGHIKLMNPQRSTVWY"
      description: "Antibody Fab"
      apo: ["fab_apo_1.pdb", "fab_apo_2.pdb", ...]
      prior: ["fab_prior_1.pdb", "fab_prior_2.pdb"] (optional)

bonds:
  - [["A", 20, "NZ"], ["D", 1, "C08"]]
```
"""

import copy
import dataclasses
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml
from rdkit import Chem

import kfold.constants as C

ALLOWED_TOKEN_DICT = {
    C.ChainType.PROTEIN: set(C.residue.PROTEIN_AMINO_ACIDS),
    C.ChainType.RNA: set(C.residue.RNA_BASES),
    C.ChainType.DNA: set(C.residue.DNA_BASES),
}
MAPPING_DICT = {
    C.ChainType.PROTEIN: C.residue.PROTEIN_ONE_TO_THREE,
    C.ChainType.RNA: C.residue.RNA_ONE_TO_THREE,
    C.ChainType.DNA: C.residue.DNA_ONE_TO_THREE,
}


# === Helper Functions === #
def _parse_modifications(
    sequence: str,
    raw_modifications: dict[Any, Any],
) -> dict[int, str]:
    """Parse YAML/JSON modification keys into 1-based integer indices."""
    if not isinstance(raw_modifications, dict):
        raise ValueError("modification must be a mapping.")
    normalized: dict[int, str] = {}
    for raw_index, ccd_code in raw_modifications.items():
        index = str(raw_index)
        if not index.isdecimal():
            raise ValueError(
                f"modification residue indices must be positive, "
                f"1-based integers, got: {raw_index}"
            )
        res_idx = int(index)
        if not 1 <= res_idx <= len(sequence):
            raise ValueError(
                f"modification residue index {index} is outside "
                f"sequence length {len(sequence)}."
            )
        if not isinstance(ccd_code, str) or not ccd_code:
            raise ValueError(
                f"modification CCD codes must be non-empty strings, got: {ccd_code}"
            )
        if res_idx in normalized:
            raise ValueError(
                "modification residue indices must be unique after normalization, "
                f"got duplicate index: {res_idx}"
            )
        normalized[res_idx] = ccd_code.upper()
    return normalized


def _normalize_id_field(id_field: str | list[str]) -> list[str]:
    """Normalize the 'id' field to a list of strings."""
    if isinstance(id_field, str):
        return [id_field]
    elif isinstance(id_field, list) and all(isinstance(i, str) for i in id_field):
        return id_field
    else:
        raise ValueError(
            f"'id' field must be a string or a list of strings, got: {id_field!r}"
        )


# === Sequence === #
@dataclasses.dataclass(kw_only=True)
class BaseSequence(ABC):
    """Dataclass for base sequence input format."""

    # class variable
    ctype: ClassVar[C.ChainType]
    seqtype: ClassVar[str]

    id: list[str]
    description: str | None = None

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of residues in the protein sequence."""

    def __post_init__(self) -> None:
        if not isinstance(self.id, list) or not self.id:
            raise ValueError("Sequence 'id' must be a non-empty list of chain IDs.")
        if not all(isinstance(chain_id, str) and chain_id for chain_id in self.id):
            raise ValueError(
                f"Sequence chain IDs must be non-empty strings, got: {self.id!r}"
            )

    @property
    def ids(self) -> list[str]:
        """Return the list of asym_ids."""
        return self.id


@dataclasses.dataclass(kw_only=True)
class BaseSequenceGroup(ABC):
    """Dataclass for base sequence-group input format."""

    # class variable
    ctype: ClassVar[C.ChainType]
    seqtype: ClassVar[str]

    id: list[tuple[str, str]]
    description: str | None = None

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of residues in the protein sequence."""

    def __post_init__(self) -> None:
        if not isinstance(self.id, list) or not self.id:
            raise ValueError("Sequence-group 'id' must be a non-empty list of ID pairs.")
        for id_pair in self.id:
            if not isinstance(id_pair, tuple) or len(id_pair) != 2:
                raise ValueError(
                    "Each sequence-group ID must be a tuple of two chain IDs, "
                    f"got: {id_pair!r}"
                )
            if not all(isinstance(chain_id, str) and chain_id for chain_id in id_pair):
                raise ValueError(
                    "Sequence-group chain IDs must be non-empty strings, "
                    f"got: {id_pair!r}"
                )

    @property
    def ids(self) -> list[tuple[str, str]]:
        """Return the list of asym_id pairs."""
        return self.id


@dataclasses.dataclass(kw_only=True)
class PolymerSequence(BaseSequence):
    """Dataclass for polymer sequence input format."""

    sequence: str
    modifications: dict[int, str] = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of residues in the protein sequence."""
        return len(self.sequence)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.sequence, str) or not self.sequence:
            raise ValueError("Polymer sequence must be a non-empty string.")
        self.modifications = _parse_modifications(self.sequence, self.modifications)

        allowed_tokens = ALLOWED_TOKEN_DICT[self.ctype]
        for restype in self.sequence:
            if restype not in allowed_tokens:
                raise ValueError(
                    f"Invalid residue/base '{restype}' for chain type {self.ctype}. "
                    f"Allowed tokens: {allowed_tokens}"
                )

    @property
    def ccd_sequence(self) -> list[str]:
        """Return the resolved CCD sequence, using modifications where applicable."""
        # First, map the standard residues/bases to their CCD codes
        mapping = MAPPING_DICT[self.ctype]
        ccd_sequence = [mapping[restype] for restype in self.sequence]

        # Then, apply any modifications to the CCD sequence
        for res_idx, ccd_code in self.modifications.items():
            ccd_sequence[res_idx - 1] = ccd_code
        return ccd_sequence


@dataclasses.dataclass(kw_only=True)
class ProteinSequence(PolymerSequence):
    """Dataclass for protein sequence input format."""

    ctype: ClassVar[C.ChainType] = C.ChainType.PROTEIN
    seqtype: ClassVar[str] = "protein"

    # One or more apo structures may be provided for protein sequences.
    apo: list[str] | None = None
    prior: list[str] | None = None


@dataclasses.dataclass(kw_only=True)
class DNASequence(PolymerSequence):
    """Dataclass for DNA sequence input format."""

    ctype: ClassVar[C.ChainType] = C.ChainType.DNA
    seqtype: ClassVar[str] = "dna"


@dataclasses.dataclass(kw_only=True)
class RNASequence(PolymerSequence):
    """Dataclass for RNA sequence input format."""

    ctype: ClassVar[C.ChainType] = C.ChainType.RNA
    seqtype: ClassVar[str] = "rna"


@dataclasses.dataclass(kw_only=True)
class LigandSequence(BaseSequence):
    """Dataclass for small molecule sequence input format."""

    ctype: ClassVar[C.ChainType] = C.ChainType.LIGAND
    seqtype: ClassVar[str] = "ligand"

    smiles: str | None = None
    ccd: str | list[str] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.smiles is None) == (self.ccd is None):
            raise ValueError("Ligand must specify exactly one of 'smiles' or 'ccd'.")
        if self.smiles is not None:
            if not isinstance(self.smiles, str) or not self.smiles.strip():
                raise ValueError("Ligand 'smiles' must be a non-empty string.")
            if Chem.MolFromSmiles(self.smiles) is None:
                raise ValueError("Invalid SMILES string for ligand.")
        else:
            codes = self.ccd_ids
            if (
                not isinstance(codes, list)
                or not codes
                or any(not isinstance(code, str) or not code.strip() for code in codes)
            ):
                raise ValueError("Ligand 'ccd' must contain non-empty CCD code strings.")
            # normalize
            codes = [code.upper() for code in codes]
            self.ccd = codes

    def __len__(self) -> int:
        """Return the number of ligand residues."""
        if self.smiles is not None or isinstance(self.ccd, str):
            return 1
        return len(self.ccd)

    @property
    def ccd_ids(self) -> list[str] | None:
        """Return the list of CCD IDs."""
        if self.ccd is None:
            return None
        if isinstance(self.ccd, str):
            return [self.ccd]
        return self.ccd


@dataclasses.dataclass(kw_only=True)
class ProteinMultimerSequence(BaseSequenceGroup):
    """A protein dimer sequence.

    ``id=[("H", "L")]`` describes one pair of physical chains. Custom
    apo/prior paths are group-level inputs: each file is expected to contain
    both dimer components in sequence order.
    """

    ctype: ClassVar[C.ChainType] = C.ChainType.PROTEIN
    seqtype: ClassVar[str] = "protein_multimer"

    sequence1: str
    sequence2: str
    modifications1: dict[int, str] = dataclasses.field(default_factory=dict)
    modifications2: dict[int, str] = dataclasses.field(default_factory=dict)
    apo: list[str] | None = None
    prior: list[str] | None = None

    def __len__(self) -> int:
        """Return the total number of residues in one dimer copy."""
        return len(self.sequence1) + len(self.sequence2)

    def __post_init__(self) -> None:
        super().__post_init__()
        allowed_tokens = ALLOWED_TOKEN_DICT[C.ChainType.PROTEIN]

        for id_pair in self.ids:
            if any(":" in chain_id for chain_id in id_pair):
                raise ValueError(
                    f"Protein multimer chain IDs must not contain ':', got: {id_pair!r}"
                )

        sequences = (self.sequence1, self.sequence2)
        for index, sequence in enumerate(sequences, start=1):
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(
                    f"Protein multimer sequence{index} must be a non-empty string."
                )
            for aa in sequence:
                if aa not in allowed_tokens:
                    raise ValueError(
                        f"Invalid residue '{aa}' in protein multimer sequence{index}."
                    )

        self.modifications1 = _parse_modifications(self.sequence1, self.modifications1)
        self.modifications2 = _parse_modifications(self.sequence2, self.modifications2)

    @property
    def ccd_sequence1(self) -> list[str]:
        """Return the resolved CCD sequence, using modifications where applicable."""
        mapping = MAPPING_DICT[C.ChainType.PROTEIN]
        ccd_sequence = [mapping[aa] for aa in self.sequence1]
        for res_idx, ccd_code in self.modifications1.items():
            ccd_sequence[res_idx - 1] = ccd_code
        return ccd_sequence

    @property
    def ccd_sequence2(self) -> list[str]:
        """Return the resolved CCD sequence, using modifications where applicable."""
        mapping = MAPPING_DICT[C.ChainType.PROTEIN]
        ccd_sequence = [mapping[aa] for aa in self.sequence2]
        for res_idx, ccd_code in self.modifications2.items():
            ccd_sequence[res_idx - 1] = ccd_code
        return ccd_sequence


# === Bond === #
@dataclasses.dataclass(kw_only=True)
class Bond:
    """Covalent bond between two atoms in the query."""

    atom1: tuple[str, int, str]  # (chain_id, res_idx, atom_name)
    atom2: tuple[str, int, str]  # (chain_id, res_idx, atom_name)


def _parse_bonds(raw_bonds: Any) -> list[Bond]:
    """Parse top-level covalent bonds from the compact input format."""
    if not isinstance(raw_bonds, list):
        raise ValueError("'bonds' must be a list of atom-reference pairs.")

    def parse_atom(atom: Any, bond_index: int, atom_index: int) -> tuple[str, int, str]:
        if not isinstance(atom, list) or len(atom) != 3:
            raise ValueError(
                f"bonds[{bond_index}][{atom_index}] must be "
                "[chain_id, residue_index, atom_name]."
            )
        chain_id, residue_index, atom_name = atom
        if not isinstance(chain_id, str) or not chain_id:
            raise ValueError(
                f"bonds[{bond_index}][{atom_index}] chain ID must be a non-empty string."
            )
        if (
            not isinstance(residue_index, int)
            or isinstance(residue_index, bool)
            or residue_index < 1
        ):
            raise ValueError(
                f"bonds[{bond_index}][{atom_index}] residue index must be a "
                "positive, 1-based integer."
            )
        if not isinstance(atom_name, str) or not atom_name:
            raise ValueError(
                f"bonds[{bond_index}][{atom_index}] atom name must be a non-empty string."
            )
        return chain_id, residue_index, atom_name

    bonds: list[Bond] = []
    for bond_index, raw_bond in enumerate(raw_bonds):
        if not isinstance(raw_bond, list) or len(raw_bond) != 2:
            raise ValueError(
                f"bonds[{bond_index}] must contain exactly two atom references."
            )
        bonds.append(
            Bond(
                atom1=parse_atom(raw_bond[0], bond_index, 0),
                atom2=parse_atom(raw_bond[1], bond_index, 1),
            )
        )
    return bonds


def _validate_bond_references(
    bonds: list[Bond],
    sequences: list[ProteinSequence | DNASequence | RNASequence | LigandSequence],
    multimer_sequences: list[ProteinMultimerSequence],
) -> None:
    """Validate bond chain IDs and 1-based residue indices."""
    chain_lengths = {
        chain_id: len(sequence) for sequence in sequences for chain_id in sequence.ids
    }
    for sequence_group in multimer_sequences:
        for chain_id1, chain_id2 in sequence_group.ids:
            chain_lengths[chain_id1] = len(sequence_group.sequence1)
            chain_lengths[chain_id2] = len(sequence_group.sequence2)

    for bond_index, bond in enumerate(bonds):
        for atom_index, atom in enumerate((bond.atom1, bond.atom2)):
            chain_id, residue_index, _ = atom
            if chain_id not in chain_lengths:
                raise ValueError(
                    f"bonds[{bond_index}][{atom_index}] references unknown "
                    f"chain ID '{chain_id}'."
                )
            if residue_index > chain_lengths[chain_id]:
                raise ValueError(
                    f"bonds[{bond_index}][{atom_index}] residue index "
                    f"{residue_index} exceeds chain '{chain_id}' length "
                    f"{chain_lengths[chain_id]}."
                )


@dataclasses.dataclass(kw_only=True)
class Query:
    name: str  # default: input file name
    sequences: list[ProteinSequence | DNASequence | RNASequence | LigandSequence] = (
        dataclasses.field(default_factory=list)
    )
    multimer_sequences: list[ProteinMultimerSequence] = dataclasses.field(
        default_factory=list
    )
    bonds: list[Bond] = dataclasses.field(default_factory=list)

    @property
    def priority(self) -> tuple[int, str]:
        """Compute a priority score for the query (prediction order).
        - Smaller complexes (less residues) have higher priority.
        - For queries of the same size, sort by name alphabetically.
        """
        return (self.estimate_size(), self.name)

    def estimate_size(self) -> int:
        """Estimate the size of the complex based on the input sequences."""
        sequence_size = sum(len(seq) * len(seq.ids) for seq in self.sequences)
        multimer_size = sum(
            len(sequence_group) * len(sequence_group.ids)
            for sequence_group in self.multimer_sequences
        )
        return sequence_size + multimer_size

    @classmethod
    def load(cls, path: str | Path) -> "Query":
        """Load a JSON/YAML query, resolving structures from CWD, then its directory."""
        path = Path(path)
        if path.suffix == ".json":
            with open(path) as f:
                input_dict = json.load(f)
        elif path.suffix in {".yaml", ".yml"}:
            with open(path) as f:
                input_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Input file must be a JSON or YAML file, got: {path}")

        if not isinstance(input_dict, dict):
            raise ValueError("Query document must be a mapping.")

        # Set default name if not provided
        input_dict.setdefault("name", path.stem)
        name = input_dict["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Query name must be a non-empty string.")

        # Parse sequences
        input_sequences = input_dict.get("sequences", [])
        input_multimer_sequences = input_dict.get("multimer_sequences", [])
        if not input_sequences and not input_multimer_sequences:
            raise ValueError(
                "Input file must contain at least one 'sequences' or "
                "'multimer_sequences' entry."
            )

        # Validate input sequence dictionaries
        validate_input_dicts(input_sequences)
        validate_multimer_input_dicts(input_multimer_sequences)

        input_dir = path.parent
        for entry in [*input_sequences, *input_multimer_sequences]:
            if "protein" not in entry:
                continue
            protein = entry["protein"]
            for field in ("apo", "prior"):
                paths = protein.get(field)
                if paths is None:
                    continue
                if isinstance(paths, str):
                    paths = [paths]
                if (
                    not isinstance(paths, list)
                    or not paths
                    or any(not isinstance(source, str) or not source for source in paths)
                ):
                    raise ValueError(
                        f"'{field}' must be a non-empty list of structure paths."
                    )
                resolved_paths = []
                for source in paths:
                    source_path = Path(source)
                    if not source_path.exists():
                        source_path = input_dir / source_path
                    if not source_path.is_file():
                        raise FileNotFoundError(
                            f"'{field}' structure file '{source}' was not found as "
                            f"supplied or relative to query directory '{input_dir}'."
                        )
                    resolved_paths.append(str(source_path.resolve()))
                if len(set(resolved_paths)) != len(resolved_paths):
                    raise ValueError(f"Duplicate structure path in '{field}'.")
                protein[field] = resolved_paths

        # Parse sequences
        sequences: list[ProteinSequence | DNASequence | RNASequence | LigandSequence] = []
        for seq_dict in input_sequences:
            chain_type, chain_info = next(iter(seq_dict.items()))

            sequence_info = dict(chain_info)
            sequence_info["id"] = _normalize_id_field(sequence_info["id"])
            match chain_type:
                case "protein":
                    seq = ProteinSequence(**sequence_info)
                case "dna":
                    seq = DNASequence(**sequence_info)
                case "rna":
                    seq = RNASequence(**sequence_info)
                case "ligand":
                    seq = LigandSequence(**sequence_info)
                case _:
                    raise ValueError(f"Unsupported chain type: {chain_type}")
            sequences.append(seq)

        # Parse multimer sequences
        multimer_sequences: list[ProteinMultimerSequence] = []
        for multimer_seq_dict in input_multimer_sequences:
            chain_info = multimer_seq_dict["protein"]
            multimer_info = dict(chain_info)

            # Convert the "id" field to a list of id pairs
            group_ids = _normalize_id_field(chain_info["id"])
            multimer_info["id"] = [tuple(group_id.split(":")) for group_id in group_ids]

            # Convert the "sequence" field to two separate sequences
            seq1, seq2 = multimer_info.pop("sequence").split(":")
            multimer_info["sequence1"] = seq1
            multimer_info["sequence2"] = seq2
            mseq = ProteinMultimerSequence(**multimer_info)
            multimer_sequences.append(mseq)

        # Validate the parsed sequences and multimer sequences
        validate_input_sequences(sequences, multimer_sequences)

        if "constraints" in input_dict:
            raise ValueError(
                "The top-level 'constraints' field has been removed. "
                "Specify covalent connections with 'bonds' instead."
            )
        bonds = _parse_bonds(input_dict.get("bonds", []))
        _validate_bond_references(bonds, sequences, multimer_sequences)

        return cls(
            name=name,
            sequences=sequences,
            multimer_sequences=multimer_sequences,
            bonds=bonds,
        )

    def to_dict(self, *, relative_to: str | Path | None = None) -> dict:
        """Build the input schema from current fields, without cached source text.

        Structure paths are absolute unless ``relative_to`` specifies the directory
        containing the serialized query. Returned containers are independent copies.
        """
        data = {"name": self.name}
        for section, sequences in (
            ("sequences", self.sequences),
            ("multimer_sequences", self.multimer_sequences),
        ):
            if not sequences:
                continue
            entries = []
            for sequence in sequences:
                entity = {
                    field.name: copy.deepcopy(getattr(sequence, field.name))
                    for field in dataclasses.fields(sequence)
                    if field.name not in {"apo", "prior"}
                    and not field.name.startswith("_")
                    and getattr(sequence, field.name) is not None
                }
                if isinstance(sequence, ProteinMultimerSequence):
                    entity["id"] = [":".join(pair) for pair in sequence.id]
                    entity["sequence"] = (
                        entity.pop("sequence1") + ":" + entity.pop("sequence2")
                    )
                for field in ("modifications", "modifications1", "modifications2"):
                    if field in entity:
                        entity[field] = {
                            str(index): code for index, code in entity[field].items()
                        }
                for field in ("apo", "prior"):
                    refs = getattr(sequence, field, None)
                    if refs is None:
                        continue
                    output = []
                    for ref in refs:
                        source_path = str(Path(ref).resolve())
                        if relative_to is not None:
                            source_path = os.path.relpath(
                                source_path, Path(relative_to).resolve()
                            )
                        output.append(source_path)
                    entity[field] = output
                kind = (
                    "protein"
                    if isinstance(sequence, ProteinMultimerSequence)
                    else sequence.seqtype
                )
                entries.append({kind: entity})
            data[section] = entries
        if self.bonds:
            data["bonds"] = [[list(bond.atom1), list(bond.atom2)] for bond in self.bonds]
        return data

    def save(self, path: str | Path) -> None:
        """Serialize current query fields with paths relative to the output file."""
        path = Path(path)
        if path.suffix not in {".json", ".yaml", ".yml"}:
            raise ValueError(f"Output file must be a JSON or YAML file, got: {path}")
        data = self.to_dict(relative_to=path.parent)
        text = (
            json.dumps(data, indent=2) + "\n"
            if path.suffix == ".json"
            else yaml.safe_dump(data)
        )
        path.write_text(text)


def validate_input_dicts(
    input_seqs: list[dict[str, Any]],
) -> None:
    """Validate the input sequence dictionary.

    Parameters
    ----------
    input_seqs : dict[str, Any]
        List of sequence entries to validate.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    if not isinstance(input_seqs, list):
        raise ValueError("'sequences' must be a list.")

    allow_types = {"protein", "dna", "rna", "ligand"}
    for entry in input_seqs:
        # Check only one chain type per entry
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("Each sequence entry must contain exactly one chain type.")

        # Check chain type
        chain_type = next(iter(entry.keys()))
        if chain_type not in allow_types:
            raise ValueError(f"Unsupported chain type: {chain_type}")

        # check required fields
        chain_info = entry[chain_type]
        if not isinstance(chain_info, dict):
            raise ValueError(f"'{chain_type}' entry must be a mapping.")

        if "id" not in chain_info:
            raise ValueError(f"Missing 'id' field for chain type: {chain_type}")
        if chain_type in {"protein", "dna", "rna"}:
            if "sequence" not in chain_info:
                raise ValueError(f"Missing 'sequence' field for chain type: {chain_type}")
        if chain_type == "ligand":
            if "smiles" not in chain_info and "ccd" not in chain_info:
                raise ValueError("Ligand chain must have either 'smiles' or 'ccd' field.")
            elif "smiles" in chain_info and "ccd" in chain_info:
                raise ValueError(
                    "Ligand chain cannot have both 'smiles' and 'ccd' fields."
                )

        if chain_type in {"dna", "rna"} and (
            "apo" in chain_info or "prior" in chain_info
        ):
            raise ValueError(
                f"'{chain_type}' entries do not support 'apo' or 'prior' fields."
            )

        # Validate the serialized chain IDs
        ids = entry[chain_type]["id"]
        if isinstance(ids, str):
            ids = [ids]
        elif not isinstance(ids, list) or not ids:
            raise ValueError(
                f"'id' for chain type '{chain_type}' must be a string or "
                "a non-empty list of strings."
            )
        for asym_id in ids:
            if not isinstance(asym_id, str) or not asym_id:
                raise ValueError(f"Chain IDs must be non-empty strings, got: {asym_id!r}")


def validate_multimer_input_dicts(
    input_groups: list[dict[str, Any]],
) -> None:
    """Validate raw protein multimer-sequence dictionaries."""
    if not isinstance(input_groups, list):
        raise ValueError("'multimer_sequences' must be a list.")

    for entry in input_groups:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                "Each multimer sequence entry must contain exactly one chain type."
            )

        chain_type = next(iter(entry))
        if chain_type != "protein":
            raise ValueError(
                f"Unsupported multimer chain type: {chain_type}. "
                "Only 'protein' is supported."
            )

        chain_info = entry[chain_type]
        if not isinstance(chain_info, dict):
            raise ValueError(f"'{chain_type}' entry must be a mapping.")
        if "id" not in chain_info:
            raise ValueError("Missing 'id' field for protein multimer sequence.")
        if "sequence" not in chain_info:
            raise ValueError("Missing 'sequence' field for protein multimer sequence.")

        group_ids = chain_info["id"]
        if isinstance(group_ids, str):
            group_ids = [group_ids]
        elif not isinstance(group_ids, list) or not group_ids:
            raise ValueError(
                "'id' for protein multimer sequence must be a string or "
                "a non-empty list of strings."
            )
        for group_id in group_ids:
            if not isinstance(group_id, str):
                raise ValueError(
                    "Protein multimer IDs must be strings in 'chain1:chain2' "
                    f"format, got: {group_id!r}"
                )
            id_pair = group_id.split(":")
            if len(id_pair) != 2 or not all(id_pair):
                raise ValueError(
                    "Protein multimer IDs must have exactly two non-empty chain IDs "
                    f"in 'chain1:chain2' format, got: {group_id!r}"
                )

        sequence = chain_info["sequence"]
        if not isinstance(sequence, str):
            raise ValueError(
                "Protein multimer 'sequence' must be a string in "
                "'sequence1:sequence2' format."
            )
        sequence_pair = sequence.split(":")
        if len(sequence_pair) != 2 or not all(sequence_pair):
            raise ValueError(
                "Protein multimer 'sequence' must contain exactly two non-empty "
                "sequences in 'sequence1:sequence2' format."
            )


def validate_input_sequences(
    seq_list: list[ProteinSequence | DNASequence | RNASequence | LigandSequence],
    multimer_sequences: list[ProteinMultimerSequence],
) -> None:
    """Validate the input sequence dataclasses.

    Parameters
    ----------
    seq_list : list[BaseSequence]
        List of sequence dataclasses to validate.
    multimer_sequences : list[ProteinMultimerSequence]
        List of protein multimer-sequence dataclasses to validate.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    asym_ids: set[str] = set()
    for sequence in seq_list:
        # Check the id(s) are unique
        for asym_id in sequence.ids:
            if asym_id in asym_ids:
                raise ValueError(f"Duplicate asym_id found: {asym_id}")
            asym_ids.add(asym_id)

    for sequence_group in multimer_sequences:
        for id_pair in sequence_group.ids:
            for asym_id in id_pair:
                if asym_id in asym_ids:
                    raise ValueError(f"Duplicate asym_id found: {asym_id}")
                asym_ids.add(asym_id)
