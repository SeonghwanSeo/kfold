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

import dataclasses
import functools
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml
from rdkit import Chem

import kfold.constants as C
from kfold.data.types.ccd import CCD

logger = logging.getLogger("kfold.inference.query")

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
        if not index.isdigit() or int(index) < 1:
            raise ValueError(
                f"modification residue indices must be positive, "
                f"1-based integers, got: {raw_index}"
            )
        if int(index) > len(sequence):
            raise ValueError(
                f"modification residue index {index} exceeds "
                f"sequence length {len(sequence)}."
            )
        if not isinstance(ccd_code, str) or not ccd_code:
            raise ValueError(
                f"modification CCD codes must be non-empty strings, got: {ccd_code}"
            )
        res_idx = int(index)
        if res_idx in normalized:
            raise ValueError(
                "modification residue indices must be unique after normalization, "
                f"got duplicate index: {res_idx}"
            )
        normalized[res_idx] = ccd_code
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

    def __post_init__(self):
        super().__post_init__()

        allowed_tokens = ALLOWED_TOKEN_DICT[self.ctype]
        for restype in self.sequence:
            if restype not in allowed_tokens:
                raise ValueError(
                    f"Invalid residue/base '{restype}' for chain type {self.ctype}. "
                    f"Allowed tokens: {allowed_tokens}"
                )

    @functools.cached_property
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

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.apo is None:
            raise NotImplementedError("Apo sampling is not integrated yet.")
        if self.apo is not None:
            if len(self.apo) == 0:
                raise ValueError("Apo must be a non-empty list of file path strings.")
            for apo_file in self.apo:
                if not Path(apo_file).is_file():
                    raise ValueError(f"Apo file does not exist: {apo_file}")
        if self.prior is not None:
            for prior_file in self.prior:
                if not Path(prior_file).is_file():
                    raise ValueError(f"Prior file does not exist: {prior_file}")


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

    id: list[str]
    smiles: str | None = None
    ccd: str | list[str] | None = None

    def __len__(self) -> int:
        """Return the number of residues in the Ligand sequence."""
        if self.smiles is not None:
            return 1
        else:
            if self.ccd is None:
                raise ValueError(
                    "Ligand sequence must have either 'smiles' or 'ccd' field."
                )
            return 1 if isinstance(self.ccd, str) else len(self.ccd)

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

        physical_ids = tuple(chain_id for id_pair in self.ids for chain_id in id_pair)
        if len(set(physical_ids)) != len(physical_ids):
            raise ValueError(
                f"Protein multimer chain IDs must be unique, got: {physical_ids}"
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

        if self.apo is None:
            raise NotImplementedError("Apo sampling is not integrated yet.")

        if self.apo is not None:
            if len(self.apo) == 0:
                raise ValueError(
                    "Protein multimer 'apo' must be a non-empty list of "
                    "file path strings."
                )
            for apo_file in self.apo:
                if not Path(apo_file).is_file():
                    raise ValueError(f"Apo file does not exist: {apo_file}")
        if self.prior is not None:
            for prior_file in self.prior:
                if not Path(prior_file).is_file():
                    raise ValueError(f"Prior file does not exist: {prior_file}")

    @functools.cached_property
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
    affinity_ligand_id: str | None = None
    seed: int = 0  # Default seed, overridden to command line argument
    yaml: str  # Original YAML content

    @property
    def priority(self) -> tuple[int, int, str]:
        """Compute a priority score for the query (prediction order).
        - Smaller complexes (less residues) have higher priority.
        - For queries of the same size, smaller seed values have higher priority.
        - For queries of the same size and seed, sort by name alphabetically.
        """
        return (self.estimate_size(), self.seed, self.name)

    def estimate_size(self) -> int:
        """Estimate the size of the complex based on the input sequences."""
        sequence_size = sum(len(seq) * len(seq.ids) for seq in self.sequences)
        multimer_size = sum(
            len(sequence_group) * len(sequence_group.ids)
            for sequence_group in self.multimer_sequences
        )
        return sequence_size + multimer_size

    def copy(self, **kwargs) -> "Query":
        """Create a copy of the Query with updated fields."""
        return dataclasses.replace(self, **kwargs)

    def save(self, path: str | Path) -> None:
        """Save the Query as a YAML file to the specified path."""
        with open(path, "w") as f:
            f.write(self.yaml)


def resolve_structure_path(structure_path: str | Path, input_dir: str | Path) -> str:
    """Resolve a custom structure path based on the search priority.

    Paths are resolved in the following priority:
    1. Absolute Path: If an absolute path is provided, it is used directly.
    2. Relative to CWD: If the path exists relative to the current working directory,
        it is used.
    3. Relative to Input File: If neither works, the path is resolved relative to the
        input file directory.

    Parameters
    ----------
    structure_path : str
        The original apo or prior file path from the input.
    input_dir : str | Path
        The directory of the input file, used as the fallback base for relative paths.

    Returns
    -------
    resolved_path: str
        The resolved absolute path to the custom structure file.
    """
    structure_path = Path(structure_path)
    if not structure_path.exists():
        structure_path = Path(input_dir) / structure_path
    return str(structure_path.resolve())


def parse_single_file(json_or_yaml_path: str | Path, ccd: CCD) -> Query:
    """Parse input YAML file into Query dataclass.

    Parameters
    ----------
    json_or_yaml_path : str | Path
        Path to the input JSON/YAML file.
    ccd : CCD
        Chemical component dictionary for validating ligand CCD codes.

    Returns
    -------
    query: Query
        Parsed input data as a Query dataclass.
    """
    if Path(json_or_yaml_path).suffix == ".json":
        with open(json_or_yaml_path) as f:
            input_dict = json.load(f)
    elif Path(json_or_yaml_path).suffix in {".yaml", ".yml"}:
        with open(json_or_yaml_path) as f:
            input_dict = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Input file must be a JSON or YAML file, got: {json_or_yaml_path}"
        )

    # Set default name if not provided
    if input_dict.get("name", None) is None:
        input_dict["name"] = Path(json_or_yaml_path).stem
    name = input_dict["name"]

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

    # Parse sequences
    sequences: list[ProteinSequence | DNASequence | RNASequence | LigandSequence] = []
    for seq_dict in input_sequences:
        chain_type, chain_info = next(iter(seq_dict.items()))

        sequence_info = dict(chain_info)
        sequence_info["id"] = _normalize_id_field(sequence_info["id"])
        if chain_type in {"protein", "dna", "rna"}:
            sequence_info["modifications"] = _parse_modifications(
                sequence_info["sequence"], sequence_info.get("modifications", {})
            )

        if chain_type == "protein":
            # Resolve apo path and prior paths
            input_dir = Path(json_or_yaml_path).parent
            if sequence_info.get("apo", None) is not None:
                apo = sequence_info["apo"]
                apo = [apo] if isinstance(apo, str) else apo
                sequence_info["apo"] = [
                    resolve_structure_path(apo_path, input_dir) for apo_path in apo
                ]

            if sequence_info.get("prior", None) is not None:
                sequence_info["prior"] = [
                    resolve_structure_path(prior_path, input_dir)
                    for prior_path in sequence_info["prior"]
                ]
            elif sequence_info.get("apo", None) is not None:
                # If apo is provided but prior is not, set prior to be the same as apo
                sequence_info["prior"] = sequence_info["apo"].copy()

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
        multimer_info["modifications1"] = _parse_modifications(
            seq1, multimer_info.get("modifications1", {})
        )
        multimer_info["modifications2"] = _parse_modifications(
            seq2, multimer_info.get("modifications2", {})
        )

        # Resolve apo path and prior paths
        input_dir = Path(json_or_yaml_path).parent
        if multimer_info.get("apo", None) is not None:
            apo = multimer_info["apo"]
            apo = [apo] if isinstance(apo, str) else apo
            multimer_info["apo"] = [
                resolve_structure_path(apo_path, input_dir) for apo_path in apo
            ]

        if multimer_info.get("prior", None) is not None:
            multimer_info["prior"] = [
                resolve_structure_path(prior_path, input_dir)
                for prior_path in multimer_info["prior"]
            ]
        elif multimer_info.get("apo", None) is not None:
            # If apo is provided but prior is not, set prior to be the same as apo
            multimer_info["prior"] = multimer_info["apo"].copy()

        mseq = ProteinMultimerSequence(**multimer_info)
        multimer_sequences.append(mseq)

    # Validate the parsed sequences and multimer sequences
    validate_input_sequences(sequences, multimer_sequences, ccd=ccd)

    if "constraints" in input_dict:
        raise ValueError(
            "The top-level 'constraints' field has been removed. "
            "Specify covalent connections with 'bonds' instead."
        )
    bonds = _parse_bonds(input_dict.get("bonds", []))
    _validate_bond_references(bonds, sequences, multimer_sequences)
    affinity_ligand_id = input_dict.get("affinity_ligand_id")
    if affinity_ligand_id is not None and (
        not isinstance(affinity_ligand_id, str) or not affinity_ligand_id
    ):
        raise ValueError("'affinity_ligand_id' must be a non-empty chain ID string.")

    return Query(
        name=name,
        sequences=sequences,
        multimer_sequences=multimer_sequences,
        bonds=bonds,
        affinity_ligand_id=affinity_ligand_id,
        yaml=yaml.safe_dump(input_dict),
    )


def parse_input_files(
    input_path: str | Path,
    ccd: CCD,
    seeds: int | list[int],
    skip_invalid: bool = True,
) -> list[Query]:
    """Parse input JSON/YAML file or all files in a directory into a list of Query.

    Parameters
    ----------
    input_path : str | Path
        Path to the input JSON/YAML file or directory containing such files.
    ccd : CCD
        Chemical component dictionary for validating ligand CCD codes.
    seeds : int | list[int]
        Random seed(s) for the queries.
    skip_invalid : bool
        Whether to skip invalid input files instead of raising an error.
        Only applicable when input_path is a directory.

    Returns
    -------
    queries: list[Query]
        List of parsed input data as Query dataclasses.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        queries = parse_directory(input_path, ccd=ccd, skip_invalid=skip_invalid)
    else:
        queries = [parse_single_file(input_path, ccd=ccd)]

    # Check all queries have the different names to avoid overwriting results
    names = set()
    for query in queries:
        if query.name in names:
            raise ValueError(f"Duplicate query name found: {query.name}")
        names.add(query.name)

    # Copy queries with updated seeds
    seeds = [seeds] if isinstance(seeds, int) else seeds
    all_queries: list[Query] = []
    for query in queries:
        for seed in seeds:
            all_queries.append(query.copy(seed=seed))
    all_queries.sort(key=lambda q: q.priority)
    return all_queries


def parse_directory(
    input_dir: str | Path,
    ccd: CCD,
    skip_invalid: bool = True,
) -> list[Query]:
    """Parse all JSON/YAML files in a directory into a list of Query.

    Parameters
    ----------
    input_dir : str | Path
        Path to the input directory containing JSON/YAML files.
    ccd : CCD
        Chemical component dictionary for validating ligand CCD codes.
    skip_invalid : bool
        Whether to skip invalid input files instead of raising an error.

    Returns
    -------
    queries: list[Query]
        List of parsed input data as Query dataclasses.
    """
    queries: list[Query] = []
    for file_path in Path(input_dir).iterdir():
        if file_path.suffix in {".json", ".yaml", ".yml"}:
            try:
                query = parse_single_file(file_path, ccd)
            except Exception as e:
                if skip_invalid:
                    logger.error(f"Skipping invalid input file {file_path}: {e}")
                else:
                    raise
            else:
                queries.append(query)
    return queries


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
    asym_ids: set[str] = set()
    for entry in input_seqs:
        # Check only one chain type per entry
        if len(entry) != 1:
            raise ValueError("Each sequence entry must contain exactly one chain type.")

        # Check chain type
        chain_type = next(iter(entry.keys()))
        if chain_type not in allow_types:
            raise ValueError(f"Unsupported chain type: {chain_type}")

        # check required fields
        chain_info = entry[chain_type]

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

        if chain_type == "protein":
            apo = chain_info.get("apo")
            valid_apo = isinstance(apo, str) and bool(apo)
            valid_apo = valid_apo or (
                isinstance(apo, list)
                and bool(apo)
                and all(isinstance(path, str) and path for path in apo)
            )
            if apo is not None and not valid_apo:
                raise ValueError(
                    "'protein' apo must be a file path string or a non-empty list "
                    "of file path strings."
                )

            prior = chain_info.get("prior")
            if prior is not None and (
                not isinstance(prior, list)
                or not prior
                or not all(isinstance(path, str) and path for path in prior)
            ):
                raise ValueError(
                    f"'{chain_type}' prior must be a list of file path strings."
                )

        # Check the id(s) are unique
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
            if asym_id in asym_ids:
                raise ValueError(f"Duplicate asym_id found: {asym_id}")
            asym_ids.add(asym_id)


def validate_multimer_input_dicts(
    input_groups: list[dict[str, Any]],
) -> None:
    """Validate raw protein multimer-sequence dictionaries."""
    if not isinstance(input_groups, list):
        raise ValueError("'multimer_sequences' must be a list.")

    physical_ids: set[str] = set()
    for entry in input_groups:
        if len(entry) != 1:
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
            for asym_id in id_pair:
                if asym_id in physical_ids:
                    raise ValueError(f"Duplicate asym_id found: {asym_id}")
                physical_ids.add(asym_id)

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

        apo = chain_info.get("apo")
        if apo is not None and not (
            (isinstance(apo, str) and bool(apo))
            or (
                isinstance(apo, list)
                and bool(apo)
                and all(isinstance(path, str) and path for path in apo)
            )
        ):
            raise ValueError(
                "Protein multimer 'apo' must be a file path string or a non-empty "
                "list of file path strings."
            )

        prior = chain_info.get("prior")
        if prior is not None and (
            not isinstance(prior, list)
            or not prior
            or not all(isinstance(path, str) and path for path in prior)
        ):
            raise ValueError(
                "Protein multimer 'prior' must be a list of file path strings."
            )


def validate_input_sequences(
    seq_list: list[ProteinSequence | DNASequence | RNASequence | LigandSequence],
    multimer_sequences: list[ProteinMultimerSequence],
    ccd: CCD,
) -> None:
    """Validate the input sequence dataclasses.

    Parameters
    ----------
    seq_list : list[BaseSequence]
        List of sequence dataclasses to validate.
    multimer_sequences : list[ProteinMultimerSequence]
        List of protein multimer-sequence dataclasses to validate.
    ccd : CCD
        Chemical component dictionary for validating ligand CCD codes.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    valid_ccd_ids = set(ccd.keys())

    asym_ids: set[str] = set()
    for sequence in seq_list:
        # Check the id(s) are unique
        for asym_id in sequence.ids:
            if asym_id in asym_ids:
                raise ValueError(f"Duplicate asym_id found: {asym_id}")
            asym_ids.add(asym_id)

        # Additional checks for PolymerSequence
        if isinstance(sequence, PolymerSequence):
            for code in sequence.modifications.values():
                if code not in valid_ccd_ids:
                    raise ValueError(f"CCD code '{code}' not found in CCD.")

        # Additional checks for LigandSequence
        if isinstance(sequence, LigandSequence):
            if sequence.smiles is not None:
                # Basic check for SMILES string
                mol = Chem.MolFromSmiles(sequence.smiles)
                if mol is None:
                    raise ValueError("Invalid SMILES string for ligand.")
            else:
                assert sequence.ccd_ids is not None
                for code in sequence.ccd_ids:
                    if code not in valid_ccd_ids:
                        raise ValueError(f"CCD code '{code}' not found in CCD.")

    for sequence_group in multimer_sequences:
        for code in sequence_group.modifications1.values():
            if code not in valid_ccd_ids:
                raise ValueError(f"CCD code '{code}' not found in CCD.")
        for code in sequence_group.modifications2.values():
            if code not in valid_ccd_ids:
                raise ValueError(f"CCD code '{code}' not found in CCD.")
        for id_pair in sequence_group.ids:
            for asym_id in id_pair:
                if asym_id in asym_ids:
                    raise ValueError(f"Duplicate asym_id found: {asym_id}")
                asym_ids.add(asym_id)
