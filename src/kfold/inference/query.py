"""Parsing input JSON/YAML files for co-folding tasks.

The input file should be a json or yaml file with the following format:

```yaml
name: <optional_name_of_the_job>
sequences:
  - protein:
      id: ["A", "B"]
      sequence: "GKMS..."
      description: "Example protein chain" (optional)
      modifications:
        "1": "6OG"
        "4": "SEP"
      apo: "/path/to/apo.pdb",
  - dna:
      id: "C"
      sequence: "ACGT..."
  - ligand:
      id: "D"
      ccd: ["GLY", "TYR"] # multi-residue ligand
constraints:
  - bond:
      atom1: ["A", 10, "NZ"]
      atom2: ["D", 1, "C1"]
  - distance:
      atom1: ["A", 30, "CA"]
      atom2: ["C", 5, "C1'"]
      range: [3.0, 10.0] # -1 means no upper bound
```

```json
{
  "name": "<optional_name_of_the_job>",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MKTS...",
        "apo": "/path/to/apo.pdb",
        "apo_range": "6:10->1:5" # 6-10 residues in sequence maps to 1-5 in input apo
      }
    },
    {
      "ligand": {
        "id": "B",
        "ccd": "ATP"
      }
    },
    {
      "ligand": {
        "id": "C",
        "smiles": "c1ccccc1"
      }
    }
  ]
}
```
"""

import dataclasses
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml
from rdkit import Chem

import kfold.constants as C
from kfold.data.types.ccd import CCD

# === Dataclasses for input formats === #

logger = logging.getLogger("kfold.inference.query")


# === Sequence === #
@dataclasses.dataclass(kw_only=True)
class BaseSequence(ABC):
    """Dataclass for base sequence input format."""

    # class variable
    ctype: ClassVar[C.ChainType]

    id: str | list[str]
    description: str | None = None

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of residues in the protein sequence."""

    @property
    def ids(self) -> list[str]:
        """Return the list of asym_ids."""
        if isinstance(self.id, str):
            return [self.id]
        return self.id

    @property
    def num_chains(self) -> int:
        """Return the number of chains for this sequence."""
        if isinstance(self.id, str):
            return 1
        return len(self.id)


@dataclasses.dataclass(kw_only=True)
class PolymerSequence(BaseSequence):
    """Dataclass for polymer sequence input format."""

    sequence: str
    modifications: dict[str, str] = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of residues in the protein sequence."""
        return len(self.sequence)


@dataclasses.dataclass(kw_only=True)
class ProteinSequence(PolymerSequence):
    """Dataclass for protein sequence input format."""

    # class variable
    ctype: ClassVar = C.ChainType.PROTEIN
    apo: str
    apo_range: str | None = None  # format: "seq_st:seq_end->apo_st:apo_end"

    def __post_init__(self):
        length = len(self)
        if not Path(self.apo).exists():
            raise ValueError(f"Apo file does not exist: {self.apo}")
        if self.apo_range is not None:
            try:
                seq_part, apo_part = self.apo_range.split("->")
                seq_start, seq_end = map(int, seq_part.split(":"))
                apo_start, apo_end = map(int, apo_part.split(":"))
                if seq_start < 1 or seq_end < seq_start:
                    raise ValueError(
                        f"Invalid sequence range in apo_range: {self.apo_range}"
                    )
                if apo_start < 1 or apo_end < apo_start:
                    raise ValueError(f"Invalid apo range in apo_range: {self.apo_range}")
                if seq_end > length:
                    raise ValueError(
                        f"Sequence end index in apo_range exceeds sequence length "
                        f"({length}): {self.apo_range}"
                    )
                if apo_end - apo_start != seq_end - seq_start:
                    raise ValueError(
                        f"Sequence range and apo range must have the same length: "
                        f"{self.apo_range}"
                    )
            except Exception as e:
                raise ValueError(f"Invalid format for apo_range: {self.apo_range}") from e


@dataclasses.dataclass(kw_only=True)
class DNASequence(PolymerSequence):
    """Dataclass for DNA sequence input format."""

    # class variable
    ctype: ClassVar = C.ChainType.DNA
    apo: str | None = None
    apo_range: str | None = None  # format: "seq_st:seq_end->apo_st:apo_end"

    def __post_init__(self):
        assert self.apo is None, (
            "DNA apo files are not accepted in query inputs. "
            "DNA apo coordinates are generated heuristically."
        )
        assert self.apo_range is None, (
            "DNA apo_range is not accepted in query inputs. "
            "DNA apo coordinates are generated as full-length heuristic helices."
        )


@dataclasses.dataclass(kw_only=True)
class RNASequence(PolymerSequence):
    """Dataclass for RNA sequence input format."""

    # class variable
    ctype: ClassVar = C.ChainType.RNA
    apo: str | None = None
    apo_range: str | None = None  # format: "seq_st:seq_end->apo_st:apo_end"

    def __post_init__(self):
        assert self.apo_range is None, "RNA apo_range is not accepted in query inputs."
        if self.apo is None:
            return

        if not Path(self.apo).exists():
            raise ValueError(f"Apo file does not exist: {self.apo}")


@dataclasses.dataclass(kw_only=True)
class LigandSequence(BaseSequence):
    """Dataclass for small molecule sequence input format."""

    # class variable
    ctype: ClassVar = C.ChainType.LIGAND

    id: str | list[str]
    smiles: str | None = None
    ccd: str | list[str] | None = None

    def __len__(self) -> int:
        """Return the number of residues in the Ligand sequence."""
        if self.smiles is not None:
            return 1
        else:
            assert self.ccd is not None
            return 1 if isinstance(self.ccd, str) else len(self.ccd)

    @property
    def ccd_ids(self) -> list[str] | None:
        """Return the list of CCD IDs."""
        if self.ccd is None:
            return None
        if isinstance(self.ccd, str):
            return [self.ccd]
        return self.ccd


# === Constraint === #
@dataclasses.dataclass(kw_only=True)
class Constraint:
    """Dataclass for bond constraint input format."""

    atom1: tuple[str, int, str]  # (chain_id, res_idx, atom_name)
    atom2: tuple[str, int, str]  # (chain_id, res_idx, atom_name)


@dataclasses.dataclass(kw_only=True)
class BondConstraint(Constraint):
    """Dataclass for bond constraint input format."""

    type: ClassVar = "bond"


@dataclasses.dataclass(kw_only=True)
class DistanceConstraint(Constraint):
    """Dataclass for distance constraint input format."""

    type: ClassVar = "distance"
    range: tuple[float, float]  # (lower_bound, upper_bound)


@dataclasses.dataclass(kw_only=True)
class Query:
    name: str  # default: input file name
    sequences: list[ProteinSequence | DNASequence | RNASequence | LigandSequence] = (
        dataclasses.field(default_factory=list)
    )
    constraints: list[BondConstraint | DistanceConstraint] = dataclasses.field(
        default_factory=list
    )
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
        return sum(len(seq) * seq.num_chains for seq in self.sequences)

    def copy(self, **kwargs) -> "Query":
        """Create a copy of the Query with updated fields."""
        return dataclasses.replace(self, **kwargs)

    def save(self, path: str | Path) -> None:
        """Save the Query as a YAML file to the specified path."""
        with open(path, "w") as f:
            f.write(self.yaml)


def resolve_apo_path(apo_path: str, input_dir: str | Path) -> str:
    """Resolve the apo file path to be absolute based on the search priority.

    Paths are resolved in the following priority:
    1. Absolute Path: If an absolute path is provided, it is used directly.
    2. Relative to CWD: If the path exists relative to the current working directory,
        it is used.
    3. Relative to Input File: If neither works, the path is resolved relative to the
        input file directory.

    Parameters
    ----------
    apo_path : str
        The original apo file path from the input.
    input_dir : str | Path
        The directory of the input file, used as the fallback base for relative paths.

    Returns
    -------
    resolved_path: str
        The resolved absolute path to the apo file.
    """
    apo_path: Path = Path(apo_path)
    if not apo_path.exists():
        apo_path = Path(input_dir) / apo_path
    return str(apo_path.resolve())


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
    if "sequences" not in input_dict:
        raise ValueError("Input file must contain 'sequences' field.")
    validate_input_dicts(input_dict["sequences"])

    sequences: list[BaseSequence] = []
    for seq_dict in input_dict["sequences"]:
        assert len(seq_dict) == 1, (
            "Each sequence entry must contain exactly one chain type."
        )
        chain_type, chain_info = next(iter(seq_dict.items()))
        if chain_type in {"protein", "dna", "rna"} and "apo" in chain_info:
            # Resolve apo path to be absolute
            input_dir = Path(json_or_yaml_path).parent
            chain_info["apo"] = resolve_apo_path(chain_info["apo"], input_dir)
        match chain_type:
            case "protein":
                sequence = ProteinSequence(**chain_info)
            case "dna":
                sequence = DNASequence(**chain_info)
            case "rna":
                sequence = RNASequence(**chain_info)
            case "ligand":
                sequence = LigandSequence(**chain_info)
            case _:
                raise ValueError(f"Unsupported chain type: {chain_type}")
        sequences.append(sequence)

    validate_input_sequences(json_or_yaml_path, sequences, ccd=ccd)

    constraints: list[BondConstraint | DistanceConstraint] = []
    for constraint in input_dict.get("constraints", []):
        if len(constraint) != 1:
            raise ValueError(
                f"Each constraint entry must contain exactly one constraint type:"
                f" {constraint}. (supported types: 'bond', 'distance')"
            )
        if "bonds" in constraint:
            cond = constraint["bonds"]
            chain1, res_idx1, atom_name1 = cond["atom1"]
            chain2, res_idx2, atom_name2 = cond["atom2"]
            constraints.append(
                BondConstraint(
                    atom1=(chain1, res_idx1, atom_name1),
                    atom2=(chain2, res_idx2, atom_name2),
                )
            )
        elif "distance" in constraint:
            cond = constraint["distance"]
            if "range" not in cond:
                # Set default range to (2.0, 8.0) if not provided
                cond["range"] = (2.0, 8.0)
            chain1, res_idx1, atom_name1 = cond["atom1"]
            chain2, res_idx2, atom_name2 = cond["atom2"]
            # Convert atom names to uppercase for consistency (e.g. "ca" -> "CA")
            atom_name1, atom_name2 = atom_name1.upper(), atom_name2.upper()
            lower_bound, upper_bound = cond["range"]
            constraints.append(
                DistanceConstraint(
                    atom1=(chain1, res_idx1, atom_name1),
                    atom2=(chain2, res_idx2, atom_name2),
                    range=(lower_bound, upper_bound),
                )
            )

    return Query(
        name=name,
        sequences=sequences,  # type: ignore
        constraints=constraints,
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

    queries: list[Query]
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

        if "modifications" in chain_info:
            mods = chain_info["modifications"]
            for key in mods.keys():
                key = str(key)
                if not key.isdigit():
                    raise ValueError(
                        f"Modification keys must be 1-based index, got: {key}"
                    )
                res_idx = int(key)
                if res_idx < 1:
                    raise ValueError(
                        f"Residue index for modification must be >= 1, got: {key}"
                    )
                num_residues = len(chain_info["sequence"])
                if res_idx > num_residues:
                    raise ValueError(
                        f"Residue index for modification exceeds sequence length "
                        f"({num_residues}), got: {key}"
                    )

        # Check the id(s) are unique
        ids = entry[chain_type]["id"]
        if isinstance(ids, str):
            ids = [ids]
        for i in ids:
            if i in asym_ids:
                raise ValueError(f"Duplicate asym_id found: {i}")
            asym_ids.add(i)


def validate_input_sequences(
    input_path: str | Path,
    seq_list: list[BaseSequence],
    ccd: CCD,
) -> None:
    """Validate the input sequence dataclasses.

    Parameters
    ----------
    input_path : str | Path
        Path to the input JSON/YAML file.
    seq_list : list[BaseSequence]
        List of sequence dataclasses to validate.
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
            ctype = sequence.ctype
            match ctype:
                case C.ChainType.PROTEIN:
                    allow_tokens = C.residue.PROTEIN_AMINO_ACIDS
                case C.ChainType.DNA:
                    allow_tokens = C.residue.DNA_BASES
                case C.ChainType.RNA:
                    allow_tokens = C.residue.RNA_BASES
                case _:
                    raise ValueError(f"Unsupported chain type: {ctype}")
            allow_tokens = set(allow_tokens)

            for res in sequence.sequence:
                if res not in allow_tokens:
                    raise ValueError(
                        f"Invalid residue '{res}' found in sequence of type {ctype}."
                    )

        # Additional checks for LigandSequence
        if isinstance(sequence, LigandSequence):
            if sequence.smiles is not None:
                # Basic check for SMILES string
                mol = Chem.MolFromSmiles(sequence.smiles)
                if mol is None:
                    raise ValueError(
                        f"Invalid SMILES string for ligand with id(s) {sequence.id}."
                    )
            else:
                assert sequence.ccd_ids is not None
                for code in sequence.ccd_ids:
                    if code not in valid_ccd_ids:
                        raise ValueError(
                            f"CCD code '{code}' not found in CCD "
                            f"for ligand with id {sequence.id}."
                        )
