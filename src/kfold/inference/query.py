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
      apo: "/path/to/apo_structure.pdb",
  - dna:
      id: "C"
      sequence: "ACGT..."
  - ligand:
      id: "D"
      ccd: ["GLY", "TYR"] # multi-residue ligand
```

```json
{
  "name": "<optional_name_of_the_job>",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MKTS...",
        "apo": "/path/to/apo_structure.pdb",
      }
    },
    {
      "ligand": {
        "id": "B",
        "ccd": "ATP" # or "smiles": "c1ccccc1"
      }
    }
  ]
}
```
"""

import dataclasses
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml
from rdkit import Chem

import kfold.constants as C
from kfold.data.types.ccd import CCD

# === Dataclasses for input formats === #


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


@dataclasses.dataclass(kw_only=True)
class DNASequence(PolymerSequence):
    """Dataclass for DNA sequence input format."""

    # class variable
    ctype: ClassVar = C.ChainType.DNA


@dataclasses.dataclass(kw_only=True)
class RNASequence(PolymerSequence):
    """Dataclass for RNA sequence input format."""

    # class variable
    ctype: ClassVar = C.ChainType.RNA


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


@dataclasses.dataclass(kw_only=True)
class Query:
    name: str  # default: input file name
    sequences: list[ProteinSequence | DNASequence | RNASequence | LigandSequence] = (
        dataclasses.field(default_factory=list)
    )
    yaml: str  # Original YAML content


def parse_single_file(
    json_or_yaml_path: str | Path,
    ccd: CCD | None = None,
) -> Query:
    """Parse input YAML file into Query dataclass.

    Parameters
    ----------
    json_or_yaml_path : str | Path
        Path to the input JSON/YAML file.
    ccd : CCD | None
        CCD data for ligand parsing. (Optional)
        If provided, used to validate ligand CCD IDs.

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

    validate_input_sequences(sequences, ccd=ccd)

    return Query(
        name=name,
        sequences=sequences,  # type: ignore[arg-type]
        yaml=yaml.safe_dump(input_dict),
    )


def parse_input_files(
    input_path: str | Path,
    ccd: CCD | None = None,
    skip_invalid: bool = True,
) -> list[Query]:
    """Parse input JSON/YAML file or all files in a directory into a list of Query.

    Parameters
    ----------
    input_path : str | Path
        Path to the input JSON/YAML file or directory containing such files.
    ccd : CCD | None
        CCD data for ligand parsing. (Optional)
        If provided, used to validate ligand CCD IDs.
    skip_invalid : bool
        Whether to skip invalid input files instead of raising an error.

    Returns
    -------
    queries: list[Query]
        List of parsed input data as Query dataclasses.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_dir():
        return parse_directory(input_path, ccd=ccd, skip_invalid=skip_invalid)
    else:
        return [parse_single_file(input_path, ccd=ccd)]


def parse_directory(
    input_dir: str | Path,
    ccd: CCD | None = None,
    skip_invalid: bool = True,
) -> list[Query]:
    """Parse all JSON/YAML files in a directory into a list of Query.

    Parameters
    ----------
    input_dir : str | Path
        Path to the input directory containing JSON/YAML files.
    ccd : CCD | None
        CCD data for ligand parsing. (Optional)
        If provided, used to validate ligand CCD IDs.
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
                query = parse_single_file(file_path, ccd=ccd)
            except Exception as e:
                if skip_invalid:
                    print(f"Skipping invalid input file {file_path}: {e}")
                else:
                    raise e
            else:
                queries.append(query)
    queries.sort(key=lambda q: q.name)
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
    seq_list: list[BaseSequence],
    ccd: CCD | None = None,
) -> None:
    """Validate the input sequence dataclasses.

    Parameters
    ----------
    seq_list : list[BaseSequence]
        List of sequence dataclasses to validate.
    ccd : CCD | None
        CCD data for ligand parsing. (Optional)
        If provided, used to validate ligand CCD IDs.

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
        elif isinstance(sequence, LigandSequence):
            if sequence.smiles is not None:
                # Basic check for SMILES string
                mol = Chem.MolFromSmiles(sequence.smiles)
                if mol is None:
                    raise ValueError(
                        f"Invalid SMILES string for ligand with id(s) {sequence.id}."
                    )
            else:
                ccd_ids = sequence.ccd_ids
                assert ccd_ids is not None
                if ccd is not None:
                    valid_ccd_ids = set(ccd.keys())
                    for ccd_id in ccd_ids:
                        if ccd_id not in valid_ccd_ids:
                            raise ValueError(
                                f"CCD ID '{ccd_id}' not found in CCD data "
                                f"for ligand with id(s) {sequence.id}."
                            )
