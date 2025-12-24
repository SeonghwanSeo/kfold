from pathlib import Path

import gemmi

Point3D = tuple[float, float, float]


def load_fasta(path: str | Path) -> dict[str, str]:
    """Load sequences from a fasta file (supports multi-line sequences).

    Parameters
    ----------
    path : str | Path
        Path to the fasta file.

    Returns
    -------
    sequences : dict[str, str]
        key: sequence ID
        value: sequence string

    """
    sequences: dict[str, str] = {}

    current_seq_id = None
    current_seq_parts: list[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines if any exist
            if not line:
                continue

            if line.startswith(">"):
                if current_seq_id is not None:
                    sequences[current_seq_id] = "".join(current_seq_parts)

                # Start a new sequence entry
                current_seq_id = line[1:]  # Remove '>'
                current_seq_parts = []
            else:
                # Append sequence lines to the current list buffer
                if current_seq_id is not None:
                    current_seq_parts.append(line)

        # Add the last sequence after the loop ends
        if current_seq_id is not None:
            sequences[current_seq_id] = "".join(current_seq_parts)

    return sequences


def load_apo_chain(
    path: str | Path,
) -> tuple[dict[int, str], dict[int, dict[str, Point3D]]]:
    """Load a apo polymer chain from a PDB file using gemmi.

    Parameters
    ----------
    path : Path
        Path to the predicted structure file (e.g., AFDB, ESMFold, ...)

    Returns
    -------
    sequences : dict[int, str]
        key: residue index (1-based)
        value: residue name (three-letter code)
    atom_coordinates: dict[int, dict[str, tuple[float, float, float]]]
        key: residue index (1-based)
        value:
            - dict of atom name to coordinates (x, y, z) as np.ndarray

    """

    filetype = Path(path).suffix.lower()
    assert filetype in {".pdb", ".cif"}, f"Unsupported file type: {filetype}"

    structure: gemmi.Structure
    if filetype == ".cif":
        structure = gemmi.read_structure(str(path))
    else:
        structure = gemmi.read_pdb(str(path))
    raw_chain = structure[0].subchains()[0]

    sequences: dict[int, str] = {}
    atom_coordinates: dict[int, dict[str, tuple[float, float, float]]] = {}
    for res in raw_chain:
        res: gemmi.Residue
        res_name = res.name
        res_idx = int(res.seqid.num)
        atom_name_to_coords = {
            atom.name: (atom.pos.x, atom.pos.y, atom.pos.z) for atom in res
        }
        sequences[res_idx] = res_name
        atom_coordinates[res_idx] = atom_name_to_coords

    return sequences, atom_coordinates
