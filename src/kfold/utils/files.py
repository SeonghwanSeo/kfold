from pathlib import Path

import gemmi

Point3D = tuple[float, float, float]


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
