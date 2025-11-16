from pathlib import Path

import gemmi


def load_apo_chain(
    pdb_path: str | Path,
) -> dict[int, tuple[str, dict[str, tuple[float, float, float]]]]:
    """Load a apo polymer chain from a PDB file using gemmi.

    Parameters
    ----------
    pdb_path : Path
        Path to the predicted PDB file (e.g., AFDB, ESMFold, ...)

    Returns
    -------
    dict[int, tuple[str, dict[str, tuple[float, float, float]]]]
        key: residue index (1-based)
        value:
            - residue name (str)
            - dict of atom name to coordinates (x, y, z) as np.ndarray

    """

    structure: gemmi.Structure = gemmi.read_pdb(str(pdb_path))
    raw_chain = structure[0].subchains()[0]

    results: dict[int, tuple[str, dict[str, tuple[float, float, float]]]] = {}
    for res in raw_chain:
        res: gemmi.Residue
        res_name = res.name
        res_idx = int(res.seqid.num)
        name_to_atom_coords = {
            atom.name: (atom.pos.x, atom.pos.y, atom.pos.z) for atom in res
        }
        results[res_idx] = (res_name, name_to_atom_coords)

    return results
