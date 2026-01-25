from pathlib import Path

import gemmi
import numpy as np

import kfold.constants as C

# Constants
atom37_order: dict[str, int] = C.atom.protein_atom37_order
protein_one_letter_to_residue_name: dict[str, C.ResidueName] = (
    C.residue.protein_one_letter_to_residue_name
)


def read_protein_structure(path: str | Path) -> tuple[str, np.ndarray]:
    """Load a protein structure from file as atom37 representation.

    Parameters
    ----------
    path : Path
        Path to the structure file

    Returns
    -------
    sequence : str
        Amino acid sequence in one-letter code
    coords : np.ndarray
        Array of shape (N, 37, 3) containing the coordinates
    """

    filetype = Path(path).name.split(".", 1)[-1].lower()
    assert filetype in {"pdb", "pdb.gz", "cif", "cif.gz"}, (
        f"Unsupported file type: {filetype}"
    )

    structure: gemmi.Structure
    if filetype in {"cif", "cif.gz"}:
        structure = gemmi.read_structure(str(path))
    else:
        structure = gemmi.read_pdb(str(path))
    raw_chain = structure[0].subchains()[0]

    # Get sequence
    L = len(raw_chain)
    coords = np.full((L, 37, 3), np.nan, dtype=np.float32)
    aa_list: list[str] = []
    for res_i, res in enumerate(raw_chain):
        res: gemmi.Residue
        res_name: C.ResidueName = C.residue.get_residue_name_with_unk(
            res.name, C.ChainType.PROTEIN
        )
        aa_list.append(res_name.one_letter)

        for atom in res:
            atom: gemmi.Atom
            atom_name = atom.name
            aidx = atom37_order.get(atom_name, None)
            if aidx is not None:
                coords[res_i, aidx] = atom.pos.tolist()
    sequence = "".join(aa_list)
    return sequence, coords


def write_protein_structure(
    sequence: str,
    coords: np.ndarray,
    path: str | Path,
    chain_id: str = "A",
) -> None:
    """Write a protein structure from atom37 representation to file.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 37, 3) containing the coordinates
    path : Path
        Path to the output structure file
    chain_id : str
        Chain identifier for the output structure
    """

    residue_atoms: dict[str, tuple[str, ...]] = C.atom.residue_atoms

    structure = gemmi.Structure()
    try:
        model = gemmi.Model(1)
    except TypeError:  # gemmi<0.6.6
        model = gemmi.Model("1")
    chain = gemmi.Chain(chain_id)

    for res_i in range(coords.shape[0]):
        aa = sequence[res_i]
        res_name = protein_one_letter_to_residue_name[aa]
        residue = gemmi.Residue()
        residue.name = res_name.name
        residue.seqid.num = res_i + 1
        for atom_name in residue_atoms[res_name.name]:
            atom_idx = atom37_order[atom_name]
            xyz = coords[res_i, atom_idx]
            if not np.isfinite(xyz).all():
                continue
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(atom_name[0])
            x, y, z = xyz.tolist()
            atom.pos = gemmi.Position(x, y, z)
            residue.add_atom(atom)
        chain.add_residue(residue)

    model.add_chain(chain)
    structure.add_model(model)
    structure.write_pdb(str(path))
