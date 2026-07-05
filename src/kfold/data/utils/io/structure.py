from pathlib import Path

import gemmi
import numpy as np
import zstandard as zstd

import kfold.constants as C

# Constants
atom37_order: dict[str, int] = C.atom.protein_atom37_order
atom29_order: dict[str, int] = C.atom.nucleic_acid_atom29_order
protein_one_letter_to_residue_name: dict[str, C.ResidueName] = (
    C.residue.protein_one_letter_to_residue_name
)


def get_structure_filetype(path: str | Path) -> str:
    """Infer the underlying structure file type."""
    name = Path(path).name.lower()
    for filetype in ("pdb", "cif"):
        if (
            name.endswith(f".{filetype}")
            or name.endswith(f".{filetype}.gz")
            or name.endswith(f".{filetype}.zst")
        ):
            return filetype
    raise ValueError(f"Unsupported file type: {name}")


def read_gemmi_structure(path: str | Path) -> gemmi.Structure:
    """Read PDB/mmCIF files, including zstd-compressed archives."""
    path = Path(path)
    filetype = get_structure_filetype(path)
    if path.name.lower().endswith(".zst"):
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                text = reader.read().decode("utf-8")
        if filetype == "pdb":
            return gemmi.read_pdb_string(text)
        return gemmi.read_structure_string(text)
    if filetype == "cif":
        return gemmi.read_structure(str(path))
    return gemmi.read_pdb(str(path))


def _read_protein_chain(raw_chain: gemmi.Chain) -> tuple[str, np.ndarray]:
    """Read one gemmi chain as atom37 protein coordinates."""
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

    structure = read_gemmi_structure(path)
    raw_chain = structure[0].subchains()[0]
    return _read_protein_chain(raw_chain)


def read_protein_multimer_structure(path: str | Path) -> dict[str, dict]:
    """Load all protein chains in a multimer structure.

    Returns a dictionary keyed by chain identifier.  Coordinates are kept in the
    original shared frame, so relative chain placement is preserved.
    """
    structure = read_gemmi_structure(path)
    chains: dict[str, dict] = {}
    for raw_chain in structure[0]:
        if len(raw_chain) == 0:
            continue
        chain_id = raw_chain.name
        sequence, coords = _read_protein_chain(raw_chain)
        chains[chain_id] = {
            "seq": sequence,
            "coords": coords,
            "chain_type": "protein",
        }
    return chains


def read_rna_structure(path: str | Path) -> tuple[str, np.ndarray]:
    """Load an RNA structure from file as atom29 representation.

    Returns
    -------
    sequence : str
        RNA sequence in one-letter code.
    coords : np.ndarray
        Array of shape (N, 29, 3) containing coordinates in
        `C.atom.nucleic_acid_atom29` order.
    """

    structure = read_gemmi_structure(path)
    raw_chain = structure[0].subchains()[0]

    L = len(raw_chain)
    coords = np.full((L, 29, 3), np.nan, dtype=np.float32)
    base_list: list[str] = []
    for res_i, res in enumerate(raw_chain):
        res_name: C.ResidueName = C.residue.get_residue_name_with_unk(
            res.name, C.ChainType.RNA
        )
        base_list.append(res_name.one_letter)

        for atom in res:
            aidx = atom29_order.get(atom.name, None)
            if aidx is not None:
                coords[res_i, aidx] = atom.pos.tolist()
    sequence = "".join(base_list)
    return sequence, coords


def read_dna_structure(path: str | Path) -> tuple[str, np.ndarray]:
    """Load a DNA structure from file as atom29 representation."""

    structure = read_gemmi_structure(path)
    raw_chain = structure[0].subchains()[0]

    L = len(raw_chain)
    coords = np.full((L, 29, 3), np.nan, dtype=np.float32)
    base_list: list[str] = []
    for res_i, res in enumerate(raw_chain):
        res_name: C.ResidueName = C.residue.get_residue_name_with_unk(
            res.name, C.ChainType.DNA
        )
        base_list.append(res_name.one_letter)

        for atom in res:
            aidx = atom29_order.get(atom.name, None)
            if aidx is not None:
                coords[res_i, aidx] = atom.pos.tolist()
    sequence = "".join(base_list)
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
