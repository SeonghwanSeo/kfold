"""PDB writer utilities."""

from functools import lru_cache

import numpy as np
from rdkit import Chem

import kfold.constants as C
from kfold.data.structure import TokenizedStructure
from kfold.utils import errors


# === Core implementation === #
@lru_cache(maxsize=1)
def _get_periodic_table() -> Chem.PeriodicTable:
    return Chem.GetPeriodicTable()


def to_pdbstring(
    struct: TokenizedStructure,
    conformer_id: int = 0,
    is_predicted: bool = True,
    save_apo: bool = False,
) -> str:  # noqa: PLR0915
    """Write a structure into a PDB file.

    Parameters
    ----------
    struct : TokenizedStructure
    structure : TokenizedStructure
        The input structure
    conformer_id : int, optional
        The conformer ID to write (default is 0)
    save_apo : bool, optional
        Whether to save the apo form (default is False)
    is_predicted : bool, optional
        Whether the structure is predicted (default is False)

    Returns
    -------
    str
        the output PDB file
    """
    tokens = struct.token  # [Ntoken, ...]
    atoms = struct.atom  # [Ntoken, 24, ...]

    if struct.num_chains > 52:
        raise errors.PDBWriterMaxChainError(
            "PDB format supports a maximum of 52 chains (A-Z, a-z). "
            f"Found {struct.num_chains} chains."
        )

    # Atom informations
    ref_atom_name_chars = atoms.ref_atom_name_chars  # [Ntoken, 24, max_name_length]
    ref_atom_charges = atoms.ref_charge  # [Ntoken, 24]
    ref_atom_elements = atoms.ref_element  # [Ntoken, 24]

    # Select coordinates and mask based on the mode
    # atom_coords: [Ntoken, 24, 3]
    # atom_mask: [Ntoken, 24]
    if save_apo:
        # NOTE: ignore `is_predicted` flag when saving apo
        atom_coords = atoms.apo_coords[:, :, conformer_id, :]
        atom_mask = atoms.apo_mask[:, :, conformer_id]
    elif is_predicted:
        atom_coords = atoms.coords[:, :, conformer_id, :]
        atom_mask = np.ones_like(atoms.resolved_mask)
    else:
        atom_coords = atoms.coords[:, :, conformer_id, :]
        atom_mask = atoms.resolved_mask

    # Load periodic table for element mapping
    periodic_table = _get_periodic_table()

    # Add all atom sites.
    atom_index = 1
    pdb_lines = []
    last_asym_id = -1
    atom_map: dict[tuple[int, int], int] = {}

    # A-Z + a-z for chain IDs
    chain_id_iter = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

    for i in range(len(tokens)):
        # Check for chain termination.
        should_terminate = i > 0 and last_asym_id != tokens.asym_id[i]
        if should_terminate:
            asym_id = last_asym_id
            chain_tag = chain_id_iter[asym_id - 1]
            res_name = C.residue.residue_id_to_name[tokens.res_type[i - 1]].name
            residue_index = tokens.residue_index[i - 1]
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_name:>3} "
                f"{chain_tag:>1}{residue_index:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

        # Chain Information
        asym_id = tokens.asym_id[i]
        chain_tag = chain_id_iter[asym_id - 1]

        # Residue Information
        res_name = C.residue.residue_id_to_name[tokens.res_type[i]].name
        residue_index = tokens.residue_index[i]

        # Atom Information
        num_atoms = tokens.num_atoms[i]
        record_type = (
            "ATOM" if tokens.chain_type[i] != C.chain.ChainType.LIGAND else "HETATM"
        )

        # Inspect each atom in the residue.
        for j in range(num_atoms):
            if not atom_mask[i, j]:
                continue

            atom_name_chars = ref_atom_name_chars[i, j]
            atom_name = "".join([chr(c + 32) for c in atom_name_chars if c != 0])
            charge = int(ref_atom_charges[i, j])
            element = periodic_table.GetElementSymbol(int(ref_atom_elements[i, j]))
            pos = atom_coords[i, j]
            occupancy = 1.00
            b_factor = 1.0
            alt_loc = ""
            insertion_code = ""

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {atom_name:<4}{alt_loc:>1}"
                f"{res_name:>3} {chain_tag:>1}"
                f"{residue_index:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_map[(i, j)] = atom_index
            atom_index += 1

    # Dump CONECT records.
    bonds = struct.bond  # [Nbond, ...]
    for bidx in range(len(bonds)):
        i1, i2 = bonds.token_index[bidx]
        # Remap
        i1 = np.argmax(tokens.token_index == i1).item()
        i2 = np.argmax(tokens.token_index == i2).item()
        j1, j2 = bonds.atom_index[bidx]
        if not (atom_mask[i1, j1] and atom_mask[i2, j2]):
            continue
        atom1_idx = atom_map[(i1, j1)]
        atom2_idx = atom_map[(i2, j2)]
        conect_line = f"CONECT{atom1_idx:>5}{atom2_idx:>5}"
        pdb_lines.append(conect_line)

    pdb_lines.append("END")
    pdb_lines.append("")
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines)
