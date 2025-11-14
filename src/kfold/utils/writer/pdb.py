import numpy as np
from rdkit import Chem

import kfold.constants as C
from kfold.data.tokenized import TokenizedStructure


def to_pdb(structure: TokenizedStructure) -> str:  # noqa: PLR0915
    """Write a structure into a PDB file.

    Parameters
    ----------
    structure : TokenizedStructure
        The input structure

    Returns
    -------
    str
        the output PDB file
    """
    pdb_lines = []

    atom_index = 1

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    chain_id_iter = [chr(i) for i in range(65, 91)]  # 'A' to 'Z'

    tokens = structure.token  # [Ntoken, ...]
    atoms = structure.atom  # [Ntoken, 24, ...]
    last_asym_id = -1
    atom_map: dict[tuple[int, int], int] = {}

    # Add all atom sites.
    for i in range(len(tokens)):
        should_terminate = last_asym_id != -1 and last_asym_id != tokens.asym_id[i]
        if should_terminate:
            asym_id = tokens.asym_id[i - 1]
            chain_tag = chain_id_iter[asym_id - 1]
            res_type = tokens.res_type[i - 1]
            res_name = str(C.residue.residue_index_to_name[res_type].name)
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

        asym_id = tokens.asym_id[i]
        chain_tag = chain_id_iter[asym_id - 1]
        res_type = tokens.res_type[i]
        res_name = str(C.residue.residue_index_to_name[res_type].name)
        residue_index = tokens.residue_index[i]

        record_type = (
            "ATOM" if tokens.chain_type[i] != C.chain.ChainType.Ligand else "HETATM"
        )

        for j in range(24):
            if not atoms.resolved_mask[i, j]:
                continue

            atom_name_chars = atoms.ref_atom_name_chars[i, j]
            atom_name_chars = [chr(c + 32) for c in atom_name_chars if c != 0]
            atom_name = "".join(atom_name_chars)
            charge = atoms.ref_charge[i, j].item()
            element = periodic_table.GetElementSymbol(atoms.ref_element[i, j].item())
            pos = atoms.label_coords[i, j, 0]
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
    bonds = structure.bond  # [Nbond, ...]
    for bidx in range(len(bonds)):
        i1, i2 = bonds.token_index[bidx]
        # Remap
        i1 = np.argmax(tokens.token_index == i1).item()
        i2 = np.argmax(tokens.token_index == i2).item()
        j1, j2 = bonds.atom_index[bidx]
        if not atoms.resolved_mask[i1, j1] or not atoms.resolved_mask[i2, j2]:
            continue
        atom1_idx = atom_map[(i1, j1)]
        atom2_idx = atom_map[(i2, j2)]
        conect_line = f"CONECT{atom1_idx:>5}{atom2_idx:>5}"
        pdb_lines.append(conect_line)

    pdb_lines.append("END")
    pdb_lines.append("")
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines)
