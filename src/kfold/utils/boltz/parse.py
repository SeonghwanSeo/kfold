import gzip
import io
import warnings
from functools import lru_cache
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem

import kfold.constants as C


@lru_cache(maxsize=128)
def get_atom_element(atom_name: str) -> int:
    atom = Chem.Atom(atom_name)
    return atom.GetAtomicNum()


def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Args:
        name (str): The atom name.

    Returns:
        tuple[int, int, int, int]: The converted atom name.
    """
    name = name.strip()
    name_int = [ord(c) - 32 for c in name]
    name_int = name_int + [0] * (4 - len(name_int))  # pad to 4 characters
    return tuple(name_int)  # pyright: ignore


def parse_mmcif(path: str | Path | io.StringIO, id: str | None = None) -> BoltzStructure:
    """Parse a structure from the mmcif file.
    Reference code: https://github.com/jwohlwend/boltz/blob/c9b6af1f13dd22d7ad394131928f0772ec37a82f/src/boltz/data/parse/mmcif.py

    Args:
        path (str | Path | io.StringIO): The path to the mmcif file or a StringIO object.
        id (str | None, optional): The structure ID. If None, it will be inferred from the
            file name. Defaults to None.
    Returns:
        BoltzStructure: The parsed structure.
    """
    # TODO: add test script.
    warnings.warn("SeoSeonghwan: I did not test this function yet.")

    # If id is not provided, infer it from the path
    if id is None and isinstance(path, str | Path):
        id = Path(path).stem

    # Load mmCIF file
    block: gemmi.cif.Block
    if isinstance(path, io.StringIO):
        block = gemmi.cif.read_string(path.getvalue())[0]
    else:
        assert Path(path).exists(), f"mmCIF file does not exist: {path}"
        block = gemmi.cif.read(str(path))[0]
    structure = gemmi.make_structure_from_block(block)

    return parse_gemmi(structure)


def parse_gemmi(structure: gemmi.Structure) -> BoltzStructure:
    """Parse a structure from the gemmi structure object.

    Args:
        structure (gemmi.Structure): The gemmi structure object.

    Returns:
        BoltzStructure: The parsed structure.
    """
    # TODO: add test script.
    warnings.warn("SeoSeonghwan: I did not test this function yet.")

    # Clean up the structure
    structure.merge_chain_parts()
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()

    # add assembly
    if structure.assemblies:
        how = gemmi.HowToNameCopiedChain.AddNumber
        assembly_name = structure.assemblies[0].name
        structure.transform_to_assembly(assembly_name, how=how)

    # === Parse entities === #
    entity: gemmi.Entity
    entities: dict[str, gemmi.Entity] = {}
    entity_ids: dict[str, int] = {}
    for i, entity in enumerate(structure.entities):
        entity_id = i + 1  # entity IDs start from 1
        if entity.entity_type.name == "Water":
            continue
        for subchain_id in entity.subchains:
            entities[subchain_id] = entity
            entity_ids[subchain_id] = entity_id

    def is_amino_acid(res_name: str) -> bool:
        """Check if a residue name is a amino acid"""
        res_info = gemmi.find_tabulated_residue(res_name)
        return res_info.is_amino_acid()

    chain_data = []
    res_data = []
    atom_data = []

    with gzip.open(pdbgz_path, "rb") as f:
        # Read the gzipped PDB file
        decoded_data = f.read().decode("utf-8")
        structure = gemmi.read_pdb_string(decoded_data)

    entity = structure.entities[0]
    raw_chain = structure[0].subchains()[0]

    # Get sequence from entity
    full_seq = entity.full_sequence  # [ALA, GLY, MSE, ...]
    # Ignore non-amino-acids
    full_seq = [is_amino_acid(res) and res or "UNK" for res in full_seq]
    # Ignore microheterogeneities (pick first)
    full_seq = [gemmi.Entity.first_mon(item) for item in full_seq]

    # Align full sequence to polymer residues
    result = gemmi.align_sequence_to_polymer(
        full_seq=full_seq,
        polymer=raw_chain,
        polymer_type=entity.polymer_type,
        scoring=gemmi.AlignmentScoring(),
    )

    chain_type = C.ChainType.Protein
    chain_name = "A"
    sym_id = 0
    asym_id = 0
    entity_id = 0

    # Add residue, atom
    i = 0
    atom_idx = 0
    for res_idx, match in enumerate(result.match_string):
        res_name = full_seq[res_idx]
        res_type = 0  # 0 indicates Protein

        # Check if we have a match in the structure
        name_to_atom: dict[str, gemmi.Atom] = {}
        if match == "|":
            res = raw_chain[i]
            assert res.name == res_name, f"Alignment mismatch! {res.name} vs {res_name}"
            i += 1
            # collect residue atoms
            name_to_atom = {a.name.upper(): a for a in res}

        # Map MSE to MET
        if res_name == "MSE":
            res_name = "MET"
            if "SE" in name_to_atom:
                name_to_atom["SD"] = name_to_atom["SE"]

        # Map non-standard residues to UNK
        aatype = RESTYPE_3TO1.get(res_name, "X")

        is_standard = aatype != "X"
        is_present = True  # all residues are present in the AFDB

        # this is not used in our case, but we keep it for consistency
        atom_center_idx = 1  # CA
        if is_standard:
            if res_name == "GLY":
                atom_disto_idx = 1  # CA
            else:
                atom_disto_idx = 3  # CB
        else:
            atom_disto_idx = 3  # CB

        atom_center = atom_idx + atom_center_idx
        atom_disto = atom_idx + atom_disto_idx
        res_data.append(
            (
                res_name,
                res_type,
                res_idx,
                atom_idx,
                len(name_to_atom),  # num atoms
                atom_center,
                atom_disto,
                is_standard,
                is_present,
            )
        )
        for name, atom in name_to_atom.items():
            coords = atom.pos.x, atom.pos.y, atom.pos.z
            charge = 0  # we do not use this property
            is_present = True  # all atoms are present in the AFDB
            atom_data.append(
                (
                    convert_atom_name(name),
                    get_atom_element(name[0]),
                    charge,
                    coords,
                    is_present,
                    charge,
                )
            )
            atom_idx += 1

    # Convert into datatypes
    atom_arr = np.array(atom_data, dtype=Atom)
    bond_arr = np.array([], dtype=Bond)
    residue_arr = np.array(res_data, dtype=Residue)
    chain_arr = np.array(chain_data, dtype=Chain)
    connection_arr = np.array([], dtype=Connection)
    mask_arr = np.ones(len(chain_data), dtype=bool)
    interface_arr = np.array([], dtype=Interface)

    # create a chain data (single chain)
    chain_data.append(
        (
            chain_name,
            chain_type,
            entity_id,
            sym_id,
            asym_id,
            0,  # starting atom_idx,
            len(atom_arr),  # number of atoms
            0,  # starting res_idx,
            len(residue_arr),  # number of residues
            0,  # cyclic_period, not used in AFDB
        )
    )

    return BoltzStructure(
        atoms=atom_arr,
        bonds=bond_arr,
        residues=residue_arr,
        chains=chain_arr,
        connections=connection_arr,
        interfaces=interface_arr,
        mask=mask_arr,
    )
