"""MMCIF writer utilities."""

import logging
from collections import defaultdict

import gemmi
import numpy as np

import kfold.constants as C
from kfold.data.types.structure import RefStructure

# Set up logging
logger = logging.getLogger(__name__)


# === Core implementation === #
def to_mmcifstring(
    struct: RefStructure,
    save_apo: bool = False,
) -> str:
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    struct : RefStructure
        The input structure containing chain metadata and coordinates.
    save_apo : bool, optional
        Whether to save the apo form (default is False).

    Returns
    ----
    str
        The output MMCIF file content.
    """
    metadata = struct.metadata

    structure = gemmi.Structure()

    # === Create entity lists === #
    entity_ctypes: dict[int, C.ChainType] = {}
    entity_sequences: dict[int, list[str]] = {}
    entity_asym_ids: dict[int, list[str]] = defaultdict(list)
    for chain_i in range(len(struct.chains)):
        chain_meta = metadata.chains[chain_i]
        ref_chain = struct.chains[chain_i]
        entity_id = ref_chain.entity_id
        if entity_id not in entity_sequences:
            # First time seeing this entity, store its type and sequence
            entity_ctypes[entity_id] = ref_chain.ctype
            entity_sequences[entity_id] = ref_chain.get_ccd_sequence()
        # Append asym_id (chain name) to the entity's list
        entity_asym_ids[entity_id].append(chain_meta.name)

    entity_list: list[gemmi.Entity] = []
    for entity_id in sorted(entity_sequences.keys()):
        entity = gemmi.Entity(str(entity_id))
        ctype = entity_ctypes[entity_id]
        if ctype.is_polymer:
            entity.entity_type = gemmi.EntityType.Polymer
            match ctype:
                case C.ChainType.PROTEIN:
                    entity.polymer_type = gemmi.PolymerType.PeptideL
                case C.ChainType.RNA:
                    entity.polymer_type = gemmi.PolymerType.Rna
                case C.ChainType.DNA:
                    entity.polymer_type = gemmi.PolymerType.Dna
                case _:
                    raise ValueError(f"Unsupported polymer chain type: {ctype}")
        else:
            # FIXME: add glycan support later
            entity.entity_type = gemmi.EntityType.NonPolymer
        entity.full_sequence = entity_sequences[entity_id]
        entity.subchains = entity_asym_ids[entity_id]
        entity_list.append(entity)

    entities: gemmi.EntityList = gemmi.EntityList(entity_list)
    del entity_list  # free memory
    structure.entities = entities

    # === Build Model === #
    model = gemmi.Model("1")
    for chain_i in range(len(struct.chains)):
        ref_chain = struct.chains[chain_i]
        chain_meta = metadata.chains[chain_i]
        ctype = ref_chain.ctype

        # Retrieve layout data
        res_layout = ref_chain.residue
        atom_layout = ref_chain.atom

        atom_names: list[str] = atom_layout.name.tolist()
        atom_elements: list[int] = atom_layout.element.tolist()
        atom_charges: list[int] = atom_layout.charge.tolist()

        if save_apo:
            atom_coords = atom_layout.apo_coords
        else:
            atom_coords = atom_layout.coords

        chain_id = chain_meta.chain_name  # e.g., "A", "B", etc.
        entity_id = chain_meta.entity_id

        # Determine if it is a polymer (ATOM) or non-polymer/ligand (HETATM)
        is_polymer = chain_meta.ctype.is_polymer
        het_flag = "A" if is_polymer else "H"

        # Create gemmi chain
        # Note: In Gemmi, chain.name usually maps to auth_asym_id
        chain = gemmi.Chain(chain_id)

        # Iterate over residues
        for res_i in range(ref_chain.num_residues):
            residue_index = res_i + 1  # 1-based indexing

            # Iterate over atoms
            atoms: list[gemmi.Atom] = []
            for atom_i in ref_chain.iter_residue_atoms(residue_index):
                # Check for valid coordinates (skip NaNs or Infs)
                xyz = atom_coords[atom_i]
                if not np.isfinite(xyz).all():
                    continue
                x, y, z = xyz.tolist()

                atom = gemmi.Atom()
                atom.name = atom_names[atom_i]
                atom.element = gemmi.Element(atom_elements[atom_i])
                atom.charge = atom_charges[atom_i]
                atom.pos = gemmi.Position(round(x, 3), round(y, 3), round(z, 3))

                # Set calculation flag (since this is likely a predicted structure)
                atom.calc_flag = gemmi.CalcFlag.Calculated

                atoms.append(atom)

            # Only add residue if it has atoms
            if len(atoms) > 0:
                residue = gemmi.Residue()
                if ctype.is_polymer:
                    residue.label_seq = residue_index  # 1-based indexing
                residue.name = str(res_layout.name[res_i])
                residue.seqid.num = residue_index
                residue.het_flag = het_flag  # 'A' for polymer, 'H' for non-polymer
                residue.entity_id = str(entity_id)  # Link to _entity category
                residue.subchain = chain_id  # Maps to _atom_site.label_asym_id
                for atom in atoms:
                    residue.add_atom(atom)
                chain.add_residue(residue)

        model.add_chain(chain)

    structure.add_model(model)

    structure.setup_entities()

    # Create the document
    doc: gemmi.cif.Document = structure.make_mmcif_document()
    # Add custom categories for OST compatibility
    block = doc[0]
    _add_pdbx_nonpoly_scheme(block, structure)
    _add_pdbx_poly_seq_scheme(block, structure)
    _update_entity_poly(block, structure)
    _update_entity_poly_seq(block, structure)
    _update_chem_comp(block)

    return doc.as_string()


def _add_pdbx_poly_seq_scheme(block: gemmi.cif.Block, structure: gemmi.Structure):
    """
    Manually add the _pdbx_poly_seq_scheme category to the CIF block.
    This is required for OST compatibility and proper polymer parsing.
    """
    # Columns required for _pdbx_poly_seq_scheme
    columns = [
        "asym_id",  # label_asym_id (residue.subchain)
        "entity_id",  # entity_id
        "mon_id",  # residue name
        "seq_id",  # residue sequence number
        "pdb_strand_id",  # auth_asym_id (chain.name)
        "pdb_seq_num",  # auth_seq_id
        "pdb_ins_code",  # PDB insertion code
    ]
    loop = block.init_loop("_pdbx_poly_seq_scheme.", columns)
    # Iterate strictly over the first model (assuming single model structure for AF3)
    model = structure[0]
    for chain in model:
        for res in chain:
            # Check if residue is part of a polymer ('A' het_flag)
            if res.het_flag == "A":
                # Map values
                asym_id = res.subchain if res.subchain else chain.name
                entity_id = res.entity_id
                mon_id = res.name
                seq_num = str(res.seqid.num)
                strand_id = chain.name  # auth_asym_id
                ins_code = "." if res.seqid.icode == " " else res.seqid.icode
                loop.add_row(
                    [
                        asym_id,  # asym_id
                        entity_id,  # entity_id
                        mon_id,  # mon_id
                        seq_num,  # seq_id
                        strand_id,  # pdb_strand_id
                        seq_num,  # pdb_seq_num
                        ins_code,  # pdb_ins_code
                    ]
                )


def _add_pdbx_nonpoly_scheme(block: gemmi.cif.Block, structure: gemmi.Structure):
    """
    Manually add the _pdbx_nonpoly_scheme category to the CIF block.
    This is required for OST compatibility and proper ligand parsing.
    """
    # Columns required for _pdbx_nonpoly_scheme
    columns = [
        "asym_id",  # label_asym_id (residue.subchain)
        "entity_id",  # entity_id
        "mon_id",  # residue name
        "ndb_seq_num",  # label_seq_id
        "pdb_seq_num",  # auth_seq_id
        "auth_seq_num",  # auth_seq_id
        "pdb_mon_id",  # auth_comp_id
        "auth_mon_id",  # auth_comp_id
        "pdb_strand_id",  # auth_asym_id (chain.name)
        "pdb_ins_code",  # PDB insertion code
    ]

    loop = block.init_loop("_pdbx_nonpoly_scheme.", columns)

    # Iterate strictly over the first model (assuming single model structure for AF3)
    model = structure[0]

    for chain in model:
        for res in chain:
            # Check if residue is explicitly marked as non-polymer ('H')
            # or if the entity it belongs to is non-polymer
            if res.het_flag == "H":
                # Map values
                asym_id = res.subchain if res.subchain else chain.name
                entity_id = res.entity_id
                mon_id = res.name
                seq_num = str(res.seqid.num)
                strand_id = chain.name  # auth_asym_id
                ins_code = "." if res.seqid.icode == " " else res.seqid.icode

                loop.add_row(
                    [
                        asym_id,  # asym_id
                        entity_id,  # entity_id
                        mon_id,  # mon_id
                        seq_num,  # ndb_seq_num
                        seq_num,  # pdb_seq_num
                        seq_num,  # auth_seq_num
                        mon_id,  # pdb_mon_id
                        mon_id,  # auth_mon_id
                        strand_id,  # pdb_strand_id
                        ins_code,  # pdb_ins_code
                    ]
                )


def _update_entity_poly(block: gemmi.cif.Block, structure: gemmi.Structure):
    """Update the _entity_poly_seq category in the CIF block to reflect sequences."""
    table: gemmi.cif.Table = block.find_mmcif_category("_entity_poly.")

    rows = []
    for row in table:
        rows.append(
            [
                row["entity_id"],
                row["type"],
                row["pdbx_strand_id"],
                row["pdbx_seq_one_letter_code"],
                row["pdbx_seq_one_letter_code"],
            ],
        )
    loop: gemmi.cif.Loop = block.init_mmcif_loop(
        "_entity_poly.",
        [
            "entity_id",
            "type",
            "pdbx_strand_id",
            "pdbx_seq_one_letter_code",
            "pdbx_seq_one_letter_code_can",
        ],
    )
    for row in rows:
        loop.add_row(row)


def _update_entity_poly_seq(block: gemmi.cif.Block, structure: gemmi.Structure):
    """Update the _entity_poly_seq category in the CIF block to reflect sequences."""
    table: gemmi.cif.Table = block.find_mmcif_category("_entity_poly_seq.")

    rows = []
    for row in table:
        rows.append([row["entity_id"], row["num"], row["mon_id"], "n"])
    loop: gemmi.cif.Loop = block.init_mmcif_loop(
        "_entity_poly_seq.",
        [
            "entity_id",
            "num",
            "mon_id",
            "hetero",
        ],
    )
    for row in rows:
        loop.add_row(row)


def _update_chem_comp(block: gemmi.cif.Block):
    """Add or modify the _chem_comp category in the CIF block to include residue types."""
    table: gemmi.cif.Table = block.find_mmcif_category("_chem_comp.")

    rows = []
    for row in table:
        res_id = row["id"]
        res: gemmi.ResidueInfo = gemmi.find_tabulated_residue(res_id)
        if res is not None:
            is_standard = res.is_standard()
            if res.kind == gemmi.ResidueKind.AA:
                res_type = "'L-peptide linking'"
            elif res.kind == gemmi.ResidueKind.RNA:
                res_type = "'RNA linking'"
            elif res.kind == gemmi.ResidueKind.DNA:
                res_type = "'DNA linking'"
            else:
                res_type = "non-polymer"
            res_weight = f"{res.weight:.3f}"
        else:
            is_standard = False
            res_type = "."
            res_weight = "."

        rows.append(
            [
                res_id,
                res_type,
                ".",
                ".",
                res_weight,
                "y" if not is_standard else "n",
            ]
        )
    loop: gemmi.cif.Loop = block.init_mmcif_loop(
        "_chem_comp.",
        [
            "id",
            "type",
            "name",
            "formula",
            "formula_weight",
            "mon_nstd_flag",
        ],
    )
    for row in rows:
        loop.add_row(row)
