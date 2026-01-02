"""Pipeline to parse structure and metadata from structure files (PDB/MMCIF)."""

import itertools
import logging
import pathlib
from typing import Any

import gemmi
import numpy as np
import scipy.spatial.distance

import kfold.constants as C
from kfold.data import schema, structure
from kfold.data.ccd import CCD, Component

logger = logging.getLogger(__name__)

# Type aliases for better readability
EntityId = int
AsymId = str
SymId = int
AuthId = str
ResKey = tuple[AsymId, str, int | None]

# Constants
polymer_type_to_chain_type: dict[gemmi.PolymerType, C.ChainType] = {
    gemmi.PolymerType.PeptideL: C.ChainType.PROTEIN,
    gemmi.PolymerType.Rna: C.ChainType.RNA,
    gemmi.PolymerType.Dna: C.ChainType.DNA,
}
chain_type_to_polymer_type: dict[C.ChainType, gemmi.PolymerType] = {
    C.ChainType.PROTEIN: gemmi.PolymerType.PeptideL,
    C.ChainType.RNA: gemmi.PolymerType.Rna,
    C.ChainType.DNA: gemmi.PolymerType.Dna,
}
# three-letter codes
chain_type_to_standard_residues: dict[C.ChainType, set[str]] = {
    C.ChainType.PROTEIN: set(C.residue.PROTEIN_RESIDUES_EXTENDED_STR),
    C.ChainType.RNA: set(C.residue.RNA_RESIDUES_STR),
    C.ChainType.DNA: set(C.residue.DNA_RESIDUES_STR),
    C.ChainType.LIGAND: set(),
    C.ChainType.ION: set(),
}
# one-letter codes
chain_type_to_standard_restypes: dict[C.ChainType, set[str]] = {
    C.ChainType.PROTEIN: set(C.residue.PROTEIN_AMINO_ACIDS),
    C.ChainType.RNA: set(C.residue.RNA_BASES),
    C.ChainType.DNA: set(C.residue.DNA_BASES),
    C.ChainType.LIGAND: set(),
    C.ChainType.ION: set(),
}
chain_type_to_unk: dict[C.ChainType, str] = {
    C.ChainType.PROTEIN: "UNK",
    C.ChainType.RNA: "N",
    C.ChainType.DNA: "DN",
    C.ChainType.LIGAND: "UNK",
    C.ChainType.ION: "UNK",
}


# ==================================================
# Template function for CIF parsing
# ==================================================
def parse_cif(
    cif_path: str | pathlib.Path,
    ccd: CCD,
    source: str = "rcsb",
    max_chains: int | None = None,
) -> structure.RefStructure:
    """Parse a CIF file and return a gemmi.cif.Document object.
    NOTE: This is just a template function. Additional filtering can be
    inserted as needed, e.g., date cutoff, number of chains, etc.
    """
    # Read CIF file
    doc: gemmi.cif.Document = gemmi.cif.read_file(str(cif_path))
    block: gemmi.cif.Block = doc[0]

    # Get metadata
    # Handle cases like "1abc.cif.gz"
    name = pathlib.Path(cif_path).name.split(".")[0]
    metadata = prepare_metadata(name, block, source)

    # Prepare raw structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)

    # Clean up raw structure
    clean_up_raw_structure(raw_struct)

    # Expand first assembly if available
    expand_first_assembly(raw_struct)

    # Prepare reference structure
    ref_struct = prepare_ref_structure(raw_struct, metadata, ccd)

    # Insert coordinates
    insert_coordinates(ref_struct, raw_struct, metadata)

    # Clean valid chains
    validate_chain_geometry(ref_struct)

    # Get interfaces
    detect_interfaces_and_prune_clashes(ref_struct)

    # Drop invalid chains
    prune_invalid_chains(ref_struct)

    if max_chains is not None:
        # Limit number of chains for testing
        crop_substructure(ref_struct, max_chains)

    return ref_struct


# ==================================================
# Helper functions for Metadata parsing
# ==================================================
def get_first_value(block: gemmi.cif.Block, tag: str, cast: type = str) -> Any | None:
    """Helper to get the first value of a tag, or None if missing."""
    values = block.find_values(tag)
    if len(values) > 0:
        try:
            return cast(values[0])
        except Exception:
            return None
    return None


def prepare_experiment_record(block: gemmi.cif.Block) -> schema.ExperimentRecord:
    """Parse RCSB PDB metadata from CIF block."""
    # PDB ID
    pdb_id = get_first_value(block, "_entry.id")

    # Release Date
    rev_dates = block.find_values("_pdbx_audit_revision_history.revision_date")
    release_date = min(rev_dates) if rev_dates else None

    # Method (e.g., X-RAY DIFFRACTION)
    method = get_first_value(block, "_exptl.method")

    # Resolution (Handle X-ray vs EM vs NMR)
    # X-ray standard
    resolution = get_first_value(block, "_refine.ls_d_res_high", float)
    if not resolution:
        # EM
        resolution = get_first_value(block, "_em_3d_reconstruction.resolution", float)
    if not resolution:
        # Fallback
        resolution = get_first_value(block, "_reflns.d_resolution_high", float)

    # Temperature (Kelvin)
    temp = get_first_value(block, "_diffrn.ambient_temp")

    # pH (Crystallization condition)
    ph = get_first_value(block, "_exptl_crystal_grow.pH")

    return schema.ExperimentRecord(
        pdb_id=pdb_id,
        resolution=resolution,
        method=method,
        release_date=release_date,
        pH=ph,
        temperature=temp,
    )


def prepare_metadata(
    name: str, block: gemmi.cif.Block, source: str = "rcsb"
) -> schema.Metadata:
    """Parse metadata from CIF block."""
    # Parse experiment record
    if source == "rcsb":
        exp_record = prepare_experiment_record(block)
        assert exp_record.pdb_id is not None, "PDB ID is missing in metadata."
    else:
        exp_record = None

    return schema.Metadata(
        id=name,
        source=source,
        exp=exp_record,
        chains=[],  # Filled later in parsing
        interfaces=[],  # Filled later in parsing
    )


# ==================================================
# Helper functions for gemmi structure validation
# ==================================================
def clean_up_raw_structure(raw_struct: gemmi.Structure) -> None:
    """Clean up gemmi Structure object in-place."""
    raw_struct.merge_chain_parts()
    raw_struct.remove_waters()
    raw_struct.remove_hydrogens()
    raw_struct.remove_alternative_conformations()
    raw_struct.remove_empty_chains()


def expand_first_assembly(raw_struct: gemmi.Structure) -> None:
    """Expand the first assembly in the gemmi Structure object in-place."""
    if len(raw_struct.assemblies) > 0:
        how = gemmi.HowToNameCopiedChain.AddNumber
        assembly_name = raw_struct.assemblies[0].name
        raw_struct.transform_to_assembly(assembly_name, how=how)


# ==================================================
# Helper functions for Structure parsing
# ==================================================
def get_residue_key(residue: gemmi.Residue) -> ResKey:
    asym_id: AsymId = residue.subchain
    seq_id: gemmi.SeqId = residue.seqid
    return (asym_id, seq_id.icode, seq_id.num)


def prepare_ref_chain(
    chain_type: C.ChainType,
    ccd_sequences: list[str],
    ccd: CCD,
    smiles: str | None = None,
    entity_id: int = 0,
    asym_id: int = 0,
    sym_id: int = 0,
    drop_leaving_atoms: bool = True,
) -> structure.Chain:
    """Get an empty reference chain structure.

    Parameters
    ----------
    chain_type : C.ChainType
        The type of the chain (protein, RNA, DNA, ligand, ion).
    ccd_sequences : list[str]
        List of residue names in the chain.
    ccd : CCD
        The CCD database object.
    smiles : str | None, optional
        SMILES string for ligand residues, by default None.
    entity_id : int
        The entity ID of the chain.
    asym_id : int
        The asymmetric unit ID of the chain.
    sym_id : int
        The symmetry ID of the chain.
    drop_leaving_atoms : bool, optional
        Whether to drop leaving atoms for polymer residues, by default True.
    """
    # ==================================================
    # Validate inputs
    # ==================================================
    if smiles is not None:
        assert chain_type == C.ChainType.LIGAND
        assert len(ccd_sequences) == 1
        assert ccd_sequences[0].startswith("LIG")

    standard_residues: set[str] = chain_type_to_standard_residues[chain_type]

    # ==================================================
    # Prepare residue information
    # ==================================================
    is_res_standards: list[bool] = []
    ref_mols: list[Component] = []
    num_residue_atoms: list[int] = []
    for name in ccd_sequences:
        if chain_type.is_protein and name == "MSE":
            # Replace selenomethionine with methionine
            name = "MET"
        if name in ccd:
            # Common molecule from CCD
            if name.startswith("LIG"):
                logging.info("Use custom ligand residue from CCD:", name)
            ref_mol = ccd[name]
        elif name.startswith("LIG"):
            # Ligand residue created from SMILES
            if smiles is None:
                raise ValueError(f"SMILES must be provided for ligand residue {name}.")
            ref_mol = Component.from_smiles(name, smiles)
        else:
            # Residue not found in CCD
            # NOTE: For polymers, this should not happen due to prior conversion to UNK.
            raise ValueError(f"Residue {name} not found in CCD database.")

        is_standard = name in standard_residues
        is_res_standards.append(is_standard)
        ref_mols.append(ref_mol)
        if drop_leaving_atoms:
            num_residue_atoms.append(ref_mol.num_non_leaving_atoms)
        else:
            num_residue_atoms.append(ref_mol.num_atoms)

    residue_struct = structure.Residue(
        name=np.array(ccd_sequences, dtype=np.dtype("<U6")),
        num_atoms=np.array(num_residue_atoms, dtype=np.uint8),
        is_standard=np.array(is_res_standards, dtype=bool),
    )

    # ==================================================
    # Prepare atom information
    # ==================================================
    atom_name_list: list[str] = []
    for ref_mol in ref_mols:
        if drop_leaving_atoms:
            atom_name_list.extend(ref_mol.non_leaving_atom_names)
        else:
            atom_name_list.extend(ref_mol.atom_names)
    num_atoms = len(atom_name_list)
    assert num_atoms == sum(num_residue_atoms), "Mismatch in total number of atoms."

    # Empty label coordinates and resolved flags
    label_coords = np.full((num_atoms, 3), np.nan, dtype=np.float32)
    is_atom_resolved = np.zeros((num_atoms,), dtype=bool)
    bfactors = np.full((num_atoms,), np.nan, dtype=np.float32)
    # Empty apo coordinates and pLDDT
    apo_coords = np.empty((0, num_atoms, 3), dtype=np.float32)
    apo_plddt = np.empty((0, num_atoms), dtype=np.float32)

    atom_struct = structure.Atom(
        name=np.array(atom_name_list, dtype=np.dtype("<U4")),
        label_coords=label_coords,
        is_resolved=is_atom_resolved,
        bfactor=bfactors,
        apo_coords=apo_coords,
        apo_plddt=apo_plddt,
    )

    # ==================================================
    # Prepare intra-residue bond information
    # Only ligand bonds are collected
    # ==================================================
    bond_residue_index_list: list[tuple[int, int]] = []
    bond_atom_name_list: list[tuple[str, str]] = []
    bond_type_list: list[int] = []
    if chain_type is C.ChainType.LIGAND:
        for residue_index, ref_mol in enumerate(ref_mols, start=1):
            # Get ref atom names
            if drop_leaving_atoms:
                ref_atom_names = ref_mol.non_leaving_atom_names
            else:
                ref_atom_names = ref_mol.atom_names
            for (atom_name1, atom_name2), bond_type in ref_mol.bonds.items():
                if atom_name1 in ref_atom_names and atom_name2 in ref_atom_names:
                    bond_residue_index_list.append((residue_index, residue_index))
                    bond_atom_name_list.append((atom_name1, atom_name2))
                    bond_type_list.append(bond_type)

    bond_struct = structure.Bond(
        residue_index=np.array(bond_residue_index_list, dtype=np.uint32).reshape(-1, 2),
        atom_name=np.array(bond_atom_name_list, dtype=np.dtype("<U4")).reshape(-1, 2),
        bond_type=np.array(bond_type_list, dtype=np.uint8),
    )

    return structure.Chain(
        chain_type=chain_type.value,
        entity_id=entity_id,
        asym_id=asym_id,
        sym_id=sym_id,
        residue=residue_struct,
        atom=atom_struct,
        bond=bond_struct,
    )


def prepare_ref_structure(
    raw_struct: gemmi.Structure,
    metadata: schema.Metadata,
    ccd: CCD,
) -> structure.RefStructure:
    """Prepare reference structure from gemmi CIF block and metadata."""

    # NOTE: According to AlphaFold3, remove crystallization aids for
    # X-ray structures
    excluded_ligands: set[str] = set(C.ccd.LIGAND_EXCLUSIONS)
    if metadata.exp is not None and metadata.exp.method is not None:
        if "XRAY" in metadata.exp.method.replace("-", "").upper():
            excluded_ligands.update(C.ccd.CRYSTALLIZATION_AIDS)

    # ==================================================
    # Prepare chain index mappings
    # ==================================================
    asym_id_to_entity_id: dict[AsymId, EntityId] = {}
    asym_id_to_sym_id: dict[AsymId, SymId] = {}
    for entity in raw_struct.entities:
        entity: gemmi.Entity
        assert entity.name.isdecimal(), "Entity name is not an integer."
        entity_id: EntityId = int(entity.name)
        assert entity_id > 0, "Entity ID should be positive integer."
        for sym_id, asym_id in enumerate(entity.subchains, start=1):
            assert asym_id not in asym_id_to_entity_id, (
                f"Duplicate asym_id {asym_id} found in entities."
            )
            asym_id_to_entity_id[asym_id] = int(entity_id)
            asym_id_to_sym_id[asym_id] = sym_id

    # Determine asym_id string to integer mapping
    asym_id_to_int: dict[AsymId, int] = {}
    for i, asym_id in enumerate(sorted(asym_id_to_entity_id.keys()), start=1):
        asym_id_to_int[asym_id] = i

    # ==================================================
    # Identify valid entities and chains
    # ==================================================
    valid_entities: list[gemmi.Entity] = []
    valid_entity_ids: set[EntityId] = set()
    valid_asym_ids: set[AsymId] = set()
    entity_id_to_entity: dict[EntityId, gemmi.Entity] = {}
    entity_id_to_chain_type: dict[EntityId, C.ChainType] = {}
    entity_id_to_seq: dict[EntityId, list[str]] = {}
    for entity in raw_struct.entities:
        entity: gemmi.Entity
        entity_id: EntityId = int(entity.name)

        if len(entity.subchains) == 0:
            # Skip entities without subchains
            continue

        if entity.entity_type == gemmi.EntityType.Polymer:
            # Protein, RNA, or DNA
            if entity.polymer_type not in polymer_type_to_chain_type:
                # Skip unsupported polymer types
                continue
            chain_type: C.ChainType = polymer_type_to_chain_type[entity.polymer_type]
            restypes: set[str] = chain_type_to_standard_restypes[chain_type]
            unk: str = chain_type_to_unk[chain_type]

            # Get CCD sequences
            ccd_sequences: list[str] = []
            for v in entity.full_sequence:
                # In the case of microheterogeneity, take the first monomer
                v = gemmi.Entity.first_mon(v)  # e.g., ALG/GLY -> ALG
                if chain_type.is_protein and v == "MSE":
                    # Replace selenomethionine with methionine
                    v = "MET"
                # Only retain standard residues and PTMs
                if C.residue.convert_ccd_name_to_one_letter(v, "?") not in restypes:
                    v = unk  # Map non-standard residues to UNK/N/DN
                ccd_sequences.append(v)

            lengths = len(ccd_sequences)
            if lengths < 4:
                # Skip too short polymer entities
                logger.debug(
                    f"{metadata.id}: "
                    f"Skipping entity {entity_id} with short length: {lengths} residues."
                )
                continue

            if set(ccd_sequences) <= {unk}:
                # Skip entities with non-standard residues
                logger.debug(
                    f"{metadata.id}: "
                    f"Skipping entity {entity_id} due to all non-standard residues."
                )
                continue

        elif entity.entity_type in {
            gemmi.EntityType.NonPolymer,
            gemmi.EntityType.Branched,
        }:
            # Ligand, ion, or branched ligands
            ref_asym_id: AsymId = entity.subchains[0]
            raw_chain: gemmi.ResidueSpan = raw_struct[0].get_subchain(ref_asym_id)
            ccd_sequences: list[str] = [res.name for res in raw_chain]

            # Check if all ligand residues are in CCD
            is_valid_entity = True
            for res_name in ccd_sequences:
                if res_name.startswith("LIG"):
                    # TODO: for Boltz1, all custom ligands are stored as NonPolymer
                    # with residue name "LIG". Handle them properly future.
                    continue
                if res_name in excluded_ligands:
                    # Exclude unwanted ligands
                    logging.info(f"Excluding ligand {res_name} in entity {entity_id}.")
                    is_valid_entity = False
                    break
                elif res_name not in ccd:
                    # Residue not found in CCD
                    logging.warning(f"Non-polymer residue {res_name} not found in CCD.")
                    is_valid_entity = False
                    break
            if not is_valid_entity:
                # Skip invalid entity
                continue

            # Check if the entity is an ion or not.
            if ccd_sequences[0] in C.ccd.IONS:
                assert len(ccd_sequences) == 1, "Ion entity has multiple residues."
                chain_type = C.ChainType.ION
            else:
                chain_type = C.ChainType.LIGAND
        else:
            # Skip other entity types
            continue

        # Store entity
        valid_entities.append(entity)
        valid_entity_ids.add(entity_id)
        entity_id_to_entity[entity_id] = entity
        entity_id_to_chain_type[entity_id] = chain_type
        entity_id_to_seq[entity_id] = ccd_sequences

        # Mark all asym_ids as valid initially
        for asym_id in entity.subchains:
            valid_asym_ids.add(asym_id)

    # ==================================================
    # Identify covalent bonded subchains
    # ==================================================
    # NOTE: gemmi Connection uses auth_chain_id instead of asym_id (subchain ID).
    # * single auth_chain_id can map to multiple asym_ids
    residue_map: dict[tuple[AuthId, str, int | None], gemmi.Residue] = {}
    for chain in raw_struct[0]:
        chain: gemmi.Chain  # This can include multiple subchains
        auth_id: AuthId = chain.name
        for residue in chain:
            residue: gemmi.Residue
            if residue.subchain in valid_asym_ids:
                seq_id: gemmi.SeqId = residue.seqid
                residue_map[(auth_id, seq_id.icode, seq_id.num)] = residue

    # Find covalent bonded entities
    # This is used to identify covalent inhibitors:
    #   NonPolymer entities with covalent bonds to other entities
    linked_asym_ids: set[AsymId] = set()
    linked_bonds: list[tuple[gemmi.Connection, gemmi.Residue, gemmi.Residue]] = []
    for connect in raw_struct.connections:
        connect: gemmi.Connection
        if connect.type != gemmi.ConnectionType.Covale:
            continue
        p1: gemmi.AtomAddress = connect.partner1
        p2: gemmi.AtomAddress = connect.partner2
        k1 = (p1.chain_name, p1.res_id.seqid.icode, p1.res_id.seqid.num)
        k2 = (p2.chain_name, p2.res_id.seqid.icode, p2.res_id.seqid.num)
        res1: gemmi.Residue | None = residue_map.get(k1)
        res2: gemmi.Residue | None = residue_map.get(k2)

        # Only consider bonds where both residues are in valid chains
        if res1 is not None and res2 is not None:
            linked_asym_ids.update({res1.subchain, res2.subchain})
            linked_bonds.append((connect, res1, res2))
            continue

        # Remove standard-alone covalent ligand/glycan chains
        for res in (res1, res2):
            if res is not None:
                asym_id: AsymId = res.subchain
                entity_id: EntityId = asym_id_to_entity_id[asym_id]
                ctype: C.ChainType = entity_id_to_chain_type[entity_id]
                if ctype.is_nonpolymer:
                    valid_asym_ids.discard(asym_id)  # use discard to avoid KeyError

    del residue_map  # free memory

    # Remove branched chains that are not linked
    for entity in valid_entities:
        entity_id: EntityId = int(entity.name)
        if entity.entity_type == gemmi.EntityType.Branched:
            for asym_id in entity.subchains:
                if asym_id not in linked_asym_ids:
                    valid_asym_ids.discard(asym_id)

    # ==================================================
    # Construct chain structs
    # ==================================================
    chain_structs: list[structure.Chain] = []
    for entity in valid_entities:
        entity: gemmi.Entity
        entity_id: EntityId = int(entity.name)
        ctype: C.ChainType = entity_id_to_chain_type[entity_id]
        ccd_sequences: list[str] = entity_id_to_seq[entity_id]

        # determine whether to drop leaving atoms
        drop_leaving_atoms = True
        if entity.entity_type == gemmi.EntityType.NonPolymer and len(ccd_sequences) == 1:
            # For single-residue non-polymers (ligands/ions), do not drop by default
            # If covalent inhibitor, drop later for specific subchains
            drop_leaving_atoms = False

        smiles: str | None = None
        if ctype is C.ChainType.LIGAND:
            if ccd_sequences[0].startswith("LIG"):
                # For custom ligands, ensure unique residue names
                # NOTE: "Boltz" saves all ligands as "LIG"
                if len(ccd_sequences) != 1:
                    raise ValueError("Multiple LIG residues not supported.")
                # TODO: Import smiles extraction from MMCIF...
                raise NotImplementedError(
                    "Custom ligand SMILES not supported in CIF parsing."
                )

        parsed_chain = prepare_ref_chain(
            chain_type=ctype,
            ccd_sequences=ccd_sequences,
            ccd=ccd,
            smiles=smiles,
            drop_leaving_atoms=drop_leaving_atoms,
        )
        for asym_id in entity.subchains:
            if asym_id not in valid_asym_ids:
                # Skip invalid chains
                continue

            if (
                entity.entity_type == gemmi.EntityType.NonPolymer
                and asym_id in linked_asym_ids
            ):
                # For covalent inhibitors, create a new chain struct without leaving atoms
                c = prepare_ref_chain(
                    chain_type=ctype,
                    entity_id=entity_id,
                    asym_id=asym_id_to_int[asym_id],
                    sym_id=asym_id_to_sym_id[asym_id],
                    ccd_sequences=ccd_sequences,
                    ccd=ccd,
                    drop_leaving_atoms=True,
                )
            else:
                # Otherwise, clone chain for each subchain
                c = parsed_chain.copy_with(
                    deepcopy=(sym_id > 1),  # deepcopy to avoid shared arrays
                    entity_id=entity_id,
                    asym_id=asym_id_to_int[asym_id],
                    sym_id=asym_id_to_sym_id[asym_id],
                )
            chain_structs.append(c)

    # ==================================================
    # Construct connection structs
    # ==================================================
    # Before constructing connections, make residue index mappings for linked bonds
    linked_residues: dict[ResKey, gemmi.Residue] = {}
    for _, res1, res2 in linked_bonds:
        for res in (res1, res2):
            linked_residues[get_residue_key(res)] = res

    linked_residue_to_index: dict[ResKey, int] = {}
    for subchain in raw_struct[0].subchains():
        subchain: gemmi.ResidueSpan
        asym_id: AsymId = subchain.subchain_id()
        if asym_id not in valid_asym_ids:
            # Skip invalid chains
            continue
        entity_id: EntityId = asym_id_to_entity_id[asym_id]
        ctype: C.ChainType = entity_id_to_chain_type[entity_id]
        for residue_index, residue in enumerate(subchain, start=1):
            residue: gemmi.Residue
            res_key = get_residue_key(residue)
            if res_key in linked_residues:
                if ctype.is_polymer:
                    # For polymer residues, use label_seq directly
                    linked_residue_to_index[res_key] = residue.label_seq
                else:
                    # For non-polymer residues, use 1-based index within the entity
                    linked_residue_to_index[res_key] = residue_index

    # Construct connections
    connections: list[structure.CovalentConnection] = []
    for connect, res1, res2 in linked_bonds:
        # Get asym_ids
        asym_id1: AsymId = res1.subchain
        asym_id2: AsymId = res2.subchain
        asym_id1_int: int = asym_id_to_int[asym_id1]
        asym_id2_int: int = asym_id_to_int[asym_id2]

        if asym_id1 not in valid_asym_ids or asym_id2 not in valid_asym_ids:
            # Skip connections involving invalid chains
            continue

        # Get residue index
        res_idx1: int = linked_residue_to_index[get_residue_key(res1)]
        res_idx2: int = linked_residue_to_index[get_residue_key(res2)]

        # Get atom names
        atom1: str = connect.partner1.atom_name
        atom2: str = connect.partner2.atom_name

        # Check atom existence
        is_atom1_found = False
        is_atom2_found = False
        for c in chain_structs:
            if c.asym_id == asym_id1_int:
                atom_idcs = c.residue.iter_residue_atoms(res_idx1)
                if atom1 in c.atom.name[atom_idcs].tolist():
                    is_atom1_found = True
            if c.asym_id == asym_id2_int:
                atom_idcs = c.residue.iter_residue_atoms(res_idx2)
                if atom2 in c.atom.name[atom_idcs].tolist():
                    is_atom2_found = True
        if not (is_atom1_found and is_atom2_found):
            continue

        connections.append(
            structure.CovalentConnection(
                asym_id=(asym_id1_int, asym_id2_int),
                residue_index=(res_idx1, res_idx2),
                atom_names=(atom1, atom2),
            )
        )

    # ==================================================
    # Add chain metadata
    # ==================================================
    for entity in raw_struct.entities:
        entity: gemmi.Entity
        entity_id: EntityId = int(entity.name)
        if entity_id not in valid_entity_ids:
            # Skip invalid entities
            continue
        length = len(entity_id_to_seq[entity_id])
        for asym_id in entity.subchains:
            if asym_id not in valid_asym_ids:
                # Skip invalid chains
                continue
            sym_id: SymId = asym_id_to_sym_id[asym_id]
            chain_meta = schema.ChainInfo(
                chain_name=asym_id,  # store asym_id as chain_name
                chain_type=entity_id_to_chain_type[entity_id],
                entity_id=entity_id,
                asym_id=asym_id_to_int[asym_id],
                sym_id=sym_id,
                num_residues=length,
            )
            metadata.chains.append(chain_meta)

    return structure.RefStructure(
        chains=chain_structs,
        connections=connections,
        metadata=metadata,
    )


def insert_chain_coordinates(
    ref_chain: structure.Chain,
    raw_chain: gemmi.ResidueSpan,
) -> None:
    """Insert Coordinates from raw gemmi ResidueSpan into reference chain."""
    ccd_sequence: list[str] = ref_chain.residue.name.tolist()
    atom_names: list[str] = ref_chain.atom.name.tolist()
    for res_i, res in enumerate(raw_chain):
        res: gemmi.Residue

        # Get residue index
        if ref_chain.ctype.is_polymer:
            residue_index: int = res.label_seq
            if residue_index is None:
                continue
        else:
            # For non-polymer residues, use 1-based index within the entity
            # assert res.label_seq is None
            residue_index: int = res_i + 1

        if residue_index < 1 or residue_index > len(ccd_sequence):
            # Skip invalid residue indices
            logger.debug(
                f"Residue index {residue_index} out of bounds for chain with length "
                f"{len(ccd_sequence)}."
            )
            continue

        # Get atoms.
        name_to_atom: dict[str, gemmi.Atom] = {a.name.upper(): a for a in res}

        # Map MSE to MET, put the selenium atom in the sulphur column
        res_name = ccd_sequence[residue_index - 1]
        if res_name == "MET" and "SE" in name_to_atom:
            # WARN: in parse_ref_chain(), MSE is already converted to MET.
            # Therefore, I place this statements outside of the res_name check.
            name_to_atom["SD"] = name_to_atom["SE"]

        for atom_i in ref_chain.residue.iter_residue_atoms(residue_index):
            n: str = atom_names[atom_i]
            if n in name_to_atom:
                atom: gemmi.Atom = name_to_atom[n]
                coords: gemmi.Position = atom.pos
                ref_chain.atom.label_coords[atom_i, 0] = coords.x
                ref_chain.atom.label_coords[atom_i, 1] = coords.y
                ref_chain.atom.label_coords[atom_i, 2] = coords.z
                ref_chain.atom.bfactor[atom_i] = atom.b_iso
                name_to_atom.pop(n)
            else:
                # Leave as NaN if atom not found
                logger.debug(f"Atom {n} not found in residue {res_name} {residue_index}.")
                continue
        for leftover_atom in name_to_atom.keys():
            if leftover_atom == "OXT":
                # Ignore loging for missing OXT atoms
                continue
            logger.debug(
                f"Atom {leftover_atom} in residue {res_name} {residue_index} not mapped."
            )
    # Update resolved flags
    ref_chain.atom.is_resolved[:] = np.isfinite(ref_chain.atom.label_coords).all(axis=-1)


def insert_coordinates(
    ref_struct: structure.RefStructure,
    raw_struct: gemmi.Structure,
    metadata: schema.Metadata,
) -> None:
    """Insert coordinates from raw gemmi Structure into reference structure."""
    # Build mapping from string asym_id to integer asym_id
    asym_id_to_int: dict[AsymId, int] = {}
    asym_id_to_str: dict[int, AsymId] = {}
    for m in metadata.chains:
        assert m.chain_name not in asym_id_to_int, (
            f"Duplicate asym_id {m.chain_name} found in metadata."
        )
        asym_id_to_int[m.chain_name] = m.asym_id
        asym_id_to_str[m.asym_id] = m.chain_name

    asym_id_to_ref_chain: dict[AsymId, structure.Chain] = {}
    for ref_chain in ref_struct.chains:
        asym_id_to_ref_chain[asym_id_to_str[ref_chain.asym_id]] = ref_chain

    # Iterate over raw chains to insert coordinates
    for raw_chain in raw_struct[0].subchains():
        raw_chain: gemmi.ResidueSpan
        asym_id: AsymId = raw_chain.subchain_id()
        if asym_id not in asym_id_to_ref_chain:
            # Skip invalid chains
            continue
        ref_chain: structure.Chain = asym_id_to_ref_chain[asym_id]
        # Insert coordinates
        insert_chain_coordinates(ref_chain, raw_chain)


# ==================================================
# Validation and interface detection
# ==================================================
def get_chain_ref_atom_coordinates(chain: structure.Chain) -> np.ndarray:
    """Get reference atom coordinates for a chain."""
    if chain.ctype.is_nonpolymer:
        # Return all atom coordinates for non-polymer chains
        return chain.atom.label_coords
    else:
        match chain.ctype:
            case C.ChainType.PROTEIN:
                ref_atom_offset = 1  # CA
            case C.ChainType.RNA:
                ref_atom_offset = 11  # C1'
            case C.ChainType.DNA:
                ref_atom_offset = 10  # C1'
        ref_atom_indices = chain.residue.atom_starts + ref_atom_offset
        ref_coords = chain.atom.label_coords[ref_atom_indices]
        return ref_coords


def validate_chain_geometry(struct: structure.RefStructure) -> None:
    """Check if a polymer chain is valid."""
    metadata: schema.Metadata = struct.metadata
    for chain_i in range(struct.num_chains):
        ref_chain: structure.Chain = struct.chains[chain_i]
        chain_meta: schema.ChainInfo = metadata.chains[chain_i]
        ctype: C.ChainType = ref_chain.ctype

        # Get reference atom coordinates and resolved flags
        ref_atom_coords = get_chain_ref_atom_coordinates(ref_chain)
        is_resolved = np.isfinite(ref_atom_coords).all(axis=-1)
        n_resolved = np.sum(is_resolved)

        if ctype.is_polymer:
            # For polymer chains, skip too short chains
            if n_resolved < 4:
                logger.debug(
                    f"{metadata.id}: Chain {ref_chain.asym_id} marked invalid "
                    f"due to insufficient resolved residues ({n_resolved})."
                )
                chain_meta.is_valid = False
        else:
            # For non-polymer chains, only check if any atom is resolved
            if n_resolved == 0:
                logger.debug(
                    f"{metadata.id}: Chain {ref_chain.asym_id} marked invalid "
                    f"due to no resolved atoms."
                )
                chain_meta.is_valid = False

        # For protein chains, check CA trace continuity
        if ctype.is_protein:
            left = ref_atom_coords[:-1]
            right = ref_atom_coords[1:]
            dists = np.linalg.norm(left - right, axis=-1)
            if np.any(dists > 10.0):
                logger.debug(
                    f"{metadata.id}: Chain {ref_chain.asym_id} marked invalid "
                    f"due to CA trace discontinuity."
                )
                chain_meta.is_valid = False
                continue


def detect_interfaces_and_prune_clashes(
    struct: structure.RefStructure,
    remove_clashed: bool = True,
) -> None:
    """
    Detect valid interfaces between chains and prune chains with severe clashes.

    This function performs a hierarchical distance check:
    1. Coarse check using reference atoms (< 15.0 A)
    2. Fine check using all atoms (< 5.0 A)

    If 'remove_clashed' is True, chains with >30% clashing atoms (< 1.7 A)
    are marked as invalid in metadata.
    """
    metadata: schema.Metadata = struct.metadata
    interfaces: list[schema.InterfaceInfo] = []
    invalid_asym_ids: set[int] = set()

    # Collect coordinates
    ref_coords_dict: dict[int, np.ndarray] = {}
    all_coords_dict: dict[int, np.ndarray] = {}

    for chain in struct.chains:
        """Get reference atom indices for a chain."""
        all_coords = chain.atom.label_coords
        ref_coords = get_chain_ref_atom_coordinates(chain)
        # Only keep finite coordinates
        all_coords_dict[chain.asym_id] = all_coords[np.isfinite(all_coords).all(axis=-1)]
        ref_coords_dict[chain.asym_id] = ref_coords[np.isfinite(ref_coords).all(axis=-1)]

    for i1, i2 in itertools.combinations(range(struct.num_chains), 2):
        # Only consider valid chains
        chain1 = struct.chains[i1]
        chain2 = struct.chains[i2]
        chain_meta_1 = metadata.chains[i1]
        chain_meta_2 = metadata.chains[i2]
        asym_id1 = chain1.asym_id
        asym_id2 = chain2.asym_id

        if not chain_meta_1.is_valid or not chain_meta_2.is_valid:
            # Skip invalid chains
            continue
        if asym_id1 in invalid_asym_ids or asym_id2 in invalid_asym_ids:
            # Skip invalid chains
            continue

        # First, check for reference atom contacts (15 Angstrom cutoff)
        ref_coords1 = ref_coords_dict[asym_id1]  # [N, 3]
        ref_coords2 = ref_coords_dict[asym_id2]  # [M, 3]
        if ref_coords1.shape[0] == 0 or ref_coords2.shape[0] == 0:
            # No reference atoms, skip
            continue
        ref_dists = scipy.spatial.distance.cdist(ref_coords1, ref_coords2)  # [N, M]
        if not np.any(ref_dists < 15.0):
            # No contact detected
            continue
        del ref_dists

        # Then, check for all atom contacts (5 Angstrom cutoff)
        coords1 = all_coords_dict[asym_id1]  # [N, 3]
        coords2 = all_coords_dict[asym_id2]  # [M, 3]
        dists = scipy.spatial.distance.cdist(coords1, coords2)  # [N, M]
        if not np.any(dists < 5.0):
            # No contact detected
            continue

        if remove_clashed:
            # Check for clash
            is_clash = dists < 1.7
            is_clash_1 = np.any(is_clash, axis=1)
            is_clash_2 = np.any(is_clash, axis=0)
            clash_ratio_1 = np.sum(is_clash_1) / is_clash_1.shape[0]
            clash_ratio_2 = np.sum(is_clash_2) / is_clash_2.shape[0]
            is_clash_chain_1 = clash_ratio_1 > 0.3
            is_clash_chain_2 = clash_ratio_2 > 0.3

            if is_clash_chain_1 and is_clash_chain_2:
                # Both chains are severely clashed, remove one:
                #   - Remove the one with higher clash ratio
                #   - If equal, remove the larger one
                #   - If still equal, remove the one with larger asym_id
                if clash_ratio_1 > clash_ratio_2:
                    remove_asym_id = asym_id1
                elif clash_ratio_1 < clash_ratio_2:
                    remove_asym_id = asym_id2
                else:
                    if chain1.num_atoms > chain2.num_atoms:
                        remove_asym_id = asym_id2
                    elif chain1.num_atoms < chain2.num_atoms:
                        remove_asym_id = asym_id1
                    else:
                        remove_asym_id = max(asym_id1, asym_id2)
                logger.debug(
                    f"{metadata.id}: Chains {asym_id1} and {asym_id2} "
                    f"marked invalid due to severe clash "
                    f"({clash_ratio_1:.2%} vs {clash_ratio_2:.2%})."
                )
                invalid_asym_ids.add(remove_asym_id)
                continue
            elif is_clash_chain_1:
                logger.debug(
                    f"{metadata.id}: Chain {asym_id1} marked invalid due to clash "
                    f"({clash_ratio_1:.2%} atoms clashed with chain {asym_id2})."
                )
                invalid_asym_ids.add(asym_id1)
                continue
            elif is_clash_chain_2:
                logger.debug(
                    f"{metadata.id}: Chain {asym_id2} marked invalid due to clash "
                    f"({clash_ratio_2:.2%} atoms clashed with chain {asym_id1})."
                )
                invalid_asym_ids.add(asym_id2)
                continue

        # Valid interface
        interfaces.append(schema.InterfaceInfo(asym_ids=(asym_id1, asym_id2)))

    # Mark invalid chains
    for chain_meta in struct.metadata.chains:
        if chain_meta.asym_id in invalid_asym_ids:
            chain_meta.is_valid = False

    # Mark interfaces involving invalid chains as invalid
    for iface in interfaces:
        if iface.asym_ids[0] in invalid_asym_ids or iface.asym_ids[1] in invalid_asym_ids:
            iface.is_valid = False

    struct.metadata.interfaces = interfaces


def prune_invalid_chains(struct: structure.RefStructure) -> None:
    """Drop invalid chains from the structure."""
    metadata: schema.Metadata = struct.metadata

    # Get valid asym_ids
    valid_asym_ids: set[int] = set(m.asym_id for m in metadata.chains if m.is_valid)

    # Remove standard-alone branched ligands
    for con in struct.connections:
        asym_id1, asym_id2 = con.asym_id
        if asym_id1 in valid_asym_ids and asym_id2 in valid_asym_ids:
            continue
        # Check if either chain is branched
        chain_meta1 = metadata.get_chain_by_asym_id(asym_id1)
        chain_meta2 = metadata.get_chain_by_asym_id(asym_id2)
        if chain_meta1.chain_type is C.ChainType.LIGAND:
            valid_asym_ids.discard(asym_id1)
        if chain_meta2.chain_type is C.ChainType.LIGAND:
            valid_asym_ids.discard(asym_id2)

    metadata.chains = [m for m in metadata.chains if m.asym_id in valid_asym_ids]

    # Prune chains, connections, and interfaces
    struct.chains = [c for c in struct.chains if c.asym_id in valid_asym_ids]
    struct.connections = [
        con
        for con in struct.connections
        if con.asym_id[0] in valid_asym_ids and con.asym_id[1] in valid_asym_ids
    ]
    metadata.interfaces = [
        iface
        for iface in struct.metadata.interfaces
        if (
            iface.is_valid
            and iface.asym_ids[0] in valid_asym_ids
            and iface.asym_ids[1] in valid_asym_ids
        )
    ]


# ==================================================
# Substructure sampling
# ==================================================
def crop_substructure(
    struct: structure.RefStructure,
    max_chains: int = 20,
    seed: int = 42,
):
    """Sample a substructure with at most max_chains chains."""
    if struct.num_chains <= max_chains:
        return struct

    # This sampling procedure should be conducted after pruning
    assert all(m.is_valid for m in struct.metadata.chains) and (
        all(iface.is_valid for iface in struct.metadata.interfaces)
    ), "Structure must be pruned before sampling."

    # Collect reference coordinates
    ref_coords_dict: dict[int, np.ndarray] = {}

    for chain in struct.chains:
        """Get reference atom indices for a chain."""
        ref_coords = get_chain_ref_atom_coordinates(chain)
        # Only keep finite coordinates
        ref_coords = ref_coords[np.isfinite(ref_coords).all(axis=-1)]
        ref_coords_dict[chain.asym_id] = ref_coords

    # Sample random interfaces
    rng = np.random.default_rng(seed)
    iface_i = rng.integers(len(struct.metadata.interfaces))
    sampled_iface = struct.metadata.interfaces[iface_i]
    asym_id1, asym_id2 = sampled_iface.asym_ids

    # Get interface atom
    ref_coords1 = ref_coords_dict[asym_id1]  # [N1, 3]
    ref_coords2 = ref_coords_dict[asym_id2]  # [N2, 3]
    dists = scipy.spatial.distance.cdist(ref_coords1, ref_coords2)
    is_contact = dists < 15.0

    if np.any(is_contact):
        if rng.random() < 0.5:
            # Start from chain 1
            is_contact_1 = np.any(is_contact, axis=1)
            contact_indices_1 = np.where(is_contact_1)[0]
            seed_atom = rng.choice(contact_indices_1)
            seed_coords = ref_coords1[seed_atom, :].reshape(1, 3)
        else:
            # Start from chain 2
            is_contact_2 = np.any(is_contact, axis=0)
            contact_indices_2 = np.where(is_contact_2)[0]
            seed_atom = rng.choice(contact_indices_2)
            seed_coords = ref_coords2[seed_atom, :].reshape(1, 3)
    else:
        # Fallback: random atom from either chain
        if rng.random() < 0.5:
            seed_atom = rng.integers(ref_coords1.shape[0])
            seed_coords = ref_coords1[seed_atom, :].reshape(1, 3)
        else:
            seed_atom = rng.integers(ref_coords2.shape[0])
            seed_coords = ref_coords2[seed_atom, :].reshape(1, 3)

    # Collect closest chains until reaching max_chains
    chain_dists: dict[int, float] = {}
    for chain in struct.chains:
        coords = ref_coords_dict[chain.asym_id]
        dists = np.linalg.norm(coords - seed_coords, axis=-1)
        min_dist = np.min(dists)
        chain_dists[chain.asym_id] = min_dist

    sorted_chains = sorted(
        chain_dists.items(),
        key=lambda x: x[1],
    )  # list of (asym_id, dist)
    selected_asym_ids = set(asym_id for asym_id, _ in sorted_chains[:max_chains])

    # Filter chains
    struct.chains = [
        chain for chain in struct.chains if chain.asym_id in selected_asym_ids
    ]
    struct.metadata.chains = [
        m for m in struct.metadata.chains if m.asym_id in selected_asym_ids
    ]
    # Filter connections
    struct.connections = [
        c
        for c in struct.connections
        if c.asym_id[0] in selected_asym_ids and c.asym_id[1] in selected_asym_ids
    ]
    # Filter interfaces
    struct.metadata.interfaces = [
        iface
        for iface in struct.metadata.interfaces
        if (
            iface.is_valid
            and iface.asym_ids[0] in selected_asym_ids
            and iface.asym_ids[1] in selected_asym_ids
        )
    ]
