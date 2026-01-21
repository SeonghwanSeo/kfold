"""Pipeline to prepare reference structures
# TODO: add PDB parsing later
"""

import itertools
import logging
import pathlib
from datetime import datetime
from typing import Any

import gemmi
import numpy as np
import scipy.spatial.distance

import kfold.constants as C
from kfold.data.pipelines import structure_preparation
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import (
    ExperimentRecord,
    InterfaceInfo,
    Metadata,
    PredictionRecord,
)
from kfold.data.types.structure import (
    Chain,
    CovalentConnection,
    RefStructure,
)

logger = logging.getLogger(__name__)

# Type aliases for better readability
LabelId = str  # Label chain ID (asym_id) from mmCIF and gemmi (A1 B1 A2 ...)
AuthId = str  # Author chain ID
EntityId = int  # Entity ID from mmCIF (1, 2, 3 ...)
AsymId = int  # Integer asym_id mapping (1, 2, 3 ...)
SymId = int  # Symmetry chain ID from mmCIF (1, 2, 3 ...)
ResKey = tuple[LabelId, str, int | None]

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
# one-letter codes
chain_type_to_unk: dict[C.ChainType, str] = {
    C.ChainType.PROTEIN: "UNK",
    C.ChainType.RNA: "N",
    C.ChainType.DNA: "DN",
}


# ==================================================
# Template function for CIF parsing
# ==================================================
def parse_cif(
    cif_path: str | pathlib.Path,
    ccd: CCD,
    max_chains: int | None = None,
) -> RefStructure:
    """Parse a CIF file and return a gemmi.cif.Document object.
    NOTE: This is just a template function. Additional filtering can be
    inserted as needed, e.g., date cutoff, number of chains, etc.
    """
    # Read CIF file
    doc: gemmi.cif.Document = gemmi.cif.read_file(str(cif_path))
    block: gemmi.cif.Block = doc[0]

    # Get metadata (without chain infos)
    # Handle cases like "1abc.cif.gz"
    name = pathlib.Path(cif_path).name.split(".")[0]
    metadata = prepare_metadata_from_rcsb(name, block)

    # --- Gemmi structure processing ---

    # Prepare gemmi structure
    raw_struct: gemmi.Structure = prepare_gemmi_structure(
        block, expand_assembly=True, clean_up=True
    )

    # --- Reference structure preparation ---

    # Prepare reference structure and add chain metadata
    ref_struct = prepare_ref_structure(raw_struct, metadata, ccd)

    # Insert coordinates
    insert_coordinates(ref_struct, raw_struct, metadata)

    # --- Cleaning and interface detection ---
    invalid_chains: set[int] = set()

    # Validate chain geometry
    validate_chain_geometry(ref_struct, invalid_chains)

    # Get interfaces and those metadata, detect clashes
    detect_interfaces_and_detect_clashes(ref_struct, invalid_chains)

    # Drop invalid chains
    prune_invalid_chains(ref_struct, invalid_chains)

    # Crop to max_chains if specified
    if max_chains is not None:
        crop_substructure(ref_struct, max_chains)

    # Final validation
    ref_struct.validate()

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


def prepare_experiment_record(block: gemmi.cif.Block) -> ExperimentRecord:
    """Parse RCSB PDB metadata from CIF block."""
    # PDB ID
    pdb_id = get_first_value(block, "_entry.id")
    assert pdb_id is not None, "PDB ID is missing in metadata."

    # Release Date
    rev_dates = block.find_values("_pdbx_audit_revision_history.revision_date")
    release_date = min(rev_dates) if rev_dates else None
    assert release_date is not None, "Release date is missing in metadata."

    # Method (e.g., X-RAY DIFFRACTION)
    method = get_first_value(block, "_exptl.method")
    assert method is not None, "Experimental method is missing in metadata."
    method = method.replace("'", "").replace('"', "").upper()  # clean quotes
    if method not in C.training.ALL_EXPERIMENT_METHODS:
        method = "OTHER"

    # Resolution (Handle X-ray vs EM vs NMR)
    # NMR: no resolution (None)
    # X-ray standard
    resolution = get_first_value(block, "_refine.ls_d_res_high", float)
    if not resolution:
        # EM
        resolution = get_first_value(block, "_em_3d_reconstruction.resolution", float)
    if not resolution:
        # Fallback
        resolution = get_first_value(block, "_reflns.d_resolution_high", float)

    # Temperature (Kelvin)
    temp = get_first_value(block, "_diffrn.ambient_temp", float)

    # pH (Crystallization condition)
    ph = get_first_value(block, "_exptl_crystal_grow.pH", float)

    return ExperimentRecord(
        pdb_id=pdb_id,
        release_date=release_date,
        method=method,
        resolution=resolution,
        pH=ph,
        temperature=temp,
    )


def prepare_metadata_from_rcsb(
    name: str,
    block: gemmi.cif.Block,
) -> Metadata:
    """Parse metadata from CIF block."""
    exp_record = prepare_experiment_record(block)
    return Metadata(
        id=name,
        source="rcsb",
        exp=exp_record,
        chains=[],  # Filled later in parsing
        interfaces=[],  # Filled later in parsing
    )


def prepare_metadata_from_synthetic_data(
    name: str,
    block: gemmi.cif.Block,
    model: str,
) -> Metadata:
    """Parse metadata from CIF block."""
    # Parse experiment record
    prediction = PredictionRecord(model=model)
    return Metadata(
        id=name,
        source="prediction",
        prediction=prediction,
        chains=[],  # Filled later in parsing
        interfaces=[],  # Filled later in parsing
    )


# =================================================
# Helper functions for metadata filtering
# ==================================================
def check_resolution_cutoff(
    metadata: Metadata,
    max_resolution: float,
    skip_nmr: bool = True,
) -> bool:
    """Returns True if the entry passes the resolution filter."""
    assert metadata.exp is not None
    if skip_nmr and metadata.exp.is_nmr_structure:
        # NMR does not have resolution, always pass
        return True
    resolution = metadata.exp.resolution
    return resolution is not None and resolution <= max_resolution


def check_date_cutoff(
    metadata: Metadata,
    date_start: datetime = datetime.min,
    date_end: datetime = datetime.max,
) -> bool:
    """Returns True if the entry passes the date filter."""
    assert metadata.exp is not None
    release_date: datetime = datetime.fromisoformat(metadata.exp.release_date)
    return date_start <= release_date <= date_end


def check_method(
    metadata: Metadata,
    exclude_methods: set[str] = set(),
) -> bool:
    """Returns True if the entry passes the experimental method filter."""
    assert metadata.exp is not None
    return metadata.exp.method not in exclude_methods


# ==================================================
# Helper functions for gemmi structure validation
# ==================================================
def prepare_gemmi_structure(
    block: gemmi.cif.Block,
    expand_assembly: bool = True,
    clean_up: bool = True,
) -> gemmi.Structure:
    """Prepare gemmi Structure object from CIF block."""
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    if expand_assembly:
        expand_first_assembly(raw_struct)
    if clean_up:
        clean_up_gemmi_structure(
            raw_struct, map_mse_to_met=True, canonicalize_arginines=True
        )
    return raw_struct


def expand_first_assembly(raw_struct: gemmi.Structure) -> None:
    """Expand the first assembly in the gemmi Structure object in-place.

    See AlphaFold3 Section 2.1 Parsing.
    """
    if len(raw_struct.assemblies) > 0:
        how = gemmi.HowToNameCopiedChain.AddNumber
        try:
            raw_struct.transform_to_assembly(raw_struct.assemblies[0].name, how=how)
        except Exception as e:
            logger.warning(f"Failed to expand assembly: {e}")


def clean_up_gemmi_structure(
    raw_struct: gemmi.Structure,
    map_mse_to_met: bool = True,
    canonicalize_arginines: bool = True,
) -> None:
    """Clean up gemmi Structure object in-place.

    See AlphaFold3 Section 2.1 Parsing.
    """
    raw_struct.merge_chain_parts()
    raw_struct.remove_alternative_conformations()
    raw_struct.remove_hydrogens()
    raw_struct.remove_waters()
    raw_struct.remove_empty_chains()

    protein_entity_ids: set[EntityId] = set()
    protein_chains: set[LabelId] = set()
    for entity in raw_struct.entities:
        # In the case of microheterogeneity, take the first monomer
        entity.full_sequence = [
            gemmi.Entity.first_mon(res) for res in entity.full_sequence
        ]
        if (
            entity.entity_type == gemmi.EntityType.Polymer
            and entity.polymer_type == gemmi.PolymerType.PeptideL
        ):
            # Collect protein asym_ids
            protein_entity_ids.add(int(entity.name))
            protein_chains.update(entity.subchains)

            if map_mse_to_met:
                # Map MSE to MET
                if entity.name in protein_entity_ids:
                    entity.full_sequence = [
                        "MET" if res == "MSE" else res for res in entity.full_sequence
                    ]

    model: gemmi.Model = raw_struct[0]
    for res_span in model.subchains():
        label_id: LabelId = res_span.subchain_id()
        if label_id in protein_chains:
            for residue in res_span:
                if map_mse_to_met and residue.name == "MSE":
                    # Map MSE to MET
                    residue.name = "MET"
                    for atom in residue:
                        if atom.name == "SE":
                            atom.name = "SD"
                            atom.element = gemmi.Element("S")
                if canonicalize_arginines and residue.name == "ARG":
                    # Ensure arginine NH1/NH2 naming is canonical
                    try:
                        cd: gemmi.Atom = residue["CD"][0]
                        nh1: gemmi.Atom = residue["NH1"][0]
                        nh2: gemmi.Atom = residue["NH2"][0]
                    except Exception:
                        continue
                    # Calculate distances
                    dist_cd_nh1 = cd.pos.dist(nh1.pos)
                    dist_cd_nh2 = cd.pos.dist(nh2.pos)
                    # Swap if NH2 is closer to CD
                    if dist_cd_nh2 < dist_cd_nh1:
                        # Swap names
                        nh1.name, nh2.name = "NH2", "NH1"


def get_residue_key(residue: gemmi.Residue) -> ResKey:
    label_id: LabelId = residue.subchain
    seq_id: gemmi.SeqId = residue.seqid
    return (label_id, seq_id.icode, seq_id.num)


# ==================================================
# Main functions for reference structure preparation
# ==================================================
def prepare_ref_structure(
    raw_struct: gemmi.Structure,
    metadata: Metadata,
    ccd: CCD,
) -> RefStructure:
    """Prepare reference structure from gemmi CIF block and metadata."""

    # Determine ligand CCDs to exclude
    excluded_ligands: set[str] = C.ccd.LIGAND_EXCLUSIONS
    if metadata.exp is not None:
        # Exclude crystallization aids for crystal structures
        if metadata.exp.is_crystal_structure:
            excluded_ligands = excluded_ligands | C.ccd.CRYSTALLIZATION_AIDS

    # ==================================================
    # Prepare chain index mappings
    # ==================================================
    label_id_to_entity_id: dict[LabelId, EntityId] = {}
    label_id_to_sym_id: dict[LabelId, SymId] = {}

    # Determine asym_id mappings (label_id: str, asym_id: int)
    label_id_to_asym_id: dict[LabelId, AsymId] = {}
    asym_id_counter = itertools.count(start=1)
    for entity in raw_struct.entities:
        entity: gemmi.Entity
        assert entity.name.isdecimal(), "Entity name is not an integer."
        entity_id: EntityId = int(entity.name)
        assert entity_id > 0, "Entity ID should be positive integer."
        for sym_id, label_id in enumerate(entity.subchains, start=1):
            assert label_id not in label_id_to_entity_id, (
                f"Duplicate asym_id {label_id} found in entities."
            )
            asym_id: AsymId = next(asym_id_counter)
            label_id_to_entity_id[label_id] = int(entity_id)
            label_id_to_sym_id[label_id] = sym_id
            label_id_to_asym_id[label_id] = asym_id
    del asym_id_counter  # free memory

    # ==================================================
    # Identify valid entities and chains
    # ==================================================
    valid_entities: list[gemmi.Entity] = []
    valid_label_ids: set[LabelId] = set()
    entity_id_to_entity: dict[EntityId, gemmi.Entity] = {}
    entity_id_to_ctype: dict[EntityId, C.ChainType] = {}
    entity_id_to_seq: dict[EntityId, list[str]] = {}
    for entity in raw_struct.entities:
        entity: gemmi.Entity
        entity_id: EntityId = int(entity.name)

        if len(entity.subchains) == 0:
            # Skip entities without subchains
            continue

        if entity.entity_type == gemmi.EntityType.Polymer:
            # Protein, RNA, or DNA
            # TODO: Do we have to consider more polymer types? e.g., DNA/RNA hybrids
            if entity.polymer_type not in polymer_type_to_chain_type:
                # Skip unsupported polymer types
                continue
            ctype: C.ChainType = polymer_type_to_chain_type[entity.polymer_type]
            unk: str = chain_type_to_unk[ctype]

            # Get CCD sequences with unknown mapping
            ccd_sequences: list[str] = entity.full_sequence
            ccd_sequences: list[str] = [
                v if (v in C.ccd.CCD_NAME_TO_ONE_LETTER and v in ccd) else unk
                for v in ccd_sequences
            ]

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
            chain_type = C.ChainType.LIGAND
            ref_label_id: LabelId = entity.subchains[0]
            raw_chain: gemmi.ResidueSpan = raw_struct[0].get_subchain(ref_label_id)
            ccd_sequences: list[str] = [res.name for res in raw_chain]

            if len(ccd_sequences) == 0:
                # Skip empty ligand entities
                continue

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
        else:
            # Skip other entity types
            continue

        # Store entity
        valid_entities.append(entity)
        entity_id_to_entity[entity_id] = entity
        entity_id_to_ctype[entity_id] = chain_type
        entity_id_to_seq[entity_id] = ccd_sequences

        # Mark all label_ids as valid initially
        for label_id in entity.subchains:
            valid_label_ids.add(label_id)

    # ==================================================
    # Identify covalent bonded subchains
    # ==================================================
    # NOTE: gemmi Connection uses auth_asym_id instead of label_asym_id
    # * single auth_asym_id can map to multiple label_asym_id (e.g., covalent inhibitor)
    residue_map: dict[tuple[AuthId, str, int | None], gemmi.Residue] = {}
    for chain in raw_struct[0]:
        chain: gemmi.Chain  # This can include multiple subchains
        auth_id: AuthId = chain.name
        for residue in chain:
            residue: gemmi.Residue
            if residue.subchain in valid_label_ids:
                seq_id: gemmi.SeqId = residue.seqid
                residue_map[(auth_id, seq_id.icode, seq_id.num)] = residue

    # Find covalent bonded entities
    # This is used to identify covalent inhibitors:
    #   NonPolymer entities with covalent bonds to other entities
    linked_label_ids: set[LabelId] = set()
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

        if res1 is None or res2 is None:
            # Remove orphaned glycans and covalent inhibitors
            for res in [r for r in (res1, res2) if r is not None]:
                if res.entity_type != gemmi.EntityType.Polymer:
                    valid_label_ids.discard(res.subchain)
            continue

        # Only consider bonds where both residues are in valid chains
        linked_label_ids.update({res1.subchain, res2.subchain})
        linked_bonds.append((connect, res1, res2))

    del residue_map  # free memory

    # ==================================================
    # Construct chain structs
    # ==================================================
    chain_structs: list[Chain] = []
    for entity in valid_entities:
        entity: gemmi.Entity
        entity_id: EntityId = int(entity.name)
        ctype: C.ChainType = entity_id_to_ctype[entity_id]
        ccd_sequences: list[str] = entity_id_to_seq[entity_id]

        # For ligand, identify smiles if available
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

        # Parse reference chain once
        # Drop leaving atoms for polymers, and keep all atoms for ligands as default
        # (Will filter later based on covalent bonds)
        parsed_chain: Chain = structure_preparation.prepare_ref_chain(
            chain_type=ctype,
            ccd_sequences=ccd_sequences,
            ccd=ccd,
            smiles=smiles,
            drop_leaving_atoms=ctype.is_polymer,
        )
        if ctype.is_nonpolymer:
            # Collect residue atom names for non-polymer chains
            ref_residue_atom_names: dict[int, set[str]] = {}
            for res_idx in range(1, parsed_chain.num_residues + 1):
                atom_st = parsed_chain.residue.atom_starts[res_idx - 1]
                atom_en = atom_st + parsed_chain.residue.num_atoms[res_idx - 1]
                ref_residue_atom_names[res_idx] = set(
                    parsed_chain.atom.name[atom_st:atom_en].tolist()
                )

        for label_id in entity.subchains:
            if label_id not in valid_label_ids:
                # Skip invalid chains
                continue

            valid_atom_names: dict[int, set[str]] = {}
            if ctype.is_nonpolymer and label_id in linked_label_ids:
                # For non-polymer chains with covalent bonds, check valid atoms.
                for i, res in enumerate(raw_struct[0].get_subchain(label_id), start=1):
                    atom_names = set(atom.name.upper() for atom in res)
                    if atom_names != ref_residue_atom_names[i]:
                        valid_atom_names[i] = atom_names

            if len(valid_atom_names) > 0:
                c = structure_preparation.prepare_ref_chain(
                    chain_type=ctype,
                    entity_id=entity_id,
                    asym_id=label_id_to_asym_id[label_id],
                    sym_id=label_id_to_sym_id[label_id],
                    ccd_sequences=ccd_sequences,
                    ccd=ccd,
                    valid_atom_names=valid_atom_names,
                )
            else:
                # Otherwise, clone chain for each subchain
                c = parsed_chain.copy_with(
                    deepcopy=(sym_id > 1),  # deepcopy to avoid shared arrays
                    entity_id=entity_id,
                    asym_id=label_id_to_asym_id[label_id],
                    sym_id=label_id_to_sym_id[label_id],
                )
            chain_structs.append(c)
    asym_id_to_chain: dict[int, Chain] = {c.asym_id: c for c in chain_structs}

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
        label_id: LabelId = subchain.subchain_id()
        if label_id not in valid_label_ids:
            # Skip invalid chains
            continue
        entity_id: EntityId = label_id_to_entity_id[label_id]
        ctype: C.ChainType = entity_id_to_ctype[entity_id]
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
    connections: list[CovalentConnection] = []
    for connect, res1, res2 in linked_bonds:
        # Get label_ids
        label_id1: LabelId = res1.subchain
        label_id2: LabelId = res2.subchain
        if label_id1 not in valid_label_ids or label_id2 not in valid_label_ids:
            # Skip connections involving invalid chains
            continue
        asym_id1: AsymId = label_id_to_asym_id[label_id1]
        asym_id2: AsymId = label_id_to_asym_id[label_id2]
        c1: Chain = asym_id_to_chain[asym_id1]
        c2: Chain = asym_id_to_chain[asym_id2]

        if c1.ctype.is_polymer and c2.ctype.is_polymer:
            # Skip polymer-polymer connections
            continue

        # Get residue index (1-based)
        res_idx1: int = linked_residue_to_index[get_residue_key(res1)]
        res_idx2: int = linked_residue_to_index[get_residue_key(res2)]
        res_i1, res_i2 = res_idx1 - 1, res_idx2 - 1  # 0-based indices

        # Get atom names
        atom1: str = connect.partner1.atom_name
        atom2: str = connect.partner2.atom_name

        # Check atom existence
        atom_st = c1.residue.atom_starts[res_i1]
        atom_en = atom_st + c1.residue.num_atoms[res_i1]
        valid_atoms1 = c1.atom.name[atom_st:atom_en]
        is_atom1_found = atom1 in valid_atoms1

        atom_st = c2.residue.atom_starts[res_i2]
        atom_en = atom_st + c2.residue.num_atoms[res_i2]
        valid_atoms2 = c2.atom.name[atom_st:atom_en]
        is_atom2_found = atom2 in valid_atoms2

        if not (is_atom1_found and is_atom2_found):
            logger.warning(
                f"Skipping connection: atoms not found "
                f"{label_id1}:{res_idx1}:{atom1} - {label_id2}:{res_idx2}:{atom2}.\n"
                f"Valid atoms in {label_id1}:{res_idx1}: {valid_atoms1.tolist()}\n"
                f"Valid atoms in {label_id2}:{res_idx2}: {valid_atoms2.tolist()}"
            )
            continue

        connections.append(
            CovalentConnection(
                asym_id=(asym_id1, asym_id2),
                residue_index=(res_idx1, res_idx2),
                atom_names=(atom1, atom2),
            )
        )

    # ==================================================
    # Add chain metadata
    # ==================================================
    asym_id_to_label_id: dict[AsymId, LabelId] = {
        v: k for k, v in label_id_to_asym_id.items()
    }
    for c in chain_structs:
        chain_info = structure_preparation.prepare_chain_metadata(
            c, name=asym_id_to_label_id[c.asym_id]
        )
        metadata.chains.append(chain_info)

    return RefStructure(
        chains=chain_structs,
        connections=connections,
        metadata=metadata,
    )


def insert_chain_coordinates(
    ref_chain: Chain,
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
        res_name = ccd_sequence[residue_index - 1]
        for atom_i in ref_chain.residue.iter_residue_atoms(residue_index):
            n: str = atom_names[atom_i]
            if n in name_to_atom:
                atom: gemmi.Atom = name_to_atom[n]
                coords: gemmi.Position = atom.pos
                ref_chain.atom.coords[atom_i, 0] = coords.x
                ref_chain.atom.coords[atom_i, 1] = coords.y
                ref_chain.atom.coords[atom_i, 2] = coords.z
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


def insert_coordinates(
    ref_struct: RefStructure,
    raw_struct: gemmi.Structure,
    metadata: Metadata,
) -> None:
    """Insert coordinates from raw gemmi Structure into reference structure."""
    # Build mapping from string asym_id to integer asym_id
    label_id_to_asym_id: dict[LabelId, int] = {}
    asym_id_to_label_id: dict[int, LabelId] = {}
    for m in metadata.chains:
        assert m.name not in label_id_to_asym_id, (
            f"Duplicate asym_id {m.name} found in metadata."
        )
        label_id_to_asym_id[m.name] = m.asym_id
        asym_id_to_label_id[m.asym_id] = m.name

    label_id_to_ref_chain: dict[LabelId, Chain] = {}
    for ref_chain in ref_struct.chains:
        label_id_to_ref_chain[asym_id_to_label_id[ref_chain.asym_id]] = ref_chain

    # Iterate over raw chains to insert coordinates
    for raw_chain in raw_struct[0].subchains():
        raw_chain: gemmi.ResidueSpan
        label_id: LabelId = raw_chain.subchain_id()
        if label_id not in label_id_to_ref_chain:
            # Skip invalid chains
            continue
        ref_chain: Chain = label_id_to_ref_chain[label_id]
        # Insert coordinates
        insert_chain_coordinates(ref_chain, raw_chain)


# ==================================================
# Validation and interface detection
# ==================================================
def get_chain_ref_atom_coordinates(chain: Chain) -> np.ndarray:
    """Get reference atom coordinates for a chain."""
    if chain.ctype.is_nonpolymer:
        # Return all atom coordinates for non-polymer chains
        return chain.atom.coords
    else:
        match chain.ctype:
            case C.ChainType.PROTEIN:
                ref_atom_name = "CA"
            case C.ChainType.RNA:
                ref_atom_name = "C1'"
            case C.ChainType.DNA:
                ref_atom_name = "C1'"
        ref_idx = chain.atom.name == ref_atom_name
        ref_coords = chain.atom.coords[ref_idx]
        return ref_coords


def validate_chain_geometry(
    struct: RefStructure,
    invalid_chains: set[int],
):
    """Check if a polymer chain is valid."""
    for chain_i in range(struct.num_chains):
        ref_chain: Chain = struct.chains[chain_i]
        ctype: C.ChainType = ref_chain.ctype

        # Get reference atom coordinates and resolved flags
        ref_atom_coords = get_chain_ref_atom_coordinates(ref_chain)
        is_resolved = np.isfinite(ref_atom_coords).all(axis=-1)
        n_resolved = np.sum(is_resolved)

        if ctype.is_polymer:
            # For polymer chains, skip too short chains
            if n_resolved < 4:
                logger.debug(
                    f"{struct.id}: Chain {ref_chain.asym_id} marked invalid "
                    f"due to insufficient resolved residues ({n_resolved})."
                )
                invalid_chains.add(ref_chain.asym_id)
        else:
            # For non-polymer chains, only check if any atom is resolved
            if n_resolved == 0:
                logger.debug(
                    f"{struct.id}: Chain {ref_chain.asym_id} marked invalid "
                    f"due to no resolved atoms."
                )
                invalid_chains.add(ref_chain.asym_id)

        # For protein chains, check CA trace continuity
        if ctype.is_protein:
            left = ref_atom_coords[:-1]
            right = ref_atom_coords[1:]
            dists = np.linalg.norm(left - right, axis=-1)
            if np.any(dists > 10.0):
                logger.debug(
                    f"{struct.id}: Chain {ref_chain.asym_id} marked invalid "
                    f"due to CA trace discontinuity."
                )
                invalid_chains.add(ref_chain.asym_id)
                continue


def detect_interfaces_and_detect_clashes(
    struct: RefStructure,
    invalid_chains: set[int],
    clash_distance_cutoff: float = 1.7,
):
    """
    Detect valid interfaces between chains and prune chains with severe clashes.

    This function performs a hierarchical distance check:
    1. Coarse check using reference atoms (< 15.0 A)
    2. Fine check using all atoms (< 5.0 A)
    """

    def is_valid(chain: Chain) -> bool:
        return chain.asym_id not in invalid_chains

    metadata: Metadata = struct.metadata
    interfaces: list[InterfaceInfo] = []

    # Collect coordinates
    ref_coords_dict: dict[int, np.ndarray] = {}
    all_coords_dict: dict[int, np.ndarray] = {}

    for chain in struct.chains:
        """Get reference atom indices for a chain."""
        all_coords = chain.atom.coords
        ref_coords = get_chain_ref_atom_coordinates(chain)
        # Only keep finite coordinates
        all_coords_dict[chain.asym_id] = all_coords[np.isfinite(all_coords).all(axis=-1)]
        ref_coords_dict[chain.asym_id] = ref_coords[np.isfinite(ref_coords).all(axis=-1)]

    for i1, i2 in itertools.combinations(range(struct.num_chains), 2):
        # Only consider valid chains
        chain1 = struct.chains[i1]
        chain2 = struct.chains[i2]
        asym_id1 = chain1.asym_id
        asym_id2 = chain2.asym_id

        if not is_valid(chain1) or not is_valid(chain2):
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

        # Check for clash
        is_clash = dists < clash_distance_cutoff  # [N, M]
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
            invalid_chains.add(remove_asym_id)
            continue
        elif is_clash_chain_1:
            logger.debug(
                f"{metadata.id}: Chain {asym_id1} marked invalid due to clash "
                f"({clash_ratio_1:.2%} atoms clashed with chain {asym_id2})."
            )
            invalid_chains.add(asym_id1)
            continue
        elif is_clash_chain_2:
            logger.debug(
                f"{metadata.id}: Chain {asym_id2} marked invalid due to clash "
                f"({clash_ratio_2:.2%} atoms clashed with chain {asym_id1})."
            )
            invalid_chains.add(asym_id2)
            continue

        # Valid interface
        interfaces.append(InterfaceInfo(asym_ids=(asym_id1, asym_id2)))

    # Remove invalid interfaces
    valid_interfaces: list[InterfaceInfo] = []
    for iface in interfaces:
        asym_id1, asym_id2 = iface.asym_ids
        if asym_id1 not in invalid_chains and asym_id2 not in invalid_chains:
            valid_interfaces.append(iface)
    struct.metadata.interfaces = valid_interfaces


def prune_invalid_chains(struct: RefStructure, invalid_chains: set[int]):
    """Drop invalid chains from the structure."""
    metadata: Metadata = struct.metadata
    # Remove orphaned branched/covalent ligand chains
    for conn in struct.connections:
        asym_id1, asym_id2 = conn.asym_id
        if asym_id1 in invalid_chains or asym_id2 in invalid_chains:
            cm1 = metadata.get_chain_by_asym_id(asym_id1)
            cm2 = metadata.get_chain_by_asym_id(asym_id2)
            if cm1.ctype is C.ChainType.LIGAND:
                invalid_chains.add(asym_id1)
            if cm2.ctype is C.ChainType.LIGAND:
                invalid_chains.add(asym_id2)

    # Prune chains, connections, and interfaces
    metadata.chains = [m for m in metadata.chains if m.asym_id not in invalid_chains]
    struct.chains = [c for c in struct.chains if c.asym_id not in invalid_chains]
    struct.connections = [
        conn
        for conn in struct.connections
        if conn.asym_id[0] not in invalid_chains and conn.asym_id[1] not in invalid_chains
    ]
    metadata.interfaces = [
        iface
        for iface in struct.metadata.interfaces
        if (
            iface.asym_ids[0] not in invalid_chains
            and iface.asym_ids[1] not in invalid_chains
        )
    ]


# ==================================================
# Substructure sampling
# ==================================================
def crop_substructure(
    struct: RefStructure,
    max_chains: int = 20,
    seed: int = 42,
):
    """Sample a substructure with at most max_chains chains."""
    if struct.num_chains <= max_chains:
        return struct

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
            iface.asym_ids[0] in selected_asym_ids
            and iface.asym_ids[1] in selected_asym_ids
        )
    ]
