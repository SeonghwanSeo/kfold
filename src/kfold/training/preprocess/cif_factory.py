"""Pipeline to prepare reference structures
# TODO: add PDB parsing later
"""

import itertools
import logging
import pathlib
from collections import defaultdict
from datetime import datetime
from typing import Any

import gemmi
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

import kfold.constants as C
from kfold.data.pipelines import structure_preparation
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import (
    ExperimentRecord,
    InterfaceInfo,
    Metadata,
    PredictionRecord,
)
from kfold.data.types.structure import Chain, CovalentConnection, RefStructure

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

    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    expand_first_assembly(raw_struct)
    clean_up_gemmi_structure(raw_struct)

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

    # Detect orphaned branched ligands
    propagate_invalidity_to_ligands(ref_struct, invalid_chains)

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
    model: str,
) -> Metadata:
    """Parse metadata from CIF block."""
    # Parse experiment record
    pred_record = PredictionRecord(model=model)
    return Metadata(
        id=name,
        source="pred",
        pred=pred_record,
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


def add_entity_info(raw_struct: gemmi.Structure, format: str = "pdb") -> None:
    """Add entity_id for pdb-format synthetic data"""

    def get_res_idx(res: gemmi.Residue) -> int:
        return res.label_seq or res.seqid.num

    raw_struct.setup_entities()
    for entity in raw_struct.entities:
        if format == "pdb":
            entity.name = str("ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(entity.name) + 1)
        # Check if full_sequence is missing (due to no SEQRES in PDB)
        if not entity.full_sequence:
            # Get the first subchain ID belonging to this entity
            target_subchain = entity.subchains[0]
            model = raw_struct[0]
            for chain in model:
                poly = chain.get_polymer()
                # Check if this polymer matches our target subchain
                if len(poly) > 0 and poly[0].subchain == target_subchain:
                    # Generate ccd sequence
                    res_dict = {get_res_idx(res): res.name for res in poly}
                    entity.full_sequence = [
                        res_dict.get(i, "UNK") for i in range(1, max(res_dict.keys()) + 1)
                    ]
                    break


def add_res_idx(raw_struct: gemmi.Structure) -> None:
    for chain in raw_struct[0]:
        for res in chain:
            res.label_seq = res.seqid.num


# ==================================================
# Main functions for reference structure preparation
# ==================================================
def prepare_ref_structure(
    raw_struct: gemmi.Structure,
    metadata: Metadata,
    ccd: CCD,
    smiles_dict: dict[str, str] | None = None,
) -> RefStructure:
    """Prepare reference structure from gemmi CIF block and metadata."""
    smiles_dict: dict[str, str] = smiles_dict or {}

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

    label_id_to_auth_id: dict[LabelId, AuthId] = {}
    for chain in raw_struct[0]:
        auth_id: AuthId = "".join(filter(str.isalpha, chain.name))
        for subchain in chain.subchains():
            label_id: LabelId = subchain.subchain_id()
            label_id = "".join(filter(str.isalpha, label_id))
            label_id_to_auth_id[label_id] = auth_id

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
            ccd_sequences: list[str] = [
                v if (v in C.ccd.CCD_NAMES and v in ccd) else unk
                for v in entity.full_sequence
            ]

        elif entity.entity_type in {
            gemmi.EntityType.NonPolymer,
            gemmi.EntityType.Branched,
        }:
            # Ligand, ion, or branched ligands
            ctype = C.ChainType.LIGAND
            ref_label_id: LabelId = entity.subchains[0]
            raw_chain: gemmi.ResidueSpan = raw_struct[0].get_subchain(ref_label_id)
            ccd_sequences: list[str] = [res.name for res in raw_chain]

            if len(ccd_sequences) == 0:
                # Skip empty ligand
                continue

            # Check if all residues are in CCD
            is_valid_entity = True
            for res_name in ccd_sequences:
                if res_name.startswith("LIG") or res_name in smiles_dict:
                    # Synthetic predictors use custom ligand names such as LIG or l01.
                    assert len(ccd_sequences) == 1, (
                        "Multi-residue custom ligands not supported in CIF parsing."
                    )
                elif res_name in excluded_ligands:
                    # Exclude unwanted ligands
                    logging.debug(f"Excluding ligand {res_name} in entity {entity_id}.")
                    is_valid_entity = False
                    break
                elif res_name not in ccd:
                    # Residue not found in CCD
                    logging.warning(f"Ligand {res_name} not found in CCD.")
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
        entity_id_to_ctype[entity_id] = ctype
        entity_id_to_seq[entity_id] = ccd_sequences

        # Mark all label_ids as valid initially
        for label_id in entity.subchains:
            valid_label_ids.add(label_id)

    # ==================================================
    # Identify covalent bonded subchains
    # ==================================================
    # NOTE: gemmi Connection uses auth_asym_id instead of label_asym_id
    # * single auth_asym_id can map to multiple label_asym_id (e.g., covalent inhibitor)
    def get_res_key(res: gemmi.Residue) -> ResKey:
        label_id: LabelId = res.subchain
        seq_id: gemmi.SeqId = res.seqid
        return (label_id, seq_id.icode, seq_id.num)

    residue_map: dict[tuple[AuthId, str, int | None], gemmi.Residue] = {}
    for chain in raw_struct[0]:
        chain: gemmi.Chain  # one auth_id can be mapped to multiple label_id(asym_id)
        auth_id: AuthId = chain.name
        for residue in chain:
            seq_id: gemmi.SeqId = residue.seqid
            residue_map[(auth_id, seq_id.icode, seq_id.num)] = residue

    residue_index_map: dict[ResKey, int] = {}
    for subchain in raw_struct[0].subchains():
        label_id: LabelId = subchain.subchain_id()
        if label_id not in valid_label_ids:
            continue
        entity_id: EntityId = label_id_to_entity_id[label_id]
        ctype: C.ChainType = entity_id_to_ctype[entity_id]
        for res_idx, residue in enumerate(subchain, start=1):
            res_key = get_res_key(residue)
            if ctype.is_polymer:
                residue_index_map[res_key] = residue.label_seq
            else:
                residue_index_map[res_key] = res_idx

    # tuple of (label_id, res_idx, atom_name) pair
    chain_linkage: dict[LabelId, set[LabelId]] = defaultdict(set)
    linked_chains: set[LabelId] = set()
    linked_bonds: list[tuple[tuple[LabelId, int, str], tuple[LabelId, int, str]]] = []
    bonded_atoms: dict[LabelId, dict[int, set[str]]] = defaultdict(dict)
    for connect in raw_struct.connections:
        connect: gemmi.Connection
        if connect.type != gemmi.ConnectionType.Covale:
            continue
        p1, p2 = connect.partner1, connect.partner2
        k1 = (p1.chain_name, p1.res_id.seqid.icode, p1.res_id.seqid.num)
        k2 = (p2.chain_name, p2.res_id.seqid.icode, p2.res_id.seqid.num)
        if k1 not in residue_map or k2 not in residue_map:
            continue
        res1, res2 = residue_map[k1], residue_map[k2]
        # Get label chain ids
        label_id1, label_id2 = res1.subchain, res2.subchain
        if label_id1 not in valid_label_ids or label_id2 not in valid_label_ids:
            continue
        chain_linkage[label_id1].add(label_id2)
        chain_linkage[label_id2].add(label_id1)
        # Get residue indices
        res_key1, res_key2 = get_res_key(res1), get_res_key(res2)
        res_idx1, res_idx2 = residue_index_map[res_key1], residue_index_map[res_key2]
        # Get atom names
        atom1, atom2 = p1.atom_name, p2.atom_name
        linked_chains.update({label_id1, label_id2})
        linked_bonds.append(((label_id1, res_idx1, atom1), (label_id2, res_idx2, atom2)))
        # Record bonded atoms for covalent ligands
        bonded_atoms[label_id1].setdefault(res_idx1, set()).add(atom1)
        bonded_atoms[label_id2].setdefault(res_idx2, set()).add(atom2)

    del residue_map, residue_index_map  # free memory

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
        if ctype is C.ChainType.LIGAND and (
            ccd_sequences[0].startswith("LIG") or ccd_sequences[0] in smiles_dict
        ):
            ref_label_id: LabelId = entity.subchains[0]
            custom_id: str = ccd_sequences[0]
            assert custom_id in smiles_dict, (
                f"Custom ligand {custom_id} missing in given SMILES dict: {smiles_dict}"
            )
            smiles = smiles_dict[custom_id]
            assert smiles is not None, "Failed to infer SMILES for ligand"

        # Parse reference chain once
        # Drop leaving atoms for polymers, and keep all atoms for ligands as default
        # (Will filter later based on covalent bonds)
        parsed_chain: Chain = structure_preparation.prepare_ref_chain(
            chain_type=ctype,
            ccd_sequences=ccd_sequences,
            ccd=ccd,
            smiles=smiles,
        )
        if ctype.is_nonpolymer:
            # Collect residue atom names for non-polymer chains
            ref_residue_atom_names: dict[int, set[str]] = {}
            for res_idx in range(1, parsed_chain.num_residues + 1):
                atom_slice = parsed_chain.residue.get_atom_slice(res_idx)
                atom_names = parsed_chain.atom.name[atom_slice].tolist()
                ref_residue_atom_names[res_idx] = set(atom_names)

        for label_id in entity.subchains:
            if label_id not in valid_label_ids:
                # Skip invalid chains
                continue
            if ctype.is_nonpolymer and label_id in linked_chains:
                c = structure_preparation.prepare_ref_chain(
                    chain_type=ctype,
                    entity_id=entity_id,
                    asym_id=label_id_to_asym_id[label_id],
                    sym_id=label_id_to_sym_id[label_id],
                    ccd_sequences=ccd_sequences,
                    ccd=ccd,
                    bonded_atoms=bonded_atoms[label_id],
                )
            else:
                # Otherwise, clone chain for each subchain
                # Coordinate insertion mutates these arrays, so every copy must own them.
                c = parsed_chain.copy_with(
                    deepcopy=True,
                    entity_id=entity_id,
                    asym_id=label_id_to_asym_id[label_id],
                    sym_id=label_id_to_sym_id[label_id],
                )
            chain_structs.append(c)
    asym_id_to_chain: dict[int, Chain] = {c.asym_id: c for c in chain_structs}

    # ==================================================
    # Construct connection structs
    # ==================================================
    connections: list[CovalentConnection] = []
    for at1, at2 in linked_bonds:
        label_id1, res_idx1, atom1 = at1
        label_id2, res_idx2, atom2 = at2
        asym_id1 = label_id_to_asym_id[label_id1]
        asym_id2 = label_id_to_asym_id[label_id2]
        c1 = asym_id_to_chain[asym_id1]
        c2 = asym_id_to_chain[asym_id2]

        # Skip polymer-polymer connections
        if c1.ctype.is_polymer and c2.ctype.is_polymer:
            continue

        # Check atom existence
        # NOTE (Seonghwan): In current pipeline, it is expected that only
        # connections involving unknown residues may have missing atoms.
        valid_atoms1 = c1.atom.name[c1.residue.get_atom_slice(res_idx1)]
        is_atom1_found = atom1 in valid_atoms1
        if not is_atom1_found and atom1.endswith("1") and atom1[:-1].isalpha():
            # fallback: remove number suffixes (e.g., "O1" -> "O")
            if atom1[:-1] in valid_atoms1:
                atom1 = atom1[:-1]
                is_atom1_found = True

        valid_atoms2 = c2.atom.name[c2.residue.get_atom_slice(res_idx2)]
        is_atom2_found = atom2 in valid_atoms2
        if not is_atom2_found and atom2.endswith("1") and atom2[:-1].isalpha():
            # fallback: remove number suffixes (e.g., "O1" -> "O")
            if atom2[:-1] in valid_atoms2:
                atom2 = atom2[:-1]
                is_atom2_found = True

        if not (is_atom1_found and is_atom2_found):
            res_name1 = c1.residue.name[res_idx1 - 1]
            res_name2 = c2.residue.name[res_idx2 - 1]
            atom1_key = f"{label_id1}:{res_idx1}({res_name1}):{atom1}"
            atom2_key = f"{label_id2}:{res_idx2}({res_name2}):{atom2}"
            logger.warning(
                f"Skipping connection: atoms not found in {metadata.id}: "
                f"({atom1_key} - {atom2_key}).\n"
                f"Valid atoms in {atom1_key}: {valid_atoms1.tolist()}\n"
                f"Valid atoms in {atom2_key}: {valid_atoms2.tolist()}"
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
        # label id: including chain letter and optional assembly number
        name: LabelId = asym_id_to_label_id[c.asym_id]
        chain_info = structure_preparation.prepare_chain_metadata(c, name=name)
        # Save label_asym_id/auth_asym_id, which is same to visualized in RCSB
        label_asym_id: LabelId = "".join(filter(str.isalpha, name))
        chain_info.label_asym_id = label_asym_id
        chain_info.auth_asym_id = label_id_to_auth_id[label_asym_id]
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

    if ref_chain.smiles is not None:
        # For custom ligands, only the atom orders are guaranteed to be the same.
        atom_names_in_cif: list[str] = []
        atom_coords_in_cif: list[tuple[float, float, float]] = []
        atom_bfactors_in_cif: list[float] = []
        for res in raw_chain:
            for atom in res:
                coords: gemmi.Position = atom.pos
                atom_names_in_cif.append(atom.name)
                atom_coords_in_cif.append((coords.x, coords.y, coords.z))
                atom_bfactors_in_cif.append(min(atom.b_iso, 100.0))

        assert len(atom_names_in_cif) == len(atom_names), (
            f"Number of atoms in CIF ({len(atom_names_in_cif)}) does not match "
            f"reference chain ({len(atom_names)})."
        )

        for a1, a2 in zip(atom_names, atom_names_in_cif, strict=True):
            # It is allowed that the atom uniq-numbering is different,
            # but the atom types should be the same for custom ligands.
            _a1 = "".join(filter(str.isalpha, a1)).lower()
            _a2 = "".join(filter(str.isalpha, a2)).lower()
            if _a1 != _a2:
                raise ValueError(
                    f"Atom name mismatch for custom ligand: {a1} vs {a2}. "
                    f"After removing numbers: {_a1} vs {_a2}."
                )
        # Map atom names to coordinates for quick lookup
        ref_chain.atom.coords[:, :] = atom_coords_in_cif
        ref_chain.atom.bfactor[:] = atom_bfactors_in_cif
        return

    for res_i, res in enumerate(raw_chain):
        res: gemmi.Residue

        # Get residue index
        if ref_chain.ctype.is_polymer:
            residue_index = res.label_seq
            if residue_index is None:
                logger.warning(
                    f"Residue {res.name} in chain {raw_chain.subchain_id()} "
                    f"missing label_seq; skipping."
                )
                continue
        else:
            # For non-polymer residues, use 1-based index within the entity
            # assert res.label_seq is None
            residue_index: int = res_i + 1

        if residue_index < 1 or residue_index > len(ccd_sequence):
            raise ValueError(
                f"Residue index {residue_index} out of bounds for chain with length "
                f"{len(ccd_sequence)}.\n"
                f"Residue info: {res.name} {res.seqid} (label_seq={res.label_seq})\n"
                f"Ref Chain info: {ref_chain}"
            )

        # Get atoms.
        name_to_atom: dict[str, gemmi.Atom] = {a.name.upper(): a for a in res}
        res_name = ccd_sequence[residue_index - 1]
        for atom_i in ref_chain.residue.iter_residue_atoms(residue_index):
            n: str = atom_names[atom_i]
            if n in name_to_atom:
                atom: gemmi.Atom = name_to_atom[n]
                coords: gemmi.Position = atom.pos
                ref_chain.atom.coords[atom_i, :] = (coords.x, coords.y, coords.z)
                ref_chain.atom.bfactor[atom_i] = min(atom.b_iso, 100.0)  # cap bfactor
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


def validate_target(struct: RefStructure) -> bool:
    """Check if a target is valid.
    See AlphaFold3 SI Section 2.5.4:
      > Filtering of targets:
        - ...
        - Any polymer chain containing fewer than 4 resolved residues is filtered out.
    """
    for c in struct.chains:
        if c.is_polymer:
            ref_atom_coords = get_chain_ref_atom_coordinates(c)
            is_resolved = np.isfinite(ref_atom_coords).all(axis=-1)
            n_resolved = np.sum(is_resolved)
            if n_resolved < 4:
                logger.debug(
                    f"{struct.id}: Target invalid due to insufficient resolved "
                    f"residues in chain {c.asym_id} ({n_resolved})."
                )
                return False
    return True


def validate_chain_geometry(
    struct: RefStructure,
    invalid_chains: set[int],
):
    """Check if a polymer chain is valid."""
    for c in struct.chains:
        ctype: C.ChainType = c.ctype

        # Get reference atom coordinates and resolved flags
        ref_atom_coords = get_chain_ref_atom_coordinates(c)
        is_resolved = np.isfinite(ref_atom_coords).all(axis=-1)
        n_resolved = np.sum(is_resolved)

        if ctype.is_polymer:
            # For polymer chains, skip too short chains
            if n_resolved < 4:
                logger.debug(
                    f"{struct.id}: Chain {c.asym_id} marked invalid "
                    f"due to insufficient resolved residues ({n_resolved})."
                )
                invalid_chains.add(c.asym_id)
            seq = c.get_sequence()
            if (ctype.is_protein and set(seq) <= {"X"}) or (
                ctype.is_nucleic_acid and set(seq) <= {"N"}
            ):
                logger.debug(
                    f"{struct.id}: Chain {c.asym_id} marked invalid "
                    f"due to all-unknown sequence."
                )
                invalid_chains.add(c.asym_id)
        else:
            # For non-polymer chains, only check if any atom is resolved
            if n_resolved == 0:
                logger.debug(
                    f"{struct.id}: Chain {c.asym_id} marked invalid "
                    f"due to no resolved atoms."
                )
                invalid_chains.add(c.asym_id)

        # For protein chains, check CA trace continuity
        if ctype.is_protein:
            left = ref_atom_coords[:-1]
            right = ref_atom_coords[1:]
            dists = np.linalg.norm(left - right, axis=-1)
            if np.any(dists > 10.0):
                logger.debug(
                    f"{struct.id}: Chain {c.asym_id} marked invalid "
                    f"due to CA trace discontinuity."
                )
                invalid_chains.add(c.asym_id)
                continue


def detect_interfaces_and_detect_clashes(
    struct: RefStructure,
    invalid_chains: set[int],
):
    """
    Detect valid interfaces between chains and prune chains with severe clashes.
    1. Interface detection using all atoms (< 5 A)
    2. Clash check using all atoms (< 1.7 A)
    """

    def is_invalid(chain: Chain) -> bool:
        return chain.asym_id in invalid_chains

    metadata: Metadata = struct.metadata
    interfaces: list[InterfaceInfo] = []

    # Collect coordinates
    all_coords_dict: dict[int, np.ndarray] = {}
    all_box_dict: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_tree_dict: dict[int, cKDTree] = {}

    for chain in struct.chains:
        """Get all valid atom coordinates for each chain."""
        if is_invalid(chain):
            continue
        all_coords = chain.atom.coords
        all_coords = all_coords[np.isfinite(all_coords).all(axis=-1)]
        assert all_coords.shape[0] > 0, f"Chain {chain.asym_id} has no valid atoms."
        all_coords_dict[chain.asym_id] = all_coords
        all_box_dict[chain.asym_id] = (all_coords.min(0), all_coords.max(0))
        all_tree_dict[chain.asym_id] = cKDTree(all_coords)

    for i1, i2 in itertools.combinations(range(struct.num_chains), 2):
        # Only consider valid chains
        chain1 = struct.chains[i1]
        chain2 = struct.chains[i2]
        asym_id1 = chain1.asym_id
        asym_id2 = chain2.asym_id
        if is_invalid(chain1) or is_invalid(chain2):
            continue

        # Check for all atom contacts (5 Angstrom cutoff)
        coords1 = all_coords_dict[asym_id1]  # [N, 3]
        coords2 = all_coords_dict[asym_id2]  # [M, 3]
        num_atoms_1 = coords1.shape[0]
        num_atoms_2 = coords2.shape[0]

        # Axis-aligned bounding box check
        min1, max1 = all_box_dict[asym_id1]
        min2, max2 = all_box_dict[asym_id2]
        if np.any(min1 - max2 > 5.0) or np.any(min2 - max1 > 5.0):
            # No contact detected
            continue

        # KDTree check
        tree1 = all_tree_dict[asym_id1]
        tree2 = all_tree_dict[asym_id2]
        if tree1.count_neighbors(tree2, r=5.0) == 0:
            # No contact detected
            continue

        if tree1.count_neighbors(tree2, r=1.7) > 0:
            # Check for clash ratio
            clash_indices_1 = tree1.query_ball_tree(tree2, r=1.7)
            clash_indices_2 = tree2.query_ball_tree(tree1, r=1.7)
            n_clash_atoms_1 = sum(1 for neighbors in clash_indices_1 if neighbors)
            n_clash_atoms_2 = sum(1 for neighbors in clash_indices_2 if neighbors)

            clash_ratio_1 = n_clash_atoms_1 / num_atoms_1
            clash_ratio_2 = n_clash_atoms_2 / num_atoms_2
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


def propagate_invalidity_to_ligands(struct: RefStructure, invalid_chains: set[int]):
    """Detect orphaned branched ligand chains linked to invalid chains."""
    polymer_asym_ids: set[int] = set(
        c.asym_id for c in struct.chains if c.ctype.is_polymer
    )
    links: dict[int, list[int]] = {c.asym_id: [] for c in struct.chains}
    for conn in struct.connections:
        asym_id1, asym_id2 = conn.asym_id
        links[asym_id1].append(asym_id2)
        links[asym_id2].append(asym_id1)

    visited: set[int] = set()

    def dfs_remove_branches(curr: int):
        if curr in visited:
            return
        visited.add(curr)

        for neighbor in links[curr]:
            if neighbor in polymer_asym_ids:
                continue
            invalid_chains.add(neighbor)
            dfs_remove_branches(neighbor)

    for start_id in list(invalid_chains):
        dfs_remove_branches(start_id)


def prune_invalid_chains(struct: RefStructure, invalid_chains: set[int]):
    """Drop invalid chains from the structure."""
    # Drop invalid chains
    struct.chains = [c for c in struct.chains if c.asym_id not in invalid_chains]
    valid_chains = set(c.asym_id for c in struct.chains)
    struct.connections = [
        conn for conn in struct.connections if set(conn.asym_id).issubset(valid_chains)
    ]
    # Update metadata
    metadata: Metadata = struct.metadata
    metadata.chains = [m for m in metadata.chains if m.asym_id in valid_chains]
    metadata.interfaces = [
        iface
        for iface in struct.metadata.interfaces
        if iface.asym_ids[0] in valid_chains and iface.asym_ids[1] in valid_chains
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
    dists = cdist(ref_coords1, ref_coords2)
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
