"""Pipeline to parse structure and metadata from structure files (PDB/MMCIF)."""

import pathlib
from typing import Any

import gemmi
import numpy as np

import kfold.constants as C
from kfold.data import schema, structure
from kfold.data.ccd import CCD, Component

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
chain_type_to_unk: dict[C.ChainType, str] = {
    C.ChainType.PROTEIN: "UNK",
    C.ChainType.RNA: "N",
    C.ChainType.DNA: "DN",
}


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

    # Dates (Deposit, Release, Revision)
    deposit = get_first_value(
        block, "_pdbx_database_status.recvd_initial_deposition_date"
    )
    # Revisions are stored in a loop.
    # Usually, the first item is the initial release, the last is the latest revision.
    rev_dates = block.find_values("_database_PDB_rev.date")
    release = rev_dates[0] if rev_dates else None
    latest_revision = rev_dates[-1] if rev_dates else None

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
        deposited=deposit,
        released=release,
        revised=latest_revision,
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
# Helper functions for Structure parsing
# ==================================================
def clean_up_structure(raw_struct: gemmi.Structure) -> None:
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

    if chain_type == C.ChainType.PROTEIN:
        standard_residues = set(C.residue.PROTEIN_RESIDUES_EXTENDED_STR)
        unk = "UNK"
    elif chain_type == C.ChainType.RNA:
        standard_residues = set(C.residue.RNA_RESIDUES_STR)
        unk = "N"
    elif chain_type == C.ChainType.DNA:
        standard_residues = set(C.residue.DNA_RESIDUES_STR)
        unk = "DN"
    else:
        standard_residues = set()
        unk = "???"  # This will raise error if used

    # ==================================================
    # Prepare residue information
    # ==================================================
    is_res_standards: list[bool] = []
    ref_mols: list[Component] = []
    num_residue_atoms: list[int] = []
    for name in ccd_sequences:
        if name == "MSE":
            # Replace selenomethionine with methionine
            name = "MET"
        if name in ccd:
            # Common molecule from CCD
            if name.startswith("LIG"):
                print("Use custom ligand residue from CCD:", name)
            ref_mol = ccd[name]
        elif name.startswith("LIG"):
            # Ligand residue created from SMILES
            if smiles is None:
                raise ValueError(f"SMILES must be provided for ligand residue {name}.")
            ref_mol = Component.from_smiles(name, smiles)
        else:
            # Fallback to UNK/N/DN for non-polymer residues
            print("Warning: residue name not in CCD:", name)
            if not chain_type.is_polymer:
                raise ValueError(f"Non-polymer residue {name} not found in CCD.")
            name = unk
            ref_mol = ccd[name]

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
    bond_atom_index_list: list[tuple[int, int]] = []
    bond_type_list: list[int] = []
    if chain_type is C.ChainType.LIGAND:
        for residue_index, ref_mol in enumerate(ref_mols, start=1):
            # Get ref atom names
            if drop_leaving_atoms:
                ref_atom_names = ref_mol.non_leaving_atom_names
            else:
                ref_atom_names = ref_mol.atom_names
            for (atom_name1, atom_name2), bond_type in ref_mol.bonds.items():
                idx1 = ref_atom_names.index(atom_name1)
                idx2 = ref_atom_names.index(atom_name2)
                bond_residue_index_list.append((residue_index, residue_index))
                bond_atom_index_list.append((idx1, idx2))
                bond_type_list.append(bond_type)

    bond_struct = structure.Bond(
        residue_index=np.array(bond_residue_index_list, dtype=np.uint32).reshape(-1, 2),
        atom_index=np.array(bond_atom_index_list, dtype=np.uint32).reshape(-1, 2),
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
    ccd: CCD,
    raw_struct: gemmi.Structure,
    metadata: schema.Metadata,
) -> structure.Structure:
    """Prepare reference structure from gemmi CIF block and metadata."""
    # ==================================================
    # Parse entities
    # ==================================================
    # Prepare chain index mappings
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
    asym_id_to_int: dict[AsymId, SymId] = {}
    for i, asym_id in enumerate(sorted(asym_id_to_entity_id.keys()), start=1):
        asym_id_to_int[asym_id] = i

    # NOTE: According to AlphaFold3, crystallograpy aids are excluded.
    exclude_crystal_aids: bool = False
    if metadata.exp is not None and metadata.exp.method is not None:
        if "XRAY" in metadata.exp.method.replace("-", "").upper():
            exclude_crystal_aids = True

    # Iterate over entities to collect valid ones
    valid_entities: list[gemmi.Entity] = []
    valid_entity_ids: set[EntityId] = set()

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
            unk: str = chain_type_to_unk[chain_type]

            # Get CCD sequences
            # gemmi.Entity.first_mon(v) gets the first name in case of microheterogeneity
            # e.g., [VAL, ALA/GLY, TRP, ASP, ...] -> [VAL, ALA, TRP, ASP, ...]
            ccd_sequences = entity.full_sequence
            ccd_sequences: list[str] = [gemmi.Entity.first_mon(v) for v in ccd_sequences]

            if len(ccd_sequences) < 4:
                # Skip too short polymer entities
                continue
            if {
                C.residue.convert_ccd_name_to_one_letter(v, unk) for v in ccd_sequences
            } < {unk}:
                # Skip all unknown polymer entities
                continue

        elif entity.entity_type in {
            gemmi.EntityType.NonPolymer,
            gemmi.EntityType.Branched,
        }:
            # Ligand, ion, or branched ligands
            # TODO: for Boltz1, all custom ligands are stored as NonPolymer with
            # residue name "LIG". Handle them properly future.

            # Read CCD sequences from the model.
            ref_asym_id: AsymId = entity.subchains[0]
            raw_chain: gemmi.ResidueSpan = raw_struct[0].get_subchain(ref_asym_id)
            ccd_sequences: list[str] = [res.name for res in raw_chain]

            is_valid_entity = True

            # Check if all ligand residues are in CCD
            for res_name in ccd_sequences:
                if res_name.startswith("LIG"):
                    # Allow custom ligand residues
                    continue
                if res_name in C.ccd.LIGAND_EXCLUSIONS:
                    # Exclude unwanted ligands
                    is_valid_entity = False
                    break
                elif exclude_crystal_aids and res_name in C.ccd.CRYSTALLIZATION_AIDS:
                    # Exclude aid ions in crystal structures
                    is_valid_entity = False
                    break
                elif res_name not in ccd:
                    # Residue not found in CCD
                    print(f"Warning: Non-polymer residue {res_name} not found in CCD.")
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

    # Collect valid asym_ids
    valid_asym_ids: set[AsymId] = set(
        asym_id
        for asym_id, entity_id in asym_id_to_entity_id.items()
        if entity_id in valid_entity_ids
    )

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
            asym_id: AsymId = residue.subchain
            if asym_id in valid_asym_ids:
                seq_id: gemmi.SeqId = residue.seqid
                residue_map[(auth_id, seq_id.icode, seq_id.num)] = residue

    # Find covalent bonded entities
    # This is used to identify covalent inhibitors:
    #   NonPolymer entities with covalent bonds to other entities
    linked_subchains: set[AsymId] = set()
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
            linked_subchains.add(res1.subchain)
            linked_subchains.add(res2.subchain)
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
    for entity in valid_entities[:]:
        entity: gemmi.Entity
        entity_id: EntityId = int(entity.name)
        if entity.entity_type == gemmi.EntityType.Branched:
            for asym_id in entity.subchains:
                if asym_id not in linked_subchains:
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
        # For custom ligands, ensure unique residue names
        # NOTE: "Boltz" saves all ligands as "LIG"
        if ctype == C.ChainType.LIGAND:
            if ccd_sequences[0].startswith("LIG"):
                assert len(ccd_sequences) == 1, (
                    "Multiple LIG residues not supported without SMILES."
                )
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
            sym_id: SymId = asym_id_to_sym_id[asym_id]
            if asym_id not in valid_asym_ids:
                # Skip invalid chains
                continue

            if (
                entity.entity_type == gemmi.EntityType.NonPolymer
                and asym_id in linked_subchains
            ):
                # For covalent inhibitors, create a new chain struct without leaving atoms
                c = prepare_ref_chain(
                    chain_type=ctype,
                    entity_id=entity_id,
                    asym_id=asym_id_to_int[asym_id],
                    sym_id=sym_id,
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
                    sym_id=sym_id,
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
            if c.asym_id == asym_id_to_int[asym_id1]:
                for atom_i in c.residue.iter_residue_atoms(res_idx1):
                    if c.atom.name[atom_i] == atom1:
                        is_atom1_found = True
                        break
            if c.asym_id == asym_id_to_int[asym_id2]:
                for atom_i in c.residue.iter_residue_atoms(res_idx2):
                    if c.atom.name[atom_i] == atom2:
                        is_atom2_found = True
                        break
        if not is_atom1_found:
            print(f"Atom {atom1} not found in residue {res_idx1} of chain {asym_id1}.")
            continue
        if not is_atom2_found:
            print(f"Atom {atom2} not found in residue {res_idx2} of chain {asym_id2}.")
            continue

        connections.append(
            structure.CovalentConnection(
                asym_id=(
                    asym_id_to_int[asym_id1],
                    asym_id_to_int[asym_id2],
                ),
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

    return structure.Structure(
        chains=tuple(chain_structs),
        connections=tuple(connections),
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
            print(
                f"Residue index {residue_index} out of bounds for chain with length "
                f"{len(ccd_sequence)}."
            )
            continue

        # Get atoms.
        name_to_atom: dict[str, gemmi.Atom] = {a.name.upper(): a for a in res}

        # Map MSE to MET, put the selenium atom in the sulphur column
        res_name = ccd_sequence[residue_index - 1]
        if res_name == "MSE":
            res_name = "MET"

        # WARN: in parse_ref_chain(), MSE is already converted to MET.
        # Therefore, I place this statements outside of the res_name check.
        if res_name == "MET" and "SE" in name_to_atom:
            name_to_atom["SD"] = name_to_atom["SE"]

        for atom_i in ref_chain.residue.iter_residue_atoms(residue_index):
            atom_name: str = atom_names[atom_i]
            if atom_name in name_to_atom:
                atom: gemmi.Atom = name_to_atom[atom_name]
                coords: gemmi.Position = atom.pos
                ref_chain.atom.label_coords[atom_i, 0] = coords.x
                ref_chain.atom.label_coords[atom_i, 1] = coords.y
                ref_chain.atom.label_coords[atom_i, 2] = coords.z
                ref_chain.atom.bfactor[atom_i] = atom.b_iso
                ref_chain.atom.is_resolved[atom_i] = True
                name_to_atom.pop(atom_name)
            else:
                # Leave as NaN if atom not found
                print(
                    f"Atom {atom_name} not found in residue {res_name} {residue_index}."
                )
                continue
        for leftover_atom in name_to_atom.keys():
            if leftover_atom == "OXT":
                # Ignore missing OXT atoms
                continue
            print(
                f"Atom {leftover_atom} in residue {res_name} {residue_index} not mapped."
            )


def insert_coordinates(
    ref_struct: structure.Structure,
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
        insert_chain_coordinates(ref_chain, raw_chain)


def parse_cif(
    cif_path: str | pathlib.Path,
    ccd: CCD,
    source: str = "rcsb",
) -> structure.Structure:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    # Read CIF file
    doc: gemmi.cif.Document = gemmi.cif.read_file(str(cif_path))
    block: gemmi.cif.Block = doc[0]

    # Get metadata
    # Handle cases like "1abc.cif.gz"
    name = pathlib.Path(cif_path).name.split(".")[0]
    metadata = prepare_metadata(name, block, source)

    # Prepare raw structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    clean_up_structure(raw_struct)
    expand_first_assembly(raw_struct)

    # Prepare reference structure
    ref_struct = prepare_ref_structure(ccd, raw_struct, metadata)

    # Insert coordinates
    insert_coordinates(ref_struct, raw_struct, metadata)

    return ref_struct
