"""Pipeline to prepare reference structures"""

import logging

import numpy as np

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import (
    Atom,
    Bond,
    Chain,
    CovalentConnection,
    RefStructure,
    Residue,
)

logger = logging.getLogger(__name__)

# Type aliases for better readability
EntityId = int
AsymId = str
SymId = int
AuthId = str
ResKey = tuple[AsymId, str, int | None]

# Constants
# three-letter codes
chain_type_to_standard_residues: dict[C.ChainType, set[str]] = {
    C.ChainType.PROTEIN: set(C.residue.PROTEIN_RESIDUES_STR),
    C.ChainType.RNA: set(C.residue.RNA_RESIDUES_STR),
    C.ChainType.DNA: set(C.residue.DNA_RESIDUES_STR),
    C.ChainType.LIGAND: set(),
    C.ChainType.ION: set(),
}


def prepare_structure(
    chains: list[Chain],
    connections: list[CovalentConnection],
    metadata: Metadata,
) -> RefStructure:
    """Prepare the reference structure from chains.

    Parameters
    ----------
    chains : list[Chain]
        List of reference chains.
    connections : list[CovalentConnection]
        List of covalent connections between chains.
    metadata : Metadata
        Metadata associated with the structure.

    Returns
    -------
    RefStructure
        The prepared reference structure.
    """
    return RefStructure(
        chains=chains,
        connections=connections,
        metadata=metadata,
    )


def prepare_ref_chain(
    chain_type: C.ChainType,
    ccd_sequences: list[str],
    ccd: CCD,
    smiles: str | None = None,
    entity_id: int = 0,
    asym_id: int = 0,
    sym_id: int = 0,
    drop_leaving_atoms: bool = True,
) -> Chain:
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
        if chain_type.is_polymer and is_standard:
            # For standard polymer residues, use standard atom counts
            num_residue_atoms.append(len(C.atom.residue_atoms[name]))
        elif drop_leaving_atoms:
            num_residue_atoms.append(ref_mol.num_non_leaving_atoms)
        else:
            num_residue_atoms.append(ref_mol.num_atoms)

    residue_struct = Residue(
        name=np.array(ccd_sequences, dtype=np.dtype("<U6")),
        num_atoms=np.array(num_residue_atoms, dtype=np.uint8),
        is_standard=np.array(is_res_standards, dtype=bool),
    )

    # ==================================================
    # Prepare atom information
    # ==================================================
    atom_name_list: list[str] = []
    for ref_mol in ref_mols:
        if chain_type.is_polymer and ref_mol.code in C.atom.residue_atoms:
            # For standard residues, use pre-defined atom names
            atom_names = C.atom.residue_atoms[ref_mol.code]
        elif drop_leaving_atoms:
            # For non-standard residues, drop leaving atoms if specified
            atom_names = ref_mol.non_leaving_atom_names
        else:
            # Use all atoms for non-standard residues
            atom_names = ref_mol.atom_names
        atom_name_list.extend(atom_names)
    num_atoms = len(atom_name_list)
    assert num_atoms == sum(num_residue_atoms), "Mismatch in total number of atoms."

    # Empty label coordinates and resolved flags
    label_coords = np.full((num_atoms, 3), np.nan, dtype=np.float32)
    is_atom_resolved = np.zeros((num_atoms,), dtype=bool)
    bfactors = np.full((num_atoms,), np.nan, dtype=np.float32)
    # Empty apo coordinates and pLDDT
    apo_coords = np.full((num_atoms, 3), np.nan, dtype=np.float32)
    apo_plddt = np.full((num_atoms), np.nan, dtype=np.float32)

    atom_struct = Atom(
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

    bond_struct = Bond(
        residue_index=np.array(bond_residue_index_list, dtype=np.uint32).reshape(-1, 2),
        atom_name=np.array(bond_atom_name_list, dtype=np.dtype("<U4")).reshape(-1, 2),
        bond_type=np.array(bond_type_list, dtype=np.uint8),
    )

    return Chain(
        chain_type=chain_type.value,
        entity_id=entity_id,
        asym_id=asym_id,
        sym_id=sym_id,
        residue=residue_struct,
        atom=atom_struct,
        bond=bond_struct,
    )
