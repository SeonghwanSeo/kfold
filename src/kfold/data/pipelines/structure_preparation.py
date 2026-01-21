"""Pipeline to prepare reference structures"""

import logging
from functools import lru_cache

import numpy as np

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.metadata import ChainInfo, Metadata
from kfold.data.types.structure import (
    Atom,
    Bond,
    Chain,
    CovalentConnection,
    RefStructure,
    Residue,
)

logger = logging.getLogger(__name__)

# Constants
# three-letter codes
chain_type_to_standard_residues: dict[C.ChainType, set[str]] = {
    C.ChainType.PROTEIN: C.residue.PROTEIN_RESIDUES_STR_SET,
    C.ChainType.RNA: C.residue.RNA_RESIDUES_STR_SET,
    C.ChainType.DNA: C.residue.DNA_RESIDUES_STR_SET,
    C.ChainType.LIGAND: set(),
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
    valid_atom_names: dict[int, set[str]] | None = None,
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
    valid_atom_names : dict[int, set[str]] | None, optional
        Optional mapping from residue index to atom names to include, by default None.
    """
    # ==================================================
    # Validate inputs
    # ==================================================
    if smiles is not None:
        assert chain_type == C.ChainType.LIGAND
        assert len(ccd_sequences) == 1
        assert ccd_sequences[0].startswith("LIG")

    if valid_atom_names is not None:
        assert chain_type == C.ChainType.LIGAND
        drop_leaving_atoms = False  # Override to keep specified atoms
        for residue_index in valid_atom_names:
            assert 1 <= residue_index <= len(ccd_sequences)

    # Normalize residue names to uppercase
    ccd_sequences = [v.upper() for v in ccd_sequences]

    standard_residues: set[str] = chain_type_to_standard_residues[chain_type]

    @lru_cache  # No cache limit within a single function call
    def get_ref_mol(ccd_code: str) -> Component:
        """Get reference molecule for a residue."""
        if ccd_code in ccd:
            return ccd[ccd_code]
        else:
            raise ValueError(f"Residue {ccd_code} not found in CCD database.")

    @lru_cache  # No cache limit within a single function call
    def get_ref_atom_names(code: str) -> tuple[str, ...]:
        """Get reference atom names for a residue."""
        if code in ccd:
            ref_mol = ccd[code]
            if chain_type.is_polymer and code in standard_residues:
                # Return pre-defined residue atoms for standard polymer residues
                return C.atom.residue_atoms[code]
            elif drop_leaving_atoms:
                # Drop leaving atoms for non-standard polymer residues, glycans,
                # and covalent ligands.
                return ref_mol.non_leaving_atom_names
            else:
                # Return all atoms for non-polymer residues.
                return ref_mol.names
        else:
            raise ValueError(f"Residue {code} not found in CCD database.")

    @lru_cache  # No cache limit within a single function call
    def get_ref_mol_info(code: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get reference molecule atom information for a residue."""
        ref_mol: Component = get_ref_mol(code)
        atom_names: tuple[str, ...] = get_ref_atom_names(code)
        atom_to_index: dict[str, int] = ref_mol.get_atom_index_map()
        atom_indices: list[int] = [atom_to_index[an] for an in atom_names]
        atom_names: np.ndarray = np.array(atom_names, dtype=np.dtype("<U4"))
        atom_elem: np.ndarray = ref_mol.elements[atom_indices]
        atom_charge: np.ndarray = ref_mol.charges[atom_indices]
        return atom_names, atom_elem, atom_charge

    # ==================================================
    # Prepare residue information
    # ==================================================
    is_res_standards: list[bool] = []
    ref_mols: list[Component] = []
    num_residue_atoms: list[int] = []
    residue_atoms_list: list[tuple[str, ...]] = []
    for code in ccd_sequences:
        if code in ccd:
            # Common molecule from CCD
            assert not code.startswith("LIG"), "LIG codes should be handled separately."
            ref_mol = get_ref_mol(code)
        elif code.startswith("LIG"):
            # Ligand residue created from SMILES
            if smiles is None:
                raise ValueError(f"SMILES must be provided for ligand residue {code}.")
            ref_mol = Component.from_smiles(code, smiles)
        else:
            # Residue not found in CCD
            # NOTE: For polymers, this should not happen due to prior conversion to UNK.
            raise ValueError(f"Residue {code} not found in CCD database.")

        is_res_standards.append(code in standard_residues)
        ref_mols.append(ref_mol)
        ref_atom_names = get_ref_atom_names(code)
        residue_atoms_list.append(ref_atom_names)
        num_residue_atoms.append(len(ref_atom_names))

    # Override atom names for residues if specified
    if valid_atom_names is not None:
        for res_idx, valid_atoms in valid_atom_names.items():
            res_i = res_idx - 1
            ref_atom_names = residue_atoms_list[res_i]
            ref_atom_names = tuple(v for v in ref_atom_names if v in valid_atoms)
            residue_atoms_list[res_i] = ref_atom_names
            num_residue_atoms[res_i] = len(ref_atom_names)

    residue_struct = Residue(
        name=np.array(ccd_sequences, dtype=np.dtype("<U6")),
        num_atoms=np.array(num_residue_atoms, dtype=np.uint8),
        is_standard=np.array(is_res_standards, dtype=bool),
    )

    # ==================================================
    # Prepare atom information
    # ==================================================
    atom_name_list: list[np.ndarray] = []
    atom_elem_list: list[np.ndarray] = []
    atom_charge_list: list[np.ndarray] = []

    for res_i, ref_mol in enumerate(ref_mols):
        res_idx = res_i + 1
        code = ref_mol.code
        assert code == ccd_sequences[res_i]

        # Get atom information for all atoms in the residue
        atom_names, atom_elem, atom_charge = get_ref_mol_info(code)

        if valid_atom_names is not None and res_idx in valid_atom_names:
            # Recompute atom information for specified atoms only
            res_atom_names = residue_atoms_list[res_i]
            atom_to_index = ref_mol.get_atom_index_map()
            atom_indices = [atom_to_index[atom_name] for atom_name in res_atom_names]
            atom_names = np.array(res_atom_names, dtype=np.dtype("<U4"))
            atom_elem = ref_mol.elements[atom_indices]
            atom_charge = ref_mol.charges[atom_indices]

        atom_name_list.append(atom_names)
        atom_elem_list.append(atom_elem)
        atom_charge_list.append(atom_charge)
    assert len(num_residue_atoms) == len(residue_atoms_list)

    # Empty coordinates, bfactors, apo coordinates, and apo pLDDT
    num_atoms = sum(num_residue_atoms)
    coords = np.full((num_atoms, 3), np.nan, dtype=np.float32)
    bfactors = np.full((num_atoms,), np.nan, dtype=np.float32)
    apo_coords = np.full((num_atoms, 3), np.nan, dtype=np.float32)
    apo_plddt = np.full((num_atoms), np.nan, dtype=np.float32)

    atom_struct = Atom(
        name=np.concatenate(atom_name_list, dtype=np.dtype("<U4")),
        element=np.concatenate(atom_elem_list, dtype=np.uint8),
        charge=np.concatenate(atom_charge_list, dtype=np.int8),
        coords=coords,
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
    if chain_type is C.ChainType.LIGAND:  # Only ligand bonds; there is no ion bonds.
        for residue_index, ref_mol in enumerate(ref_mols, start=1):
            # Get ref atom names
            ref_atom_names: tuple[str, ...] = residue_atoms_list[residue_index - 1]
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


def prepare_chain_metadata(chain: Chain, name: str) -> ChainInfo:
    """Prepare chain metadata.

    Parameters
    ----------
    chain : Chain
        The reference chain.
    name : str
        The user-defined chain name.

    Returns
    -------
    ChainInfo
        The prepared chain metadata.
    """
    return ChainInfo(
        name=name,
        type=chain.chain_type,
        entity_id=chain.entity_id,
        asym_id=chain.asym_id,
        sym_id=chain.sym_id,
        num_residues=chain.num_residues,
        num_atoms=chain.num_atoms,
        num_tokens=chain.num_tokens,
        smiles=chain.smiles,
    )
