"""Pipeline to prepare reference structures"""

import logging
from functools import lru_cache

import numpy as np

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.metadata import ChainInfo, Metadata
from kfold.data.types.structure import (
    AtomLayout,
    BondLayout,
    Chain,
    CovalentConnection,
    RefStructure,
    ResidueLayout,
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
    bonded_atoms: dict[int, set[str]] | None = None,
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
    bonded_atoms : dict[int, set[str]] | None, optional
        List of bonded atoms for covalent ligands (res_idx: atom_name), by default None.
    """
    # ==================================================
    # Validate inputs
    # ==================================================
    if smiles is not None:
        assert chain_type == C.ChainType.LIGAND
        assert len(ccd_sequences) == 1
        assert ccd_sequences[0].startswith("LIG")

    bonded_atoms = bonded_atoms or {}

    # Normalize residue names to uppercase
    ccd_sequences = [v.upper() for v in ccd_sequences]

    @lru_cache  # No cache limit within a single function call
    def get_ref_mol(ccd_code: str) -> Component:
        """Get reference molecule for a residue."""
        if ccd_code in ccd:
            return ccd[ccd_code]
        else:
            raise ValueError(f"Residue {ccd_code} not found in CCD database.")

    # ==================================================
    # Prepare residue and atom information
    # ==================================================
    standard_residues: set[str] = chain_type_to_standard_residues[chain_type]

    is_standard_list: list[bool] = []
    atom_name_list: list[np.ndarray] = []
    atom_elem_list: list[np.ndarray] = []
    atom_charge_list: list[np.ndarray] = []
    ref_mols: list[Component] = []

    for res_idx, code in enumerate(ccd_sequences, start=1):
        is_standard_list.append(code in standard_residues)
        if code.startswith("LIG"):
            # Ligand residue created from SMILES
            if smiles is None:
                raise ValueError(f"SMILES must be provided for ligand residue {code}.")
            ref_mol = Component.from_smiles(code, smiles)
        elif code in ccd:
            # Common molecule from CCD
            assert not code.startswith("LIG"), "LIG codes should be handled separately."
            ref_mol = get_ref_mol(code)
        else:
            # Residue not found in CCD
            # NOTE: For polymers, this should not happen due to prior conversion to UNK.
            raise ValueError(f"Residue {code} not found in CCD database.")

        if chain_type.is_polymer:
            # Return pre-defined atoms for standard polymer residues to
            # ensure consistency across different CCD versions. Otherwise,
            # use all non-leaving atoms.
            if code in standard_residues:
                atom_names = C.atom.residue_atoms[code]
            else:
                atom_names = ref_mol.get_atom_names(drop_leaving_atoms=True)
        else:
            # Special handling for glycans in covalent ligands
            res_bonded_atoms = bonded_atoms.get(res_idx, set())
            if code in C.ccd.GLYCANS:
                # Only retain oxygen if it is participating in the covalent bond
                atom_names = ref_mol.get_atom_names()
                if "O1" not in res_bonded_atoms:
                    atom_names = [n for n in atom_names if n != "O1"]
            else:
                # For common ligands, keep all atoms.
                # For covalent ligands, keep leaving atoms only if
                # any of them is involved in the covalent bond.
                is_covalent = len(res_bonded_atoms) > 0
                atom_names = ref_mol.get_atom_names(drop_leaving_atoms=is_covalent)
                if not res_bonded_atoms <= set(atom_names):
                    atom_names = ref_mol.get_atom_names()

        atom_indices: list[int] = ref_mol.get_atom_indices(atom_names)
        atom_name_list.append(np.array(atom_names, dtype=np.dtype("<U4")))
        atom_elem_list.append(ref_mol.elements[atom_indices])
        atom_charge_list.append(ref_mol.charges[atom_indices])
        ref_mols.append(ref_mol)

    num_res_atoms = [arr.shape[0] for arr in atom_name_list]
    residue_struct = ResidueLayout(
        name=np.array(ccd_sequences, dtype=np.dtype("<U6")),
        num_atoms=np.array(num_res_atoms, dtype=np.uint8),
        is_standard=np.array(is_standard_list, dtype=bool),
    )
    num_atoms = sum(num_res_atoms)
    atom_struct = AtomLayout(
        name=np.concatenate(atom_name_list, dtype=np.dtype("<U4")),
        element=np.concatenate(atom_elem_list, dtype=np.uint8),
        charge=np.concatenate(atom_charge_list, dtype=np.int8),
        # Empty coordinates, bfactors, apo coordinates, and apo pLDDT
        coords=np.full((num_atoms, 3), np.nan, dtype=np.float32),
        bfactor=np.full((num_atoms,), np.nan, dtype=np.float16),
        apo_coords=np.full((num_atoms, 3), np.nan, dtype=np.float32),
        apo_plddt=np.full((num_atoms), np.nan, dtype=np.float16),
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
            ref_atom_names = set(atom_name_list[residue_index - 1].tolist())
            for (atom_name1, atom_name2), bond_type in ref_mol.bonds.items():
                if atom_name1 in ref_atom_names and atom_name2 in ref_atom_names:
                    bond_residue_index_list.append((residue_index, residue_index))
                    bond_atom_name_list.append((atom_name1, atom_name2))
                    bond_type_list.append(bond_type)

    bond_struct = BondLayout(
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
        smiles=smiles,
        is_covalent_ligand=chain_type.is_ligand and len(bonded_atoms) > 0,
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
        is_covalent_ligand=chain.is_covalent_ligand,
        is_ion=chain.is_ion,
        # placeholders
        description=None,
        cluster_id=None,
        is_low_homology=False,
    )
