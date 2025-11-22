"""MMCIF writer utilities."""

import io
from collections.abc import Iterator
from functools import lru_cache

import ihm
import modelcif
import numpy as np
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem

import kfold.constants as C
from kfold.data.structure import TokenizedStructure

# === Simple wrappers === #
def to_mmcifstring_apo(
    structure: TokenizedStructure,
    conformer_id: int = 0,
) -> str:
    return to_mmcifstring(
        structure, conformer_id=conformer_id, is_predicted=False, save_apo=True
    )

# === Core implementation === #
@lru_cache(maxsize=1)
def _get_periodic_table() -> Chem.PeriodicTable:
    return Chem.GetPeriodicTable()


def to_mmcifstring(
    structure: TokenizedStructure,
    conformer_id: int = 0,
    is_predicted: bool = True,
    save_apo: bool = False,
) -> str:  # noqa: PLR0915
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    structure : TokenizedStructure
        The input structure
    conformer_id : int, optional
        The conformer ID to write (default is 0)
    save_apo : bool, optional
        Whether to save the apo form (default is False)
    is_predicted : bool, optional
        Whether the structure is predicted (default is False)

    Returns
    -------
    str
        the output MMCIF file
    """
    tokens = structure.token # [Ntoken, ...]
    atoms = structure.atom # [Ntoken, 24, ...]

    # Atom informations
    ref_atom_name_chars = atoms.ref_atom_name_chars # [Ntoken, 24, max_name_length]
    ref_atom_elements = atoms.ref_element # [Ntoken, 24]
    # TODO: ref_atom_charge


    # Select coordinates and mask based on the mode
    # atom_coords: [Ntoken, 24, 3]
    # atom_mask: [Ntoken, 24]
    if save_apo:
        # NOTE: ignore `is_predicted` flag when saving apo
        atom_coords = atoms.apo_coords[:, :, conformer_id, :]
        atom_mask = atoms.apo_mask[:, :, conformer_id]
    elif is_predicted:
        atom_coords = atoms.coords[:, :, conformer_id, :]
        atom_mask = np.ones_like(atoms.resolved_mask)
    else:
        atom_coords = atoms.coords[:, :, conformer_id, :]
        atom_mask = atoms.resolved_mask

    # Load periodic table for element mapping
    periodic_table = _get_periodic_table()

    # Initialize System
    system = System()

    # --- 1. Identify Entities (Unique Sequences) ---
    # In kfold, entity_id groups tokens that share the same sequence/molecule type.
    entity_map = {}  # entity_id -> ihm.Entity
    
    unique_entity_ids = np.unique(tokens.entity_id)
    for ent_id in unique_entity_ids:
        # Find indices for this entity
        ent_mask = tokens.entity_id == ent_id
        ent_indices = np.where(ent_mask)[0]
        
        # Get representative sequence from the first chain of this entity
        first_idx = ent_indices[0]
        first_asym_id = tokens.asym_id[first_idx]
        
        # Filter tokens for this entity AND this chain to get the unique sequence
        chain_mask = (tokens.asym_id == first_asym_id) & ent_mask
        chain_indices = np.where(chain_mask)[0]
        
        # Extract sequence
        sequence = []
        for idx in chain_indices:
            res_name = C.residue.residue_index_to_name[tokens.res_type[idx]].name
            sequence.append(res_name)
            
        # Determine MolType
        chain_type = tokens.chain_type[first_idx]
        
        # Create ChemComp and Alphabet based on chain_type
        if chain_type == C.chain.ChainType.PROTEIN:
            alphabet = ihm.LPeptideAlphabet()
            chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")
        elif chain_type == C.chain.ChainType.DNA:
            alphabet = ihm.DNAAlphabet()
            chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")
        elif chain_type == C.chain.ChainType.RNA:
            alphabet = ihm.RNAAlphabet()
            chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")
        elif len(sequence) > 1:
            # Polysaccharide or other polymer
            alphabet = {}
            chem_comp = lambda x: ihm.SaccharideChemComp(id=x)
        else:
            # Ligand / Non-polymer
            alphabet = {}
            chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)

        # Handle Ligands
        if chain_type == C.chain.ChainType.LIGAND:
             seq_objs = [chem_comp(item) for item in sequence]
             entity = Entity(seq_objs, description=f"Ligand {ent_id}") # add description
        else:
            seq_objs = [
                alphabet[item] if item in alphabet else chem_comp(item)
                for item in sequence
            ]
            entity = Entity(seq_objs, description=f"Polymer {ent_id}")
            
        entity_map[ent_id] = entity 

    # --- 2. Create AsymUnits (Chains) ---
    asym_unit_map = {} # asym_id -> ihm.AsymUnit
    
    unique_asym_ids = np.unique(tokens.asym_id)
    # Generate chain tag helper
    def _get_chain_tag(idx: int) -> str:
        chars = []
        while True:
            idx, rem = divmod(idx, 26)
            chars.append(chr(65 + rem))
            if idx == 0:
                break
            idx -= 1
        return "".join(reversed(chars))

    for asym_id in unique_asym_ids:
        # Get entity_id for this chain
        chain_mask = tokens.asym_id == asym_id
        first_token_idx = np.where(chain_mask)[0][0]
        ent_id = tokens.entity_id[first_token_idx]
        
        entity = entity_map[ent_id]
        
        # Generate chain tag
        # mmCIF supports arbitrary string IDs. We generate A-Z, AA-ZZ, etc.
        # asym_id is 1-based in kfold.
        chain_tag = _get_chain_tag(asym_id - 1)

        # Handle Water Entities
        # If the entity represents water (e.g. HOH), ihm detects it and sets entity.type to 'water'.
        if entity.type == "water":
            asym = ihm.WaterAsymUnit(
                entity,
                1, # count, usually 1 for asym unit
                details=f"Model subunit {chain_tag}",
                id=chain_tag,
            )
        else:
            asym = AsymUnit(
                entity,
                details=f"Model subunit {chain_tag}",
                id=chain_tag,
            )
        asym_unit_map[asym_id] = asym

    modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

    # --- 3. Define Model Class ---
    class _KfoldModel(AbInitioModel):
        def get_atoms(self) -> Iterator[Atom]:
            # Iterate over all tokens and atoms
            for i in range(len(tokens)):
                # Get current chain and residue info
                asym_id = tokens.asym_id[i]
                asym_unit = asym_unit_map[asym_id]
                
                residue_index = tokens.residue_index[i]
                
                # Check if HETATM
                chain_type = tokens.chain_type[i]
                is_hetatm = chain_type == C.chain.ChainType.LIGAND
                
                num_atoms = tokens.num_atoms[i]
                
                for j in range(num_atoms):
                    if not atom_mask[i, j]:
                        continue

                    # Atom Name
                    atom_name_chars = ref_atom_name_chars[i, j]
                    atom_name = "".join([chr(c + 32) for c in atom_name_chars if c != 0])
                    
                    # Element
                    element_idx = int(ref_atom_elements[i, j])
                    element = periodic_table.GetElementSymbol(element_idx).upper()
                    
                    # Coords
                    pos = atom_coords[i, j]
                    
                    # B-factor (Fixed to 1.0 (no plddt factor))
                    biso = 1.00
                    
                    yield Atom(
                        asym_unit=asym_unit,
                        type_symbol=element,
                        seq_id=residue_index,
                        atom_id=atom_name,
                        x=f"{pos[0]:.5f}",
                        y=f"{pos[1]:.5f}",
                        z=f"{pos[2]:.5f}",
                        het=is_hetatm,
                        biso=biso,
                        occupancy=1.00,
                    )

    # --- 4. Write Output ---
    model = _KfoldModel(assembly=modeled_assembly, name="Model")
    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)
    
    # Disable line wrapping for cleaner output
    ihm.dumper.set_line_wrap(False)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return fh.getvalue()