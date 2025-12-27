import dataclasses
import itertools
import logging
import pathlib
from typing import Any

import numpy as np

import kfold.constants as C
from kfold.data import apo_perturbation, featurize, metadata, model_input, structure
from kfold.data.processing.component import CCD, Component
from kfold.utils.files import load_apo_chain

from . import query

logger = logging.getLogger(__name__)


class InputDataPipeline:
    def __init__(
        self,
        ccd: CCD,
        seq_embedding_dim: int | None,
        struct_embedding_dim: int | None,
        seed: int = 1,
    ) -> None:
        self.ccd: CCD = ccd
        self.seed: int = seed

        # Initialize apo perturbation if needed
        self.apo_perturbation: apo_perturbation.ApoPerturbation = (
            apo_perturbation.ApoPerturbation(
                use_perturbation=False,
                use_random_rotation=True,
                use_symmetry_correction=False,
            )
        )

        # Initialize featurizer
        self.featurizer: featurize.InputFeaturizer = featurize.InputFeaturizer(
            seq_embedding_dim=seq_embedding_dim,
            struct_embedding_dim=struct_embedding_dim,
            augment_ref_pos=True,
        )

    def process_input_file(
        self, input_file: query.InputFile
    ) -> tuple[
        structure.TokenizedStructure,
        model_input.FoldingInput,
    ]:
        """Process an InputFile into model-ready inputs.

        Parameters
        ----------
        input_file : InputFile
            The input file containing sequences and metadata.

        Returns
        -------
        struct : TokenizedStructure
            The tokenized structure representation.
        f_input : FoldingInput
            The featurized model input.
        """
        rng = np.random.default_rng(self.seed)

        # Prepare structure from input file
        struct = self.prepare_structure_from_input_file(input_file)

        # Add metadata to preserve the chain ids
        struct.metadata = self.prepare_metadata_from_input_file(input_file)

        # Random rotation/translation.
        struct = self.apo_perturbation(struct, rng=rng)

        # Featurize input
        seq_emb_paths = self.collect_precomputed_embeddings(input_file, "sequence")
        struct_emb_paths = self.collect_precomputed_embeddings(input_file, "structure")
        f_input = self.featurizer(struct, seq_emb_paths, struct_emb_paths, rng=rng)
        return struct, f_input

    def prepare_structure_from_input_file(
        self,
        input_file: query.InputFile,
        rng: np.random.Generator | None = None,
    ) -> structure.TokenizedStructure:
        """Prepare the tokenized structure from the input file.

        Parameters
        ----------
        input_file : InputFile
            The input query file.

        Returns
        -------
        struct : TokenizedStructure
            The tokenized structure representation.
        """
        chain_structs: list[structure.TokenizedStructure] = []
        entity_id_iter = itertools.count(1)
        asym_id_iter = itertools.count(1)
        for seq in input_file.sequences:
            # Parse sequence
            if isinstance(seq, query.LigandSequence):
                entity_struct = parse_ligand_sequence(seq, self.ccd, rng=rng)
            else:
                entity_struct = parse_polymer_sequence(seq, self.ccd, rng=rng)

            # Update entity ids
            entity_id = next(entity_id_iter)
            entity_struct.chain.entity_id.fill(entity_id)
            entity_struct.residue.entity_id.fill(entity_id)
            entity_struct.token.entity_id.fill(entity_id)

            # Clone for multiple chains if needed and update asym_id/sym_id
            num_chains = 1 if isinstance(seq.id, str) else len(seq.id)
            sym_id_iter = itertools.count(1)
            for i in range(num_chains):
                # Copy sequence structure for multiple chains if needed
                asym_id = next(asym_id_iter)
                sym_id = next(sym_id_iter)
                if i == num_chains - 1:
                    chain_struct = entity_struct
                else:
                    chain_struct = entity_struct.copy_with(deepcopy=True)
                chain_struct.chain.asym_id.fill(asym_id)
                chain_struct.chain.sym_id.fill(sym_id)
                chain_struct.residue.asym_id.fill(asym_id)
                chain_struct.residue.sym_id.fill(sym_id)
                chain_struct.token.asym_id.fill(asym_id)
                chain_struct.token.sym_id.fill(sym_id)
                chain_struct.bond.asym_id.fill(asym_id)

                chain_structs.append(chain_struct)

        # TODO: add constraints if needed

        # Concatenate all chains into a single TokenizedStructure
        struct = structure.TokenizedStructure.concatenate(chain_structs)
        return struct

    def prepare_metadata_from_input_file(
        self, input_file: query.InputFile
    ) -> metadata.Metadata:
        """Prepare the metadata from the input file.

        Parameters
        ----------
        input_file : InputFile
            The input query file.

        Returns
        -------
        metadata : Metadata
            The metadata representation.
        """
        # parse chain info
        entity_id_iter = itertools.count(1)
        asym_id_iter = itertools.count(1)
        chain_infos: list[metadata.ChainInfo] = []
        for seq in input_file.sequences:
            entity_id = next(entity_id_iter)
            chain_type = seq.ctype
            chain_names = [seq.id] if isinstance(seq.id, str) else seq.id
            num_residues = len(seq)
            sym_id_iter = itertools.count(1)
            for chain_name in chain_names:
                asym_id = next(asym_id_iter)
                sym_id = next(sym_id_iter)
                chain_info = metadata.ChainInfo(
                    chain_type=chain_type,
                    chain_name=chain_name,
                    entity_id=entity_id,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    num_residues=num_residues,
                    cluster_id="",
                    valid=True,
                    description=seq.description,
                )
                chain_infos.append(chain_info)

        meta = metadata.Metadata(
            id=input_file.name,
            source="query",
            chains=chain_infos,
        )
        return meta

    def collect_precomputed_embeddings(
        self, input_file: query.InputFile, key: str = "sequence"
    ) -> dict[int, pathlib.Path]:
        """Collect precomputed embeddings from the input file.

        Parameters
        ----------
        input_file : InputFile
            The input query file.

        Returns
        -------
        embedding_paths : dict[int, pathlib.Path]
            A dictionary mapping entity ids to embedding file paths.
        """
        assert key in {"sequence", "structure"}, (
            "Key must be either 'sequence' or 'structure'."
        )
        embedding_paths: dict[int, pathlib.Path] = {}
        entity_id_iter = itertools.count(1)
        for seq in input_file.sequences:
            entity_id = next(entity_id_iter)
            if key == "sequence" and seq.seq_emb is not None:
                embedding_paths[entity_id] = pathlib.Path(seq.seq_emb)
            elif key == "structure" and seq.struct_emb is not None:
                embedding_paths[entity_id] = pathlib.Path(seq.struct_emb)
        for file in embedding_paths.values():
            if not file.exists():
                raise FileNotFoundError(f"Precomputed embedding file not found: {file}")
        return embedding_paths


# ================================================================================
# CCD Residue Parsing Utilities
# ================================================================================


def parse_standard_residue(
    component: Component,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Parse a standard residue from the CCD component.

    Parameters
    ----------
    component : Component
        The CCD component representing the residue.
    rng : np.random.Generator | None, optional
        Random number generator for any stochastic processes. Default is None.

    Returns
    -------
    dict[str, Any]
        A dictionary containing atom names, elements, charges, conformers
    """

    # Get the atom names with standard order
    assert component.code in C.ResidueName.__members__, (
        f"Residue {component.code} is not a standard residue."
    )
    res_name: C.ResidueName = C.ResidueName[component.code]
    ref_atom_names: list[str] = [v.value for v in C.atom.RESIDUE_ATOMS[res_name]]

    # Get the reference conformer of component
    conf = component.get_conformer("auto", rng=rng)
    assert conf is not None, f"No conformer found for residue {res_name}."

    # Reorder the conformer to match the standard residue atom order
    ref_elements: list[int] = []
    ref_charges: list[float] = []
    ref_coords: list[np.ndarray] = []

    for atom_name in ref_atom_names:
        atom_idx: int = component.atom_names.index(atom_name)
        ref_elements.append(component.elements[atom_idx])
        ref_charges.append(component.charges[atom_idx])
        ref_coords.append(conf[atom_idx])

    ref_atom_name_chars = [C.atom.encode_atom_name(name) for name in ref_atom_names]

    return {
        "atom_name": ref_atom_names,
        "atom_name_chars": np.array(ref_atom_name_chars, dtype=np.int8),
        "element": np.array(ref_elements, dtype=np.int8),
        "charge": np.array(ref_charges, dtype=np.float16),
        "pos": np.array(ref_coords, dtype=np.float32),
    }


def parse_non_standard_residue(
    component: Component,
    include_leaving_atoms: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Parse a non-standard residue (e.g., PTM, ligand) from the CCD component.

    Parameters
    ----------
    component : Component
        The CCD component representing the molecule.
    include_leaving_atoms : bool, optional
        Whether to include leaving atoms in the output. Default is False.
    rng : np.random.Generator | None, optional
        Random number generator for any stochastic processes. Default is None.

    Returns
    -------
    dict[str, Any]
        A dictionary containing atom names, elements, charges, conformers, and bonds
    """
    ref_atom_names: list[str] = []
    ref_elements: list[int] = []
    ref_charges: list[float] = []
    ref_coords: list[np.ndarray] = []
    ref_bonds: list[tuple[int, int]] = []
    ref_bond_types: list[int] = []

    conf = component.get_conformer("auto", rng=rng)
    assert conf is not None, f"No conformer found for molecule {component.code}."

    idx_map: dict[int, int] = {}
    for i, atom_name in enumerate(component.atom_names):
        if not include_leaving_atoms and component.is_leaving_atom[i]:
            continue
        idx_map[i] = len(ref_atom_names)
        ref_atom_names.append(atom_name)
        ref_elements.append(component.elements[i])
        ref_charges.append(component.charges[i])
        ref_coords.append(conf[i])

    # Get bonds with re-indexed atom indices
    for bond in component.mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in idx_map and end_idx in idx_map:
            ref_bonds.append((idx_map[begin_idx], idx_map[end_idx]))
            ref_bond_types.append(
                C.bond.bond_type_to_connection_type(bond.GetBondType()).value
            )

    ref_atom_name_chars = [C.atom.encode_atom_name(name) for name in ref_atom_names]

    return {
        "atom_name": ref_atom_names,
        "atom_name_chars": np.array(ref_atom_name_chars, dtype=np.int8),
        "element": np.array(ref_elements, dtype=np.int8),
        "charge": np.array(ref_charges, dtype=np.float16),
        "pos": np.array(ref_coords, dtype=np.float32),
        "bonds": np.array(ref_bonds, dtype=np.int32),
        "bond_types": np.array(ref_bond_types, dtype=np.int8),
    }


# ================================================================================
# Chain Parsing Functions
# ================================================================================


def parse_polymer_sequence(
    seq: query.PolymerSequence,
    ccd: CCD,
    rng: np.random.Generator | None = None,
) -> structure.TokenizedStructure:
    """Parse a polymer chain from the sequence input.

    Parameters
    ----------
    seq : PolymerSequence
        The polymer sequence input.
    ccd : CCD
        The CCD component.

    Returns
    -------
    struct : TokenizedStructure
        The tokenized structure representation.

    Notes
    -----
    The chain ids (entity_id, asym_id, sym_id) are all set to placeholder (zero)
    """
    # Determine chain type
    ctype = seq.ctype
    ctype_value = ctype.value

    # Set placeholder (they are 1-based index)
    entity_id = asym_id = sym_id = 0

    # Initialize data dictionaries
    residue_dict: dict[str, list] = {
        field.name: [] for field in dataclasses.fields(structure.Residue)
    }
    token_dict: dict[str, list] = {
        field.name: [] for field in dataclasses.fields(structure.Token)
    }
    atom_dict: dict[str, list] = {
        field.name: [] for field in dataclasses.fields(structure.Atom)
    }
    bond_dict: dict[str, list] = {
        field.name: [] for field in dataclasses.fields(structure.Bond)
    }

    # Load sequence and modifications
    sequence: str = seq.sequence
    modifications: dict[int, str] = {int(k): v for k, v in seq.modifications.items()}

    # Load apo structure if provided
    if seq.apo is not None:
        # TODO: we may want to use multiple apo structures in the future
        _, apo_atom_coordinates = load_apo_chain(seq.apo)
    else:
        apo_atom_coordinates = {}

    token_idx = 0
    for res_idx, aa in enumerate(sequence, start=1):
        if res_idx in modifications:
            # FIXME: retain the original residue name for modified residues.
            ccd_code = modifications[res_idx]
            if ccd_code not in ccd:
                logger.warning(
                    f"Modified residue '{ccd_code}' at position {res_idx} "
                    "not found in CCD. Mapping to UNK."
                )
                ccd_code = "UNK"
            is_standard = False
        else:
            res_name = C.residue.map_one_letter_to_residue_name(aa, ctype)
            ccd_code = res_name.value
            is_standard = True

        res_name: C.ResidueName = C.residue.get_residue_name_with_unk(ccd_code, ctype)
        res_type: int = C.residue.residue_name_to_id[res_name]

        # Parse reference molecule
        ccd_residue_mol: Component = ccd[ccd_code]
        ref_mol_data: dict[str, np.ndarray]
        if is_standard:
            ref_mol_data = parse_standard_residue(ccd_residue_mol, rng=rng)
        else:
            ref_mol_data = parse_non_standard_residue(ccd_residue_mol, rng=rng)
        num_atoms = len(ref_mol_data["element"])

        # Get the residue coordinates from the apo structure if available
        if res_idx in apo_atom_coordinates:
            residue_apo_coords = apo_atom_coordinates[res_idx]
        else:
            residue_apo_coords = {}

        if is_standard:
            # Standard residue
            num_tokens = 1  # Standard residues map to one token

            residue_atoms: tuple[C.AtomName, ...] = C.atom.RESIDUE_ATOMS[res_name]
            assert num_atoms == len(residue_atoms)  # Always true

            # Add Token-level data
            token_dict["res_type"].append(res_type)
            token_dict["chain_type"].append(ctype_value)
            token_dict["entity_id"].append(entity_id)
            token_dict["asym_id"].append(asym_id)
            token_dict["sym_id"].append(sym_id)
            token_dict["token_index"].append(token_idx)
            token_dict["residue_index"].append(res_idx)
            token_dict["num_atoms"].append(num_atoms)
            token_dict["disto_index"].append(
                residue_atoms.index(C.atom.REF_ATOM[res_name])
            )
            token_dict["center_index"].append(
                residue_atoms.index(C.atom.PSEUDO_BETA_ATOM[res_name])
            )
            token_dict["resolved_mask"].append(True)
            token_dict["is_standard"].append(True)

            # Add Atom-level data
            ref_atom_name_chars = np.zeros((24, 4), dtype=np.uint8)
            ref_element = np.zeros(24, dtype=np.uint8)
            ref_charge = np.zeros(24, dtype=np.float16)
            ref_pos = np.full((24, 3), fill_value=np.nan, dtype=np.float32)
            coords = np.full((24, 3), fill_value=np.nan, dtype=np.float32)
            apo_coords = np.full((24, 3), fill_value=np.nan, dtype=np.float32)
            apo_mask = np.zeros(24, dtype=np.bool_)
            pad_mask = np.zeros(24, dtype=np.bool_)

            ref_atom_name_chars[:num_atoms] = ref_mol_data["atom_name_chars"]
            ref_element[:num_atoms] = ref_mol_data["element"]
            ref_charge[:num_atoms] = ref_mol_data["charge"]
            ref_pos[:num_atoms] = ref_mol_data["pos"]
            pad_mask[:num_atoms] = True

            # Add apo coordinates if available
            for i, atom_name in enumerate(residue_atoms):
                atom_name_str = atom_name.value
                if atom_name_str in residue_apo_coords:
                    apo_coords[i] = residue_apo_coords[atom_name_str]
                    apo_mask[i] = True

            atom_dict["ref_atom_name_chars"].append(ref_atom_name_chars)
            atom_dict["ref_element"].append(ref_element)
            atom_dict["ref_charge"].append(ref_charge)
            atom_dict["ref_pos"].append(ref_pos)
            atom_dict["pad_mask"].append(pad_mask)
            atom_dict["resolved_mask"].append(pad_mask)  # Use pad_mask as resolved_mask
            atom_dict["coords"].append(coords)
            atom_dict["apo_coords"].append(apo_coords)
            atom_dict["apo_mask"].append(apo_mask)
        else:
            # Non-standard residue: each atom becomes a token
            num_tokens = num_atoms

            token_st = token_idx  # Starting token index for this residue

            # Add Token-level data
            token_dict["res_type"].extend([res_type] * num_tokens)
            token_dict["chain_type"].extend([ctype_value] * num_tokens)
            token_dict["entity_id"].extend([entity_id] * num_tokens)
            token_dict["asym_id"].extend([asym_id] * num_tokens)
            token_dict["sym_id"].extend([sym_id] * num_tokens)
            token_dict["token_index"].extend(range(token_st, token_st + num_tokens))
            token_dict["residue_index"].extend([res_idx] * num_tokens)
            token_dict["num_atoms"].extend([1] * num_tokens)
            token_dict["disto_index"].extend([0] * num_tokens)
            token_dict["center_index"].extend([0] * num_tokens)
            token_dict["resolved_mask"].extend([True] * num_tokens)
            token_dict["is_standard"].extend([False] * num_tokens)

            for atom_i in range(num_atoms):
                atom_name = ref_mol_data["atom_name"][atom_i]

                # Add Atom-level data
                ref_atom_name_chars = np.zeros((24, 4), dtype=np.uint8)
                ref_element = np.zeros(24, dtype=np.uint8)
                ref_charge = np.zeros(24, dtype=np.float16)
                ref_pos = np.full((24, 3), fill_value=np.nan, dtype=np.float32)
                coords = np.full((24, 3), fill_value=np.nan, dtype=np.float32)
                apo_coords = np.full((24, 3), fill_value=np.nan, dtype=np.float32)
                apo_mask = np.zeros(24, dtype=np.bool_)
                pad_mask = np.zeros(24, dtype=np.bool_)

                ref_atom_name_chars[0] = ref_mol_data["atom_name_chars"][atom_i]
                ref_element[0] = ref_mol_data["element"][atom_i]
                ref_charge[0] = ref_mol_data["charge"][atom_i]
                ref_pos[0] = ref_mol_data["pos"][atom_i]
                pad_mask[0] = True

                # Load apo coordinates if available
                if atom_name in residue_apo_coords:
                    apo_coords[0] = residue_apo_coords[atom_name]
                    apo_mask[0] = True

                atom_dict["ref_atom_name_chars"].append(ref_atom_name_chars)
                atom_dict["ref_element"].append(ref_element)
                atom_dict["ref_charge"].append(ref_charge)
                atom_dict["ref_pos"].append(ref_pos)
                atom_dict["coords"].append(coords)
                atom_dict["apo_coords"].append(apo_coords)
                atom_dict["resolved_mask"].append(pad_mask)  # Use pad_mask
                atom_dict["apo_mask"].append(apo_mask)
                atom_dict["pad_mask"].append(pad_mask)

            # Add bond information
            bonds = ref_mol_data["bonds"]
            bond_types = ref_mol_data["bond_types"]
            for bond_i in range(len(bonds)):
                atom_1, atom_2 = bonds[bond_i]
                token_1, token_2 = token_st + int(atom_1), token_st + int(atom_2)
                bond_type = bond_types[bond_i]
                bond_dict["asym_id"].append((0, 0))
                bond_dict["token_index"].append((token_1, token_2))
                bond_dict["atom_index"].append((0, 0))
                bond_dict["bond_type"].append(bond_type)

        # Add residue-level data
        residue_dict["chain_type"].append(ctype_value)
        residue_dict["name"].append(np.array(ccd_code, dtype="<U5"))
        residue_dict["res_type"].append(res_type)
        residue_dict["entity_id"].append(entity_id)
        residue_dict["asym_id"].append(asym_id)
        residue_dict["sym_id"].append(sym_id)
        residue_dict["residue_index"].append(res_idx)
        residue_dict["num_tokens"].append(num_tokens)
        residue_dict["num_atoms"].append(num_atoms)
        residue_dict["resolved_mask"].append(True)  # Placeholder (not used in inference)
        residue_dict["is_standard"].append(is_standard)

        # Increment token index
        token_idx += num_tokens

    # Convert lists to numpy arrays
    chain_dtypes = structure.Chain.get_default_dtype()
    res_dtypes = structure.Residue.get_default_dtype()
    token_dtypes = structure.Token.get_default_dtype()
    atom_dtypes = structure.Atom.get_default_dtype()
    bond_dtypes = structure.Bond.get_default_dtype()

    residue_arr: dict[str, np.ndarray] = {
        k: np.array(residue_dict[k], dtype=res_dtypes[k]) for k in residue_dict
    }
    token_arr: dict[str, np.ndarray] = {
        k: np.array(token_dict[k], dtype=token_dtypes[k]) for k in token_dict
    }
    atom_arr: dict[str, np.ndarray] = {
        k: np.stack(atom_dict[k], dtype=atom_dtypes[k]) for k in atom_dict
    }
    bond_arr: dict[str, np.ndarray] = {
        k: np.array(bond_dict[k], dtype=bond_dtypes[k]) for k in bond_dict
    }

    # Fill nans in atom coordinates with zeros
    # TODO: Remove these lines and handle nans properly in future steps (apo perturbation)
    atom_arr["apo_coords"] = np.nan_to_num(atom_arr["apo_coords"], nan=0.0)
    atom_arr["ref_pos"] = np.nan_to_num(atom_arr["ref_pos"], nan=0.0)

    # Add Napo/Nholo dimension (N, 24, 1, 3) / (N, 24, 1)
    atom_arr["coords"] = np.expand_dims(atom_arr["coords"], axis=-2)
    atom_arr["apo_coords"] = np.expand_dims(atom_arr["apo_coords"], axis=-2)
    atom_arr["apo_mask"] = np.expand_dims(atom_arr["apo_mask"], axis=-1)

    # Ensure bond indices are in correct shape
    bond_arr["asym_id"] = bond_arr["asym_id"].reshape(-1, 2)
    bond_arr["token_index"] = bond_arr["token_index"].reshape(-1, 2)
    bond_arr["atom_index"] = bond_arr["atom_index"].reshape(-1, 2)

    residue_layout = structure.Residue(**residue_arr)
    token_layout = structure.Token(**token_arr)
    atom_layout = structure.Atom(**atom_arr)
    bond_layout = structure.Bond(**bond_arr)

    # Construct chain-level data
    chain_dict = {
        "chain_type": [ctype_value],
        "entity_id": [entity_id],
        "asym_id": [asym_id],
        "sym_id": [sym_id],
        "num_residues": [len(residue_layout)],
        "num_tokens": [len(token_layout)],
        "num_atoms": [np.sum(residue_layout.num_atoms)],
    }
    chain_layout = structure.Chain(
        **{k: np.array(chain_dict[k], dtype=chain_dtypes[k]) for k in chain_dict}
    )

    # Construct TokenizedStructure
    struct = structure.TokenizedStructure(
        chain=chain_layout,
        residue=residue_layout,
        token=token_layout,
        atom=atom_layout,
        bond=bond_layout,
    )

    return struct


def parse_ligand_sequence(
    seq: query.LigandSequence,
    ccd: CCD,
    rng: np.random.Generator | None = None,
) -> structure.TokenizedStructure:
    """Parse a polymer chain from the sequence input.

    Parameters
    ----------
    seq : LigandSequence
        The ligand sequence input.
    ccd : CCD
        The CCD component.

    Returns
    -------
    struct : TokenizedStructure
        The tokenized structure representation.

    Notes
    -----
    The chain ids (entity_id, asym_id, sym_id) are all set to placeholder (zero)
    """
    # Determine chain type
    ctype: C.ChainType = C.ChainType.LIGAND
    ctype_value = ctype.value

    # Determine residue type (Protein UNK for ligands)
    res_type: int = C.residue.residue_name_to_id[C.ResidueName.UNK]

    # Set placeholder (they are 1-based index)
    entity_id = asym_id = sym_id = 0

    # Parse residues
    residue_dict: dict[str, list] = {
        field.name: [] for field in dataclasses.fields(structure.Residue)
    }
    token_dict: dict[str, list] = {
        field.name: [] for field in dataclasses.fields(structure.Token)
    }
    atom_dict: dict[str, list] = {
        field.name: [] for field in dataclasses.fields(structure.Atom)
    }
    bond_dict: dict[str, list] = {
        field.name: [] for field in dataclasses.fields(structure.Bond)
    }

    # Load ccd or smiles
    components: list[Component] = []
    if seq.ccd_ids is not None:
        for code in seq.ccd_ids:
            if code not in ccd:
                raise ValueError(f"CCD code '{code}' not found in CCD.")
            components.append(ccd[code])
    else:
        assert seq.smiles is not None, "Either CCD code or SMILES must be provided."
        # NOTE: Using "LIG" as a placeholder code for ligands from SMILES
        # This will be replaced later during mmcif writing.
        components = [
            Component.from_smiles(
                code="LIG",
                smiles=seq.smiles,
                num_confs=1,
                rng=rng,
            )
        ]

    token_idx = 0
    for res_idx, lig_residue in enumerate(components, start=1):
        ccd_code = lig_residue.code

        # Parse reference molecule
        ref_mol_data: dict[str, np.ndarray]
        ref_mol_data = parse_non_standard_residue(
            lig_residue, include_leaving_atoms=True, rng=rng
        )
        num_atoms = len(ref_mol_data["element"])

        # Ligand: each atom becomes a token
        token_st = token_idx  # Starting token index for this residue
        num_tokens = num_atoms

        # Add token information
        token_dict["res_type"].extend([res_type] * num_tokens)
        token_dict["chain_type"].extend([ctype_value] * num_tokens)
        token_dict["entity_id"].extend([entity_id] * num_tokens)
        token_dict["asym_id"].extend([asym_id] * num_tokens)
        token_dict["sym_id"].extend([sym_id] * num_tokens)
        token_dict["token_index"].extend(range(token_st, token_st + num_tokens))
        token_dict["residue_index"].extend([res_idx] * num_tokens)
        token_dict["num_atoms"].extend([1] * num_tokens)
        token_dict["disto_index"].extend([0] * num_tokens)
        token_dict["center_index"].extend([0] * num_tokens)
        token_dict["resolved_mask"].extend([True] * num_tokens)
        token_dict["is_standard"].extend([False] * num_tokens)

        # NOTE: Allocating each atom as the first atom in the token
        ref_atom_name_chars = np.zeros((num_atoms, 24, 4), dtype=np.uint8)
        ref_element = np.zeros((num_atoms, 24), dtype=np.uint8)
        ref_charge = np.zeros((num_atoms, 24), dtype=np.float16)
        ref_pos = np.full((num_atoms, 24, 3), fill_value=np.nan, dtype=np.float32)
        coords = np.full((num_atoms, 24, 3), fill_value=np.nan, dtype=np.float32)
        pad_mask = np.zeros((num_atoms, 24), dtype=np.bool_)

        ref_atom_name_chars[:, 0, :] = ref_mol_data["atom_name_chars"]
        ref_element[:, 0] = ref_mol_data["element"]
        ref_charge[:, 0] = ref_mol_data["charge"]
        ref_pos[:, 0, :] = ref_mol_data["pos"]
        pad_mask[:, 0] = True

        # Use apo coordinates as ref coordinates
        apo_coords = ref_pos.copy()
        apo_mask = pad_mask & np.isfinite(apo_coords).all(axis=-1)

        atom_dict["ref_atom_name_chars"].append(ref_atom_name_chars)
        atom_dict["ref_element"].append(ref_element)
        atom_dict["ref_charge"].append(ref_charge)
        atom_dict["ref_pos"].append(ref_pos)
        atom_dict["coords"].append(coords)
        atom_dict["apo_coords"].append(apo_coords)
        atom_dict["resolved_mask"].append(pad_mask)  # Use pad_mask as resolved_mask
        atom_dict["apo_mask"].append(apo_mask)
        atom_dict["pad_mask"].append(pad_mask)

        # Add bond information
        bonds = ref_mol_data["bonds"]
        bond_types = ref_mol_data["bond_types"]
        for bond_i in range(len(bonds)):
            atom_a, atom_b = bonds[bond_i]
            token_a, token_b = token_st + int(atom_a), token_st + int(atom_b)
            bond_type = bond_types[bond_i]
            bond_dict["asym_id"].append((0, 0))
            bond_dict["token_index"].append((token_a, token_b))
            bond_dict["atom_index"].append((0, 0))
            bond_dict["bond_type"].append(bond_type)

        # Add residue-level data
        residue_dict["chain_type"].append(ctype_value)
        residue_dict["name"].append(np.array(ccd_code, dtype="<U5"))
        residue_dict["res_type"].append(res_type)
        residue_dict["entity_id"].append(entity_id)
        residue_dict["asym_id"].append(asym_id)
        residue_dict["sym_id"].append(sym_id)
        residue_dict["residue_index"].append(res_idx)
        residue_dict["num_tokens"].append(num_tokens)
        residue_dict["num_atoms"].append(num_atoms)
        residue_dict["resolved_mask"].append(True)  # Placeholder (not used in inference)
        residue_dict["is_standard"].append(False)

        # Increment token index
        token_idx += num_tokens

    # Convert lists to numpy arrays
    chain_dtypes = structure.Chain.get_default_dtype()
    res_dtypes = structure.Residue.get_default_dtype()
    token_dtypes = structure.Token.get_default_dtype()
    atom_dtypes = structure.Atom.get_default_dtype()
    bond_dtypes = structure.Bond.get_default_dtype()

    residue_arr: dict[str, np.ndarray] = {
        k: np.array(residue_dict[k], dtype=res_dtypes[k]) for k in residue_dict
    }
    token_arr: dict[str, np.ndarray] = {
        k: np.array(token_dict[k], dtype=token_dtypes[k]) for k in token_dict
    }
    atom_arr: dict[str, np.ndarray] = {
        k: np.concatenate(atom_dict[k], dtype=atom_dtypes[k]) for k in atom_dict
    }
    bond_arr: dict[str, np.ndarray] = {
        k: np.array(bond_dict[k], dtype=bond_dtypes[k]) for k in bond_dict
    }

    # Add Napo/Nholo dimension (N, 24, 1, 3) / (N, 24, 1)
    atom_arr["coords"] = np.expand_dims(atom_arr["coords"], axis=-2)
    atom_arr["apo_coords"] = np.expand_dims(atom_arr["apo_coords"], axis=-2)
    atom_arr["apo_mask"] = np.expand_dims(atom_arr["apo_mask"], axis=-1)

    # Ensure bond indices are in correct shape
    bond_arr["asym_id"] = bond_arr["asym_id"].reshape(-1, 2)
    bond_arr["token_index"] = bond_arr["token_index"].reshape(-1, 2)
    bond_arr["atom_index"] = bond_arr["atom_index"].reshape(-1, 2)

    residue_layout = structure.Residue(**residue_arr)
    token_layout = structure.Token(**token_arr)
    atom_layout = structure.Atom(**atom_arr)
    bond_layout = structure.Bond(**bond_arr)

    # Construct chain-level data
    chain_dict = {
        "chain_type": [ctype_value],
        "entity_id": [entity_id],
        "asym_id": [asym_id],
        "sym_id": [sym_id],
        "num_residues": [len(residue_layout)],
        "num_tokens": [len(token_layout)],
        "num_atoms": [np.sum(residue_layout.num_atoms)],
    }
    chain_layout = structure.Chain(
        **{k: np.array(chain_dict[k], dtype=chain_dtypes[k]) for k in chain_dict}
    )

    # Construct TokenizedStructure
    struct = structure.TokenizedStructure(
        chain=chain_layout,
        residue=residue_layout,
        token=token_layout,
        atom=atom_layout,
        bond=bond_layout,
    )

    return struct
