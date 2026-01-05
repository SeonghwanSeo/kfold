"""MMCIF writer utilities."""

import io
import logging
from collections.abc import Callable, Generator
from functools import lru_cache

import ihm
import numpy as np
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem

import kfold.constants as C
from kfold.data.types.tokenized import TokenizedStructure

# Set up logging
logger = logging.getLogger(__name__)


# === Helpers === #
@lru_cache(maxsize=1)
def _get_periodic_table() -> Chem.PeriodicTable:
    return Chem.GetPeriodicTable()


@lru_cache(maxsize=4)
def _get_alphabet(chain_type: C.ChainType) -> ihm.Alphabet:
    match chain_type:
        case C.ChainType.PROTEIN:
            return ihm.LPeptideAlphabet()
        case C.ChainType.DNA:
            return ihm.DNAAlphabet()
        case C.ChainType.RNA:
            return ihm.RNAAlphabet()
        case _:
            raise ValueError(f"Unsupported chain type for alphabet: {chain_type}")


@lru_cache(maxsize=3)
def _get_polymer_chemcomp_factory(
    chain_type: C.ChainType,
) -> Callable[[str], ihm.ChemComp]:
    match chain_type:
        case C.ChainType.PROTEIN:
            return lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
        case C.ChainType.DNA:
            return lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        case C.ChainType.RNA:
            return lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        case _:
            raise ValueError(f"Unsupported chain type for chemcomp: {chain_type}")


@lru_cache(maxsize=2)
def _get_nonpolymer_chemcomp_factory(ligand_type: str) -> Callable[[str], ihm.ChemComp]:
    if ligand_type not in {"saccharide", "nonpolymer"}:
        raise ValueError(f"Unsupported ligand type for chemcomp: {ligand_type}")
    if ligand_type == "saccharide":
        return lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
    else:
        return lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731


def _get_chain_tag(index: int) -> str:
    """Generate chain tag from index (0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, ...)."""
    chars: list[str] = []
    while True:
        index, rem = divmod(index, 26)
        chars.append(chr(rem + ord("A")))
        if index == 0:
            break
        index -= 1  # Adjust for 0-based index
    return "".join(reversed(chars))


# === Core implementation === #
def to_mmcifstring(
    struct: TokenizedStructure,
    save_apo: bool = False,
) -> str:  # noqa: PLR0915
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    struct : TokenizedStructure
        The input structure
    save_apo : bool, optional
        Whether to save the apo form (default is False)

    Returns
    -------
    str
        the output MMCIF file
    """
    chains = struct.chain  # [Nchain, ...]
    residues = struct.residue  # [Nresidue, ...]
    tokens = struct.token  # [Ntoken, ...]
    atoms = struct.atom  # [Ntoken, 24, ...]

    # Get chain tags
    chain_tags: list[str] = []
    entity_descriptions: dict[int, str | None] = {}
    if struct.metadata is not None:
        # Use metadata if available
        if struct.metadata.num_chains == len(chains):
            for chain_info in struct.metadata.chains:
                chain_tags.append(chain_info.chain_name)
                entity_descriptions[int(chain_info.entity_id)] = chain_info.description
        else:
            logger.warning(
                "Metadata chain count does not match structure chain count. "
                "Falling back to default chain tags."
            )
    if len(chain_tags) == 0:
        # Fallback to default chain tags (A, B, C, ...)
        for i in range(len(chains)):
            chain_tags.append(_get_chain_tag(i))

    # Load periodic table for element mapping
    periodic_table = _get_periodic_table()

    # --- 1. Identify Entities (Unique Sequences) ---
    entity_map: dict[int, ihm.Entity] = {}
    ligand_count: int = 0

    for chain_i in range(len(chains)):
        # Chain info
        entity_id = int(chains.entity_id[chain_i])

        if entity_id in entity_map:
            continue  # already processed

        # Extract sequence
        res_st = int(chains.residue_start[chain_i])
        res_end = res_st + int(chains.num_residues[chain_i])
        sequence: list[str] = []
        for res_name in residues.name[res_st:res_end]:
            sequence.append(res_name)

        # Create ChemComp and Alphabet based on chain_type
        chain_type = C.ChainType(chains.chain_type[chain_i])
        if chain_type in {C.ChainType.PROTEIN, C.ChainType.DNA, C.ChainType.RNA}:
            # Handling polymer chains
            alphabet = _get_alphabet(chain_type)
            chem_comp = _get_polymer_chemcomp_factory(chain_type)
            seq_objs = [alphabet[v] if v in alphabet else chem_comp(v) for v in sequence]
        else:
            # Handling ligand / non-polymer chains
            if len(sequence) == 1 and sequence[0] == "LIG":
                # Ligand / Non-polymer
                ligand_count += 1
                lig_id = f"LIG{ligand_count}"
                seq_objs = [ihm.NonPolymerChemComp(id=lig_id)]
            else:
                # Polysaccharide or other polymer
                chem_comp = lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
                seq_objs = [chem_comp(v) for v in sequence]

        # Create Entity
        if (description := entity_descriptions.get(entity_id, None)) is None:
            description = f"{str(chain_type)} {entity_id}"
        entity = Entity(seq_objs, description=description)
        entity_map[entity_id] = entity

    # --- 2. Create AsymUnits (Chains) ---
    asym_unit_map: dict[int, ihm.AsymUnit] = {}

    for chain_i in range(len(chains)):
        # Get entity_id for this chain
        asym_id = int(chains.asym_id[chain_i])
        entity_id = int(chains.entity_id[chain_i])
        chain_tag = chain_tags[chain_i]
        asym = AsymUnit(
            entity=entity_map[entity_id],
            details=f"Model subunit {chain_tag}",
            id=chain_tag,
        )
        asym_unit_map[asym_id] = asym

    modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

    # --- 3. Create atom models ---
    # Select coordinates and mask based on the mode
    # atom_coords: [Ntoken, 24, 3]
    # atom_mask: [Ntoken, 24]
    if save_apo:
        atom_coords = atoms.apo_coords
    else:
        atom_coords = atoms.coords
    atom_mask = np.isfinite(atom_coords).all(axis=-1)

    class _KfoldModel(AbInitioModel):
        def get_atoms(self) -> Generator[Atom, None, None]:
            for t_i in range(len(tokens)):
                asym_unit = asym_unit_map[tokens.asym_id[t_i]]
                residue_index = tokens.residue_index[t_i]
                is_hetatm = tokens.is_ligand[t_i]

                for a_i in range(tokens.num_atoms[t_i]):
                    if not atom_mask[t_i, a_i]:
                        continue

                    # Atom Name
                    atom_name = C.atom.decode_atom_name(
                        atoms.ref_atom_name_chars[t_i, a_i]
                    )

                    # Element
                    element = periodic_table.GetElementSymbol(
                        int(atoms.ref_element[t_i, a_i])
                    ).upper()

                    # Coords
                    pos_x, pos_y, pos_z = atom_coords[t_i, a_i, :]

                    # B-factor (Fixed to 1.0 (no plddt factor))
                    biso = 1.00

                    yield Atom(
                        asym_unit=asym_unit,
                        type_symbol=element,
                        seq_id=residue_index,
                        atom_id=atom_name,
                        x=f"{pos_x:.5f}",
                        y=f"{pos_y:.5f}",
                        z=f"{pos_z:.5f}",
                        het=is_hetatm,
                        biso=biso,
                        occupancy=1.00,
                    )

    # --- 4. Write Output ---
    # Initialize System
    system = System()

    model = _KfoldModel(assembly=modeled_assembly, name="Model")
    model_group = ModelGroup([model], name="All models")

    system.model_groups.append(model_group)

    # Disable line wrapping for cleaner output
    ihm.dumper.set_line_wrap(False)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return fh.getvalue()
