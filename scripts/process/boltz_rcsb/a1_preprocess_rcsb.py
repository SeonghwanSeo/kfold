"""
Script to save apo-holo pairs.

Logic:
1. Load boltz1 processed dataset.
2. Load protein apo structures.
3. Replace each TokenizedStructure's protein chains' apo structures with the
    loaded apo structures.
    - Protein chain:
        - If apo structure exists, save the apo-holo pair to the output file.
        - Otherwise, use holo structure as apo structure.
    - DNA/RNA chain:
        - Right now, use holo structure as apo structure.
    - Ligand:
        - Use `ref_pos`(ETKDG conformer) as apo structure.

TODO:
1. Add multiple ETKDG conformers for ligand.
2. Add mapping to non-standard residues:
    Currently, I simply masked them.
"""

import argparse
import functools
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

import kfold.constants as C
from kfold.constants.chain import ChainType
from kfold.data.structure import TokenizedStructure
from kfold.utils import errors
from kfold.utils.boltz.process import tokenize_structure
from kfold.utils.boltz.structure import BoltzStructure
from kfold.utils.files import load_apo_chain
from kfold.utils.geometry.random_augment import do_centering

# type alias
Point3D = tuple[float, float, float]

logger = logging.getLogger(__name__)


# FIXME: remove default path before publish
def parse_args():
    parser = argparse.ArgumentParser(
        description="Save apo-holo pairs from a dataset of protein structures."
    )
    parser.add_argument(
        "--boltz_structure_dir",
        type=Path,
        help="Path to the input file containing boltz1 processed dataset.",
        default="/cache/wykim_lab/rcsb_processed_targets/structures/",
    )
    parser.add_argument(
        "--apo_structure_dir",
        type=Path,
        help="Path to the input file containing apo protein structures.",
        default="/cache/wykim_lab/kfold_data/rcsb_apo_esmfold/",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the output directory to save Tokenized structure.",
        default="/cache/wykim_lab/kfold_data/structures/kfold_rcsb_processed_v251120_npz/",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of workers for parallel processing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to overwrite existing output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to enable verbose logging.",
    )
    parser.add_argument(
        "--allow_large_complex",
        action="store_true",
        help="Whether to include large structures (>300 chains).",
    )
    return parser.parse_args()


def extract_apo_coords(
    apo_chain: tuple[dict[int, str], dict[int, dict[str, Point3D]]],
    residue_indices: np.ndarray,
    num_atoms: np.ndarray,
    res_types: np.ndarray,
    atom_name_chars: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_tokens = len(residue_indices)

    chain_apo_coords = np.zeros((num_tokens, 24, 3), dtype=np.float32)
    chain_apo_mask = np.zeros((num_tokens, 24), dtype=bool)

    apo_sequences, apo_coordinates = apo_chain
    for token_idx in range(num_tokens):
        # === Get residue information === #
        res_idx = int(residue_indices[token_idx])  # 1-based index
        res_type = int(res_types[token_idx])
        res_name = C.residue.residue_id_to_name[res_type]

        if res_idx not in apo_chain:
            # Missing residue in apo structure
            logging.debug(f"Missing residue in apo structure: {res_idx} {res_name}")
            continue

        apo_res_name = C.residue.ResidueName(apo_sequences[res_idx])
        apo_atom_coords: dict[str, Point3D] = apo_coordinates[res_idx]
        assert res_name == apo_res_name, (
            f"Residue name mismatch! {res_name} vs {apo_res_name}"
            f" at residue index {res_idx}"
        )

        # === Map atom coordinates === #
        natoms = int(num_atoms[token_idx])
        for atom_idx in range(natoms):
            atom_name = "".join(
                [chr(c + 32) for c in atom_name_chars[token_idx, atom_idx] if c != 0]
            )
            if atom_name not in apo_atom_coords:
                # Missing atom in apo structure
                logging.debug(
                    "Missing atom in apo structure:"
                    + f" {res_idx} {res_name} {atom_name}",
                )
                continue
            chain_apo_coords[token_idx, atom_idx] = apo_atom_coords[atom_name]
            chain_apo_mask[token_idx, atom_idx] = True

    return chain_apo_coords, chain_apo_mask


def parse_structure(
    boltz_npz_file: Path,
    apo_structure_path: Path,
    output_path: Path,
    allow_large_complex: bool = False,
    force: bool = False,
) -> bool:
    """Parse a boltz structure file and save the tokenized structure with apo coords.

    Parameters
    ----------
    boltz_npz_file : Path
        Path to the boltz npz file.
    apo_structure_path : Path
        Path to the directory containing apo structures.
    output_path : Path
        Path to save the output tokenized structure npz file.
    force : bool, optional
        Whether to overwrite existing output files, by default False.

    Returns
    -------
    bool
        True if the structure was processed and saved successfully, False otherwise.
    """

    if output_path.exists():
        if force:
            logger.debug(f"Output file already exists, overwriting: {output_path}")
        else:
            logger.debug(f"Output file already exists, skipping: {output_path}")
            return True

    pdb_id = boltz_npz_file.stem

    boltz_structure: BoltzStructure = BoltzStructure.load(boltz_npz_file)
    try:
        tokenized_structure: TokenizedStructure = tokenize_structure(boltz_structure)
    except errors.BoltzDataProcessingError as e:
        logging.warning(
            f"Failed to tokenize structure: {boltz_npz_file.stem}, error: {e}"
        )
        return False

    chain_data = tokenized_structure.chain
    token_data = tokenized_structure.token
    atom_data = tokenized_structure.atom

    # Skip large complexes
    if not allow_large_complex and tokenized_structure.num_chains > 300:
        logger.info(
            f"Skipping large complex (>300 chains): {boltz_npz_file.stem} "
            f"with {tokenized_structure.num_chains} chains."
        )
        return False

    # Update `apo_coords` (num_tokens, atoms_per_token, num_apos, 3)
    # TODO: handle multiple apo conformers
    num_all_tokens = tokenized_structure.num_tokens
    all_apo_coords = np.zeros((num_all_tokens, 24, 1, 3), dtype=np.float32)
    all_apo_mask = np.zeros((num_all_tokens, 24, 1), dtype=bool)

    # Cache loaded apo chains
    entity_apo_chains: dict[int, Any] = {}

    token_start = 0
    for i in range(tokenized_structure.num_chains):
        chain_type = ChainType(chain_data.chain_type[i])
        entity_id = int(chain_data.entity_id[i])
        num_tokens = int(chain_data.num_tokens[i])
        token_end = token_start + num_tokens

        match chain_type:
            case ChainType.PROTEIN | ChainType.DNA | ChainType.RNA:
                # === Load apo structure for protein/DNA/RNA chain === #
                # entity_id starts from 1
                filename = f"{pdb_id}_{entity_id}_{chain_type.name.lower()}.pdb"
                apo_chain_pdb_path = apo_structure_path / filename

                if entity_id in entity_apo_chains:
                    # Apo structure already loaded
                    apo_chain = entity_apo_chains[entity_id]
                elif apo_chain_pdb_path.exists():
                    # Load apo structure
                    # Cache the loaded apo structure
                    entity_apo_chains[entity_id] = load_apo_chain(apo_chain_pdb_path)
                else:
                    # Apo structure not found
                    apo_chain = None
                    logging.debug(
                        "Apo structure not found for"
                        f"entity_id {entity_id} at {apo_chain_pdb_path}"
                    )

                # === Extract apo coordinates === #
                if apo_chain is None:
                    # Use centered holo structure as apo structure
                    holo_coords = atom_data.coords[token_start:token_end, :, 0]
                    mask = atom_data.resolved_mask[token_start:token_end]
                    chain_apo_coords = do_centering(holo_coords, mask)
                    chain_apo_mask = mask
                else:
                    chain_apo_coords, chain_apo_mask = extract_apo_coords(
                        apo_chain,
                        token_data.residue_index[token_start:token_end],
                        token_data.num_atoms[token_start:token_end],
                        token_data.res_type[token_start:token_end],
                        atom_data.ref_atom_name_chars[token_start:token_end],
                    )
            case C.chain.ChainType.LIGAND:
                # Currently, use `ref_pos` as apo structure. (single ETKDG conformer)
                # TODO: create multiple ETKDG conformers for ligand apo structure
                ref_pos = atom_data.ref_pos[token_start:token_end]  # From ETKDG
                ref_mask = atom_data.resolved_mask[token_start:token_end]
                chain_apo_coords = ref_pos
                chain_apo_mask = ref_mask

        # Save apo coordinates
        all_apo_coords[token_start:token_end, :, 0] = chain_apo_coords
        all_apo_mask[token_start:token_end, :, 0] = chain_apo_mask

        # Update indices
        token_start = token_end

    atom_data = atom_data.copy_with(apo_coords=all_apo_coords, apo_mask=all_apo_mask)
    tokenized_structure = tokenized_structure.copy_with(atom=atom_data)

    # Save the updated tokenized structure
    tokenized_structure.dump_npz(output_path)
    return True


def parse_structure_safe(
    boltz_npz_file: Path,
    apo_structure_dir: Path,
    output_dir: Path,
    allow_large_complex: bool = False,
    force: bool = False,
) -> bool:
    """Wrapper with better error handling."""
    pdb_id = boltz_npz_file.stem
    output_path = output_dir / f"{pdb_id}.npz"
    apo_path = apo_structure_dir / pdb_id[:2] / pdb_id
    try:
        success = parse_structure(
            boltz_npz_file, apo_path, output_path, allow_large_complex, force
        )
        return success
    except Exception as e:
        logger.error(f"Failed to process {pdb_id}: {e}")
        return False


def main(args):
    boltz_structure_dir = Path(args.boltz_structure_dir)
    apo_structure_dir = Path(args.apo_structure_dir)
    output_dir = Path(args.output_dir)
    allow_large_complex = args.allow_large_complex

    num_cpus = args.num_cpus
    force = args.force

    assert boltz_structure_dir.exists(), (
        f"Boltz structure dir not found: {boltz_structure_dir}"
    )
    assert apo_structure_dir.exists(), f"Apo structure dir not found: {apo_structure_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)

    all_boltz_npz_files = sorted(list(boltz_structure_dir.glob("*.npz")))

    with multiprocessing.Pool(num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    functools.partial(
                        parse_structure_safe,
                        apo_structure_dir=apo_structure_dir,
                        output_dir=output_dir,
                        allow_large_complex=allow_large_complex,
                        force=force,
                    ),
                    all_boltz_npz_files,
                ),
                total=len(all_boltz_npz_files),
            )
        )
    num_success = sum(results)
    num_failed = len(results) - num_success
    logger.info(
        f"Processing completed: {num_success} succeeded, "
        f"{num_failed} failed or skipped(>300chain)."
    )


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    main(args)
