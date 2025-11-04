import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from kfold.utils.boltz.process import parse_structure
from kfold.utils.boltz.structure import BoltzStructure
from kfold.utils.boltz.utils.featurizer import featurize
from kfold.utils.boltz.utils.tokenize import tokenize

BOLTZ_PATH = Path("/storage/share/Boltz1/rcsb_processed_targets/")
BOLTZ_STRUCTURE_DIR = BOLTZ_PATH / "structures"


def check_eq(a: torch.Tensor, b: torch.Tensor, key):
    assert a.shape == b.shape, f"Shape mismatch in {key}: {a.shape} vs {b.shape}"
    assert torch.eq(a, b).all(), f"Mismatch in {key}: {a} vs {b}"


def check_close(a: torch.Tensor, b: torch.Tensor, key):
    assert a.shape == b.shape, f"Shape mismatch in {key}: {a.shape} vs {b.shape}"
    assert torch.allclose(a, b.to(a.dtype), rtol=1e-3, atol=1e-4), f"Mismatch in {key}"


def check_structure(key: str, verbose: bool = False):
    print_ = lambda msg: print(msg) if verbose else None  # noqa: E731

    print_(f"Processed {key}")
    st = time.time()
    path = BOLTZ_STRUCTURE_DIR / f"{key}.npz"
    boltz_structure = BoltzStructure.load(path)
    print_(f"Loaded structure in {time.time() - st:.2f} seconds")

    if not 0 < len(boltz_structure.chains) < 100:
        print_(
            f"Skipping large structure {key} with {len(boltz_structure.chains)} chains"
        )
        return

    st = time.time()
    tokenized = tokenize(boltz_structure)
    featurized = featurize(tokenized)
    print_(f"Tokenized and featurized in {time.time() - st:.2f} seconds")

    st = time.time()
    chains = boltz_structure.chains[boltz_structure.mask]
    folding_input = parse_structure(chains, boltz_structure)
    print_(f"Parsed structure in {time.time() - st:.2f} seconds")

    # Check the input is the same
    atom_layout = folding_input.atom
    token_layout = folding_input.token
    bond_layout = folding_input.bond
    total_keys = set(featurized.keys())

    # === Test token features === #
    for key in ["cyclic_period"]:
        v1 = getattr(token_layout, key)
        v2 = featurized[key]
        check_eq(v1, v2, key)
        total_keys.remove(key)

    for key in [
        "token_index",
        "residue_index",
        "entity_id",
        "asym_id",
        "sym_id",
    ]:
        # NOTE: I follow AF3 indexing rule, starting from 1.
        v1 = getattr(token_layout, key)
        v2 = featurized[key] + 1
        check_eq(v1, v2, key)
        total_keys.remove(key)

    # check remaining token features
    check_eq(token_layout.frames_index, featurized["frames_idx"], "frames_idx")
    total_keys.remove("frames_idx")

    check_eq(
        token_layout.frames_mask,
        featurized["frame_resolved_mask"],
        "frame_resolved_mask",
    )
    total_keys.remove("frame_resolved_mask")

    check_eq(token_layout.pad_mask, featurized["token_pad_mask"], "token_pad_mask")
    total_keys.remove("token_pad_mask")

    check_eq(token_layout.chain_type, featurized["mol_type"], "mol_type")
    total_keys.remove("mol_type")

    check_eq(
        token_layout.resolved_mask,
        featurized["token_resolved_mask"],
        "token_resolved_mask",
    )
    total_keys.remove("token_resolved_mask")

    check_eq(
        token_layout.disto_index,
        featurized["token_to_rep_atom"].argmax(-1),
        "token_to_rep_atom",
    )
    total_keys.remove("token_to_rep_atom")

    check_eq(
        token_layout.center_index[~token_layout.is_ligand & token_layout.resolved_mask],
        featurized["r_set_to_rep_atom"].argmax(-1),
        "r_set_to_rep_atom",
    )
    total_keys.remove("r_set_to_rep_atom")

    check_eq(
        atom_layout.resolved_mask[token_layout.disto_index],
        featurized["token_disto_mask"],
        "token_disto_mask",
    )
    total_keys.remove("token_disto_mask")

    disto_center1 = atom_layout.label_coords[token_layout.disto_index].squeeze(1)
    disto_center2 = featurized["disto_center"]
    disto_center1 = disto_center1[featurized["token_disto_mask"].bool()]
    disto_center2 = disto_center2[featurized["token_disto_mask"].bool()]
    if len(disto_center1) > 0:
        shift = disto_center2[0] - disto_center1[0]
        disto_center1 = disto_center1 + shift  # align first point
        check_close(
            disto_center1,
            disto_center2,
            key="disto_center",
        )
    total_keys.remove("disto_center")

    # ========================================= #

    # === Test atom features === #
    for key in ["ref_charge", "ref_pos", "ref_space_uid"]:
        v1 = getattr(atom_layout, key)
        v2 = featurized[key]
        check_eq(v1, v2, key)
        total_keys.remove(key)

    for key in ["ref_atom_name_chars", "ref_element"]:
        v1 = getattr(atom_layout, key)
        v2 = featurized[key]
        check_eq(v1, v2.argmax(-1), key)
        total_keys.remove(key)

    # check remaining atom features
    check_eq(atom_layout.pad_mask, featurized["atom_pad_mask"], "atom_pad_mask")
    total_keys.remove("atom_pad_mask")

    check_eq(
        atom_layout.token_index,
        featurized["atom_to_token"].argmax(1),
        "atom_to_token",
    )
    total_keys.remove("atom_to_token")

    check_close(
        atom_layout.label_coords.squeeze(1)[atom_layout.resolved_mask],
        featurized["coords"].squeeze(0)[featurized["atom_resolved_mask"]],
        key="coords",
    )
    total_keys.remove("coords")

    check_eq(
        atom_layout.resolved_mask,
        featurized["atom_resolved_mask"],
        "atom_resolved_mask",
    )
    total_keys.remove("atom_resolved_mask")

    # ========================================= #

    # === Test bond features === #
    adj = torch.zeros(
        (token_layout.token_index.shape[0], token_layout.token_index.shape[0]),
        dtype=torch.bool,
    )
    for i, j in zip(
        bond_layout.token_index[:, 0], bond_layout.token_index[:, 1], strict=True
    ):
        adj[i, j] = True
        adj[j, i] = True  # Undirected
    check_eq(adj, featurized["token_bonds"].squeeze(-1), "token_bonds")
    total_keys.remove("token_bonds")

    print_(f"Remaining keys: {total_keys}")


if __name__ == "__main__":
    keys = sorted([p.stem for p in BOLTZ_STRUCTURE_DIR.glob("*.npz")])
    random.seed(42)
    random.shuffle(keys)

    for key in tqdm(keys[:10000]):
        try:
            check_structure(key)
        except Exception as e:
            print(f"Test failed for {key}: {e}")
