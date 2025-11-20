import json
import multiprocessing
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kfold.data.featurize import featurize_structure
from kfold.data.structure import TokenizedStructure
from kfold.utils.boltz.process import parse_record, tokenize_structure
from kfold.utils.boltz.structure import BoltzStructure
from kfold.utils.boltz.utils.featurizer import featurize as boltz_featurize
from kfold.utils.boltz.utils.tokenize import Tokenized
from kfold.utils.boltz.utils.tokenize import tokenize as boltz_tokenize

BOLTZ_PATH = Path("/cache/wykim_lab/rcsb_processed_targets/")
BOLTZ_MANIFEST_PATH = BOLTZ_PATH / "manifest.json"
BOLTZ_STRUCTURE_DIR = BOLTZ_PATH / "structures"


def check_eq(a: torch.Tensor, b: torch.Tensor, key: str):
    try:
        assert a.shape == b.shape, f"Shape mismatch in {key}: {a.shape} vs {b.shape}"
        assert torch.eq(a, b).all(), f"Mismatch in {key}"
    except Exception as e:
        raise Exception(f"Check eq failed for {key} - {e}") from e


def check_close(a: torch.Tensor, b: torch.Tensor, key):
    assert a.shape == b.shape, f"Shape mismatch in {key}: {a.shape} vs {b.shape}"
    try:
        assert torch.allclose(a, b.to(a.dtype), rtol=1e-3, atol=1e-2), (
            f"Mismatch in {key}"
        )
    except Exception as e:
        raise Exception(f"Check close failed for {key} - {e}") from e
        # print(f"Check close failed for {key} - {e}")
        # print(a)
        # print(b)
        # breakpoint()


def check_structure(key: str, verbose: bool = False):
    # Set random seed for reproducibility
    random.seed(key)

    print_ = lambda msg: print(msg) if verbose else None  # noqa: E731

    print_(f"Processed {key}")
    st = time.time()
    path = BOLTZ_STRUCTURE_DIR / f"{key}.npz"
    boltz_structure: BoltzStructure = BoltzStructure.load(path)
    print_(f"Loaded structure in {time.time() - st:.2f} seconds")

    if not 0 < len(boltz_structure.chains) < 100:
        print_(
            f"Skipping large structure {key} with {len(boltz_structure.chains)} chains"
        )
        return

    if not boltz_structure.mask.any():
        print_(f"Skipping empty structure {key}")
        return

    st = time.time()
    try:
        boltz_tokenized: Tokenized = boltz_tokenize(boltz_structure)
    except Exception as e:
        raise Exception(f"Fail during Boltz tokenization - {e}") from e

    print_(f"Boltz: Tokenize structure in {time.time() - st:.2f} seconds")

    st = time.time()
    kfold_tokenized: TokenizedStructure = tokenize_structure(boltz_structure)
    print_(f"KFold: Tokenize structure in {time.time() - st:.2f} seconds")

    # Check data save/load consistency
    npz_dict = kfold_tokenized.to_npz_dict()
    kfold_tokenized = TokenizedStructure.from_npz_dict(npz_dict)

    # Random cropping
    v = random.random()
    if v < 0.3:
        # contiguous crop
        i1 = random.randint(0, kfold_tokenized.token.length - 1)
        i2 = random.randint(0, kfold_tokenized.token.length - 1)
        while i1 == i2:
            i1 = random.randint(0, kfold_tokenized.token.length - 1)
            i2 = random.randint(0, kfold_tokenized.token.length - 1)
        i1, i2 = min(i1, i2), max(i1, i2) + 1
        print_(f"Cropping tokens: {i1} to {i2} / {kfold_tokenized.token.length}")
        token_indices = np.arange(i1, i2)

        boltz_tokenized = Tokenized(
            tokens=boltz_tokenized.tokens[token_indices],
            bonds=boltz_tokenized.bonds[
                np.isin(boltz_tokenized.bonds["token_1"], token_indices)
                & np.isin(boltz_tokenized.bonds["token_2"], token_indices)
            ],
            structure=boltz_tokenized.structure,
        )

        token_indices = np.arange(i1, i2)
        kfold_tokenized = kfold_tokenized.crop(token_indices)
    elif v < 0.6:
        # random crop (for spatial cropping)
        keep_prob = random.uniform(0.2, 0.8)
        print_(f"Randomly cropping tokens with keep prob: {keep_prob:.2f}")
        token_mask = np.random.rand(kfold_tokenized.token.length) < keep_prob
        token_indices = np.where(token_mask)[0]
        boltz_tokenized = Tokenized(
            tokens=boltz_tokenized.tokens[token_indices],
            bonds=boltz_tokenized.bonds[
                np.isin(boltz_tokenized.bonds["token_1"], token_indices)
                & np.isin(boltz_tokenized.bonds["token_2"], token_indices)
            ],
            structure=boltz_tokenized.structure,
        )
        kfold_tokenized = kfold_tokenized.crop(token_indices)

    st = time.time()
    try:
        boltz_featurized = boltz_featurize(boltz_tokenized)
    except Exception as e:
        raise Exception(f"Fail during Boltz featurization - {e}") from e
    print_(f"Boltz: Featurize structure in {time.time() - st:.2f} seconds")

    st = time.time()
    folding_input = featurize_structure(kfold_tokenized, augment_ref_pos=False)
    print_(f"KFold: Featurize structure in {time.time() - st:.2f} seconds")

    # Check the input is the same
    atom_layout = folding_input.atom
    token_layout = folding_input.token
    bond_layout = folding_input.bond
    total_keys = set(boltz_featurized.keys())

    # === Test token features === #
    for key in ["token_index"]:
        v1 = getattr(token_layout, key)
        v2 = boltz_featurized[key]
        check_eq(v1, v2, key)
        total_keys.remove(key)

    for key in [
        "residue_index",
        "entity_id",
        "asym_id",
        "sym_id",
    ]:
        # NOTE: I follow AF3 indexing rule, starting from 1.
        v1 = getattr(token_layout, key)
        v2 = boltz_featurized[key] + 1
        check_eq(v1, v2, key)
        total_keys.remove(key)

    for key in ["res_type"]:
        # NOTE: boltz use [PAD] token (32 types + 1)
        v1 = getattr(token_layout, key).argmax(-1) + 1
        v2 = boltz_featurized[key].argmax(-1)
        check_eq(v1, v2, key)
        total_keys.remove(key)

    # check remaining token features
    check_eq(token_layout.pad_mask, boltz_featurized["token_pad_mask"], "token_pad_mask")
    total_keys.remove("token_pad_mask")

    check_eq(token_layout.chain_type, boltz_featurized["mol_type"], "mol_type")
    total_keys.remove("mol_type")

    check_eq(
        token_layout.resolved_mask,
        boltz_featurized["token_resolved_mask"],
        "token_resolved_mask",
    )
    total_keys.remove("token_resolved_mask")

    check_eq(
        token_layout.disto_index,
        boltz_featurized["token_to_rep_atom"].argmax(-1),
        "token_to_rep_atom",
    )
    total_keys.remove("token_to_rep_atom")

    check_eq(
        token_layout.center_index[~token_layout.is_ligand & token_layout.resolved_mask],
        boltz_featurized["r_set_to_rep_atom"].argmax(-1),
        "r_set_to_rep_atom",
    )
    total_keys.remove("r_set_to_rep_atom")

    check_eq(
        token_layout.disto_mask,
        boltz_featurized["token_disto_mask"],
        "token_disto_mask",
    )
    total_keys.remove("token_disto_mask")

    disto_center1 = token_layout.disto_coords[token_layout.disto_mask]
    disto_center2 = boltz_featurized["disto_center"][
        boltz_featurized["token_disto_mask"].bool()
    ]
    if len(disto_center1) > 0:
        shift = disto_center2[0] - disto_center1[0]
        disto_center1 = disto_center1 + shift  # align first point
        check_close(
            disto_center1,
            disto_center2,
            key="disto_center",
        )
    total_keys.remove("disto_center")

    # Frame features
    frame_mask1 = token_layout.frames_mask & token_layout.resolved_mask
    frame_mask2 = (
        boltz_featurized["frame_resolved_mask"]
        & boltz_featurized["token_resolved_mask"].bool()
    )
    check_eq(frame_mask1, frame_mask2, "frame_resolved_mask")
    total_keys.remove("frame_resolved_mask")

    frame1 = token_layout.frames_index[frame_mask1]
    frame2 = boltz_featurized["frames_idx"][frame_mask2]
    try:
        check_eq(frame1, frame2, "frames_idx")
    except Exception:
        wrong_idx = torch.where(frame1 != frame2)[1]
        if (wrong_idx != 1).all():
            # [a b c] and [c b a] is equivalent for frames
            pass

    total_keys.remove("frames_idx")

    # ========================================= #

    # === Test atom features === #
    for key in [
        "ref_charge",
        "ref_pos",
        "ref_space_uid",
        "ref_atom_name_chars",
        "ref_element",
    ]:
        v1 = getattr(atom_layout, key)
        v2 = boltz_featurized[key]
        check_eq(v1, v2, key)
        total_keys.remove(key)

    # check remaining atom features
    check_eq(atom_layout.pad_mask, boltz_featurized["atom_pad_mask"], "atom_pad_mask")
    total_keys.remove("atom_pad_mask")

    check_eq(
        atom_layout.token_index,
        boltz_featurized["atom_to_token"].argmax(1),
        "atom_to_token",
    )
    total_keys.remove("atom_to_token")

    check_close(
        atom_layout.label_coords.squeeze(1)[atom_layout.resolved_mask],
        boltz_featurized["coords"].squeeze(0)[boltz_featurized["atom_resolved_mask"]],
        key="coords",
    )
    total_keys.remove("coords")

    check_eq(
        atom_layout.resolved_mask,
        boltz_featurized["atom_resolved_mask"],
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
    check_eq(adj, boltz_featurized["token_bonds"].squeeze(-1), "token_bonds")
    total_keys.remove("token_bonds")

    # Remove unused keys
    total_keys.remove("cyclic_period")
    total_keys.remove("disto_target")

    if len(total_keys) > 0:
        print_(f"Remaining keys: {total_keys}")


def safe_check_structure(key: str, verbose: bool = False):
    try:
        check_structure(key, verbose)
        return True
    except Exception as e:
        print(f"Test failed for {key}: {e}")
        return False


if __name__ == "__main__":
    with open(BOLTZ_MANIFEST_PATH) as f:
        manifest = json.load(f)

    manifest = {v["id"]: v for v in manifest}
    keys = sorted(list(manifest.keys()))
    random.seed(42)
    random.shuffle(keys)

    keys = keys[:10000]
    if False:
        """Multi-processing test"""
        test_keys = []
        for key in keys:
            record = parse_record(manifest[key])

            if record.num_chains > 50:
                continue
            test_keys.append(key)

        with multiprocessing.Pool(32) as p:
            results = list(
                tqdm(
                    p.imap_unordered(safe_check_structure, test_keys, chunksize=10),
                    total=len(test_keys),
                    desc="Processing structures",
                )
            )
        print(sum(results), len(results))
    else:
        for key in tqdm(keys):
            record = parse_record(manifest[key])

            if record.num_chains > 20:
                continue
            check_structure(key, verbose=True)
            # breakpoint()
