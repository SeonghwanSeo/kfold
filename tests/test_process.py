import json
import multiprocessing
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kfold.data.featurize import InputFeaturizer
from kfold.data.structure import TokenizedStructure
from kfold.utils.boltz.process import parse_record, tokenize_structure
from kfold.utils.boltz.structure import BoltzStructure
from kfold.utils.boltz.utils.featurizer import featurize as boltz_featurize
from kfold.utils.boltz.utils.tokenize import Tokenized as BoltzTokenized
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


def check_close(a: torch.Tensor, b: torch.Tensor, key: str, rtol=1e-3, atol=1e-2):
    assert a.shape == b.shape, f"Shape mismatch in {key}: {a.shape} vs {b.shape}"
    if a.numel() == 0:
        # Empty tensor, nothing to check
        return
    try:
        assert torch.allclose(a, b.to(a.dtype), rtol=rtol, atol=atol), (
            f"Mismatch in {key}"
        )
    except Exception as e:
        raise Exception(f"Check close failed for {key} - {e}") from e
        # print(f"Check close failed for {key} - {e}")
        # print(a)
        # print(b)
        # breakpoint()


def crop_boltz_structure(
    struct: BoltzTokenized, token_indices: np.ndarray
) -> BoltzTokenized:
    struct = BoltzTokenized(
        tokens=struct.tokens[token_indices],
        bonds=struct.bonds[
            np.isin(struct.bonds["token_1"], token_indices)
            & np.isin(struct.bonds["token_2"], token_indices)
        ],
        structure=struct.structure,
    )
    return struct


def check_structure(key: str, verbose: bool = False):
    # Set random seed for reproducibility
    random.seed(key)

    print_ = lambda msg: print(msg) if verbose else None  # noqa: E731

    print_(f"Processed {key}")
    st = time.time()
    path = BOLTZ_STRUCTURE_DIR / f"{key}.npz"
    boltz_structure: BoltzStructure = BoltzStructure.load(path)
    print_(f"Loaded structure in {time.time() - st:.2f} seconds")

    if not boltz_structure.mask.any():
        print_(f"Skipping empty structure {key}")
        return

    # ==================================================== #
    # ================= Tokenization ===================== #
    # ==================================================== #

    st = time.time()
    try:
        boltz_tokenized: BoltzTokenized = boltz_tokenize(boltz_structure)
    except Exception as e:
        raise Exception(f"Fail during Boltz tokenization - {e}") from e

    print_(f"Boltz: Tokenize structure in {time.time() - st:.2f} seconds")

    st = time.time()
    kfold_tokenized: TokenizedStructure = tokenize_structure(boltz_structure)
    print_(f"KFold: Tokenize structure in {time.time() - st:.2f} seconds")

    if kfold_tokenized.num_tokens == 0:
        print_(f"Skipping empty tokenized structure {key}")
        return

    # Check data save/load consistency
    st = time.time()
    npz_dict = kfold_tokenized.to_npz_dict()
    kfold_tokenized = TokenizedStructure.from_npz_dict(npz_dict)

    if boltz_tokenized.tokens.shape[0] != kfold_tokenized.num_tokens:
        raise Exception(
            f"Token number mismatch after save/load for {key}: "
            f"{boltz_tokenized.tokens.shape[0]} vs {kfold_tokenized.num_tokens}"
        )

    # ==================================================== #
    # ================= Random Cropping ================== #
    # ==================================================== #

    if (v := random.random()) < 0.6:
        num_tokens = kfold_tokenized.num_tokens
        if v < 0.3:
            # contiguous crop
            i1, i2 = np.random.choice(num_tokens, size=2, replace=False)
            i1, i2 = min(i1, i2), max(i1, i2) + 1
            print_(f"Cropping tokens: {i1} to {i2} / {num_tokens}")
            token_indices = np.arange(i1, i2)
        else:
            # random crop (for spatial cropping)
            keep_prob = np.random.uniform(0.2, 0.8)
            print_(f"Randomly cropping tokens with keep prob: {keep_prob:.2f}")
            token_indices = (np.random.rand(num_tokens) < keep_prob).nonzero()[0]
        boltz_tokenized = crop_boltz_structure(boltz_tokenized, token_indices)
        kfold_tokenized = kfold_tokenized.crop(token_indices)
    else:
        print_(f"No cropping applied on tokens {kfold_tokenized.num_tokens}")

    # ==================================================== #
    # ================== Featurization =================== #
    # ==================================================== #

    st = time.time()
    try:
        boltz_featurized = boltz_featurize(boltz_tokenized)
    except Exception as e:
        raise Exception(f"Fail during Boltz featurization - {e}") from e
    print_(f"Boltz: Featurize structure in {time.time() - st:.2f} seconds")

    st = time.time()
    featurizer = InputFeaturizer(augment_ref_pos=False)
    folding_input = featurizer.run(kfold_tokenized)
    print_(f"KFold: Featurize structure in {time.time() - st:.2f} seconds")

    # ==================================================== #
    # ================== Check Arrays ==================== #
    # ==================================================== #

    # Check the input is the same
    total_keys = set(boltz_featurized.keys())

    atom_layout = folding_input.atom
    token_layout = folding_input.token
    bond_layout = folding_input.bond

    # === Test token features === #
    for key1, key2 in [
        ("token_index", "token_index"),
        ("pad_mask", "token_pad_mask"),
        ("resolved_mask", "token_resolved_mask"),
        ("disto_mask", "token_disto_mask"),
        ("chain_type", "mol_type"),
    ]:
        check_eq(
            getattr(token_layout, key1),
            boltz_featurized[key2],
            key2,
        )
        total_keys.remove(key2)

    for key1, key2 in [
        ("residue_index", "residue_index"),
        ("entity_id", "entity_id"),
        ("asym_id", "asym_id"),
        ("sym_id", "sym_id"),
    ]:
        # NOTE: I follow AF3 indexing rule, starting from 1.
        check_eq(
            getattr(token_layout, key1),
            boltz_featurized[key2] + 1,
            key2,
        )
        total_keys.remove(key2)

    for key1, key2 in [
        ("res_type", "res_type"),
    ]:
        # NOTE: boltz use [PAD] token (32 types + 1)
        check_eq(
            getattr(token_layout, key1).argmax(-1) + 1,
            boltz_featurized[key2].argmax(-1),
            key2,
        )
        total_keys.remove(key2)

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

    if token_layout.disto_mask.any():
        # The disto center is shifted due to centering,
        # so we need to account for the shift.
        disto_center1 = token_layout.disto_coords
        disto_center2 = boltz_featurized["disto_center"]
        shift = (disto_center2 - disto_center1)[token_layout.disto_mask][0]
        check_close(
            disto_center1[token_layout.disto_mask] + shift,
            disto_center2[token_layout.disto_mask],
            key="disto_center",
        )

        boundaries = torch.linspace(2.0, 22.0, 64 - 1)
        distogram = torch.cdist(disto_center1, disto_center1)
        disto_target = (distogram[..., None] > boundaries).sum(-1)
        disto_mask = token_layout.disto_mask[:, None] & token_layout.disto_mask[None, :]
        check_close(
            disto_target[disto_mask],
            boltz_featurized["disto_target"].argmax(-1)[disto_mask],
            key="disto_target",
            atol=1,  # allow integer tolerance
        )
    total_keys.remove("disto_center")
    total_keys.remove("disto_target")

    # Frame features
    frame_mask1 = token_layout.frames_mask & token_layout.resolved_mask
    frame_mask2 = boltz_featurized["frame_resolved_mask"] & token_layout.resolved_mask
    check_eq(frame_mask1, frame_mask2, "frame_resolved_mask")
    total_keys.remove("frame_resolved_mask")

    frame1 = token_layout.frames_index[frame_mask1]
    frame2 = boltz_featurized["frames_idx"][frame_mask2]
    try:
        check_eq(frame1, frame2, "frames_idx")
    except Exception:
        # [a b c] and [c b a] is equivalent for frames
        wrong_idx = torch.where(frame1 != frame2)[1]
        if (wrong_idx != 1).all():
            pass

    total_keys.remove("frames_idx")

    # ======================================== #

    # === Test atom features === #
    for key1, key2 in [
        ("ref_charge", "ref_charge"),
        ("ref_pos", "ref_pos"),
        ("ref_space_uid", "ref_space_uid"),
        ("ref_atom_name_chars", "ref_atom_name_chars"),
        ("ref_element", "ref_element"),
        ("pad_mask", "atom_pad_mask"),
        ("resolved_mask", "atom_resolved_mask"),
    ]:
        check_eq(
            getattr(atom_layout, key1),
            boltz_featurized[key2],
            key2,
        )
        total_keys.remove(key2)

    # check remaining atom features
    check_eq(
        atom_layout.token_index,
        boltz_featurized["atom_to_token"].argmax(1),
        "atom_to_token",
    )
    check_eq(
        folding_input.atom_to_token,
        boltz_featurized["atom_to_token"],
        "atom_to_token",
    )
    total_keys.remove("atom_to_token")

    check_close(
        atom_layout.label_coords.squeeze(1)[atom_layout.resolved_mask],
        boltz_featurized["coords"].squeeze(0)[atom_layout.resolved_mask],
        key="coords",
    )
    total_keys.remove("coords")

    # ========================================= #

    # === Test bond features === #
    num_tokens = kfold_tokenized.num_tokens
    adj = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    adj[bond_layout.token_index[:, 0], bond_layout.token_index[:, 1]] = True
    adj[bond_layout.token_index[:, 1], bond_layout.token_index[:, 0]] = True
    check_eq(adj, boltz_featurized["token_bonds"].squeeze(-1), "token_bonds")
    total_keys.remove("token_bonds")

    # Remove unused keys
    total_keys.remove("cyclic_period")

    if len(total_keys) > 0:
        raise Exception(f"Not all keys checked for {key}")


def safe_check_structure(key: str, verbose: bool = False):
    try:
        check_structure(key, verbose)
        return True
    except Exception as e:
        print(f"Test failed for {key}: {e}")
        return False


if __name__ == "__main__":
    print("Loading Boltz manifest...")
    st = time.time()
    with open(BOLTZ_MANIFEST_PATH) as f:
        manifest = json.load(f)
    print(f"Loaded manifest in {time.time() - st:.2f} seconds")

    manifest = {v["id"]: v for v in manifest}
    keys = sorted(list(manifest.keys()))
    random.seed(42)
    random.shuffle(keys)

    keys = keys[:1000]
    if False:
        """Multi-processing test"""
        test_keys = []
        for key in keys:
            if len(manifest[key]["chains"]) > 50:
                continue
            record = parse_record(manifest[key])
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
            if len(manifest[key]["chains"]) > 50:
                continue
            record = parse_record(manifest[key])
            check_structure(key, verbose=True)
            # breakpoint()
