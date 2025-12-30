#!/usr/bin/env python3
"""
Standalone sanity check for ApoPerturbation Langevin dynamics on tokenized LMDB.

NOTE:
- This file is intentionally named with `test_` prefix to live under `tests/`,
  but it is NOT a pytest unit test module. It has no `test_*` functions and is
  meant to be executed as a CLI script.

It:
1) loads a TokenizedStructure from an LMDB (values are NPZ bytes)
2) runs ApoPerturbation.run(struct) with Langevin enabled
3) prints before/after diagnostics, NA RMSD-to-holo comparisons, and
   optional bond distance raw samples. Batch/grid evaluation is supported.

Typical usage (on gpu05 where the LMDB is accessible):
  python tests/test_debug_langevin_lmdb.py \
    --lmdb /cache/wykim_lab/kfold_data/kfold_rcsb_processed_v251120.lmdb \
    --find-na 3000 --eval-na-batch 20 --na-rmsd \
    --steps-grid 64 128 --dt-grid 0.25 --bond-coef-grid 1.0 2.0 --seeds 0 1
"""

from __future__ import annotations

import argparse
import io
import itertools
from pathlib import Path

import lmdb
import numpy as np

import kfold.constants as C
from kfold.data.apo_perturbation import ApoPerturbation
from kfold.data.tokenized import TokenizedStructure
from kfold.utils.geometry.rigid_align import compute_rmsd


def _open_lmdb(path: str) -> lmdb.Environment:
    p = Path(path)
    # Allow both directory (".../xxx.lmdb/") and direct data.mdb path.
    if p.is_file() and p.name == "data.mdb":
        p = p.parent
    return lmdb.open(
        str(p),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def _load_struct(env: lmdb.Environment, key: str) -> TokenizedStructure:
    with env.begin(write=False) as txn:
        value_bytes = txn.get(key.encode("utf-8"))
        if value_bytes is None:
            raise KeyError(f"Key not found in LMDB: {key}")
    with io.BytesIO(value_bytes) as bio:
        return TokenizedStructure.load_npz(bio)


def _first_keys(env: lmdb.Environment, limit: int) -> list[str]:
    keys: list[str] = []
    with env.begin(write=False) as txn:
        with txn.cursor() as cur:
            for k, _v in cur:
                keys.append(k.decode("utf-8"))
                if len(keys) >= limit:
                    break
    return keys


def _compute_basic_stats(struct: TokenizedStructure) -> dict[str, float]:
    # Work on dense [Ntoken,24,3] apo coords (single apo assumed) and apo mask.
    apo = struct.atom.apo_coords[..., 0, :]  # [Ntoken,24,3]
    mask = struct.atom.apo_mask[..., 0].astype(bool)  # [Ntoken,24]
    if apo.size == 0:
        return {"num_apo_atoms": 0.0, "rms_radius": 0.0, "zero_frac": 1.0}

    flat = apo.reshape(-1, 3)
    flat_mask = mask.reshape(-1)
    num = int(flat_mask.sum())
    if num == 0:
        return {"num_apo_atoms": 0.0, "rms_radius": 0.0, "zero_frac": 1.0}

    coords = flat[flat_mask]
    rms_radius = float(np.sqrt(np.mean(np.sum(coords**2, axis=-1))))
    zero_frac = float(np.mean(np.all(flat == 0.0, axis=-1)))
    return {"num_apo_atoms": float(num), "rms_radius": rms_radius, "zero_frac": zero_frac}


def _bond_distance_stats(struct: TokenizedStructure) -> dict[str, float]:
    # Compute bond distances in apo coords for bonds whose endpoints are apo-masked.
    if struct.num_bonds == 0:
        return {"num_bonds_used": 0.0, "bond_dist_mean": 0.0, "bond_dist_p95": 0.0}

    apo = struct.atom.apo_coords[..., 0, :]  # [Ntoken,24,3]
    mask = struct.atom.apo_mask[..., 0].astype(bool)  # [Ntoken,24]

    t = struct.bond.token_index.astype(np.int64)
    a = struct.bond.atom_index.astype(np.int64)
    t1, t2 = t[:, 0], t[:, 1]
    a1, a2 = a[:, 0], a[:, 1]

    valid = (0 <= a1) & (a1 < 24) & (0 <= a2) & (a2 < 24) & mask[t1, a1] & mask[t2, a2]
    idx = np.where(valid)[0]
    if idx.size == 0:
        return {"num_bonds_used": 0.0, "bond_dist_mean": 0.0, "bond_dist_p95": 0.0}

    p1 = apo[t1[idx], a1[idx], :]
    p2 = apo[t2[idx], a2[idx], :]
    d = np.linalg.norm(p1 - p2, axis=-1)
    return {
        "num_bonds_used": float(d.size),
        "bond_dist_mean": float(np.mean(d)),
        "bond_dist_p95": float(np.percentile(d, 95.0)),
    }


def _chain_slices(struct: TokenizedStructure) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    for chain_i in range(struct.num_chains):
        st = int(struct.chain.token_start[chain_i])
        L = int(struct.chain.num_tokens[chain_i])
        slices.append((st, st + L))
    return slices


def _is_na_chain(struct: TokenizedStructure, chain_i: int) -> bool:
    ct = int(struct.chain.chain_type[chain_i])
    return ct in (C.chain.ChainType.DNA.value, C.chain.ChainType.RNA.value)


def _na_bond_stats_from_coords(
    struct: TokenizedStructure,
    coords_by_chain: dict[int, np.ndarray],
    mask_by_chain: dict[int, np.ndarray],
) -> dict[str, float]:
    """Bond distance stats for NA-only bonds using provided coords/masks per NA chain."""
    if struct.num_bonds == 0:
        return {"na_bonds_used": 0.0, "na_bond_mean": 0.0, "na_bond_p95": 0.0}

    all_d: list[np.ndarray] = []
    bond_token = struct.bond.token_index.astype(np.int64, copy=False)
    bond_atom = struct.bond.atom_index.astype(np.int64, copy=False)

    for chain_i, coords in coords_by_chain.items():
        st = int(struct.chain.token_start[chain_i])
        L = int(struct.chain.num_tokens[chain_i])
        end = st + L
        mask = mask_by_chain[chain_i]

        in_slice = (bond_token[:, 0] >= st) & (bond_token[:, 0] < end)
        in_slice &= (bond_token[:, 1] >= st) & (bond_token[:, 1] < end)
        idxs = np.where(in_slice)[0]
        if idxs.size == 0:
            continue

        t1 = (bond_token[idxs, 0] - st).astype(np.int64)
        t2 = (bond_token[idxs, 1] - st).astype(np.int64)
        a1 = bond_atom[idxs, 0].astype(np.int64)
        a2 = bond_atom[idxs, 1].astype(np.int64)
        valid = (0 <= a1) & (a1 < 24) & (0 <= a2) & (a2 < 24)
        if not np.any(valid):
            continue
        t1, t2, a1, a2 = t1[valid], t2[valid], a1[valid], a2[valid]
        valid2 = mask[t1, a1] & mask[t2, a2]
        if not np.any(valid2):
            continue
        t1, t2, a1, a2 = t1[valid2], t2[valid2], a1[valid2], a2[valid2]

        p1 = coords[t1, a1, :]
        p2 = coords[t2, a2, :]
        d = np.linalg.norm(p1 - p2, axis=-1)
        all_d.append(d)

    if not all_d:
        return {"na_bonds_used": 0.0, "na_bond_mean": 0.0, "na_bond_p95": 0.0}

    d_all = np.concatenate(all_d, axis=0)
    return {
        "na_bonds_used": float(d_all.size),
        "na_bond_mean": float(np.mean(d_all)),
        "na_bond_p95": float(np.percentile(d_all, 95.0)),
    }


def _na_rms_radius_from_coords(
    coords_by_chain: dict[int, np.ndarray],
    mask_by_chain: dict[int, np.ndarray],
) -> dict[str, float]:
    vals: list[float] = []
    for chain_i, coords in coords_by_chain.items():
        mask = mask_by_chain[chain_i]
        flat = coords.reshape(-1, 3)
        flat_mask = mask.reshape(-1)
        if not np.any(flat_mask):
            continue
        c = flat[flat_mask]
        vals.append(float(np.sqrt(np.mean(np.sum(c**2, axis=-1)))))
    if not vals:
        return {"na_rms_radius_mean": 0.0}
    return {
        "na_rms_radius_mean": float(np.mean(vals)),
        "na_rms_radius_min": float(np.min(vals)),
        "na_rms_radius_max": float(np.max(vals)),
    }


def _na_eval_once(
    struct: TokenizedStructure,
    steps: int,
    dt: float,
    res_r: float,
    ent_r: float,
    sphere_r: float,
    bond_coef: float,
    seed: int,
) -> dict[str, float]:
    """Evaluate NA chains for one (steps,dt,bond_coef,seed)."""
    apo_pert = ApoPerturbation(
        use_perturbation=True,
        prob_perturbation=1.0,
        metric_lmdb_path=None,
        metric_comp=None,
        random_walk=None,
        langevin={
            "num_steps": int(steps),
            "dt": float(dt),
            "res_r": float(res_r),
            "ent_r": float(ent_r),
            "sphere_r": float(sphere_r),
            "bond_coef": float(bond_coef),
        },
        seed=int(seed),
    )

    chain_slices = _chain_slices(struct)
    rmsd_init: list[float] = []
    rmsd_after: list[float] = []
    init_coords_by_chain: dict[int, np.ndarray] = {}
    after_coords_by_chain: dict[int, np.ndarray] = {}
    mask_by_chain: dict[int, np.ndarray] = {}

    for chain_i, (st, end) in enumerate(chain_slices):
        if not _is_na_chain(struct, chain_i):
            continue

        holo = struct.atom.coords[st:end, :, 0, :].astype(np.float32, copy=False)
        mask = struct.atom.resolved_mask[st:end].astype(bool, copy=False)
        if not np.any(mask):
            continue

        rng0 = np.random.default_rng(seed)
        x0 = rng0.normal(loc=0.0, scale=sphere_r, size=holo.shape).astype(np.float32)
        x0[~mask] = 0.0

        rng_ld = np.random.default_rng(seed)
        x_after = apo_pert.langevin_dynamics_perturbation(
            apo_coords=None,
            mask=mask,
            rng=rng_ld,
            struct=struct,
            chain_i=chain_i,
            record_id="<na_batch_eval>",
            entity_id=int(struct.chain.entity_id[chain_i]),
        ).astype(np.float32, copy=False)

        flat_mask = mask.reshape(-1).astype(np.float32)
        r0 = compute_rmsd(x0.reshape(-1, 3), holo.reshape(-1, 3), flat_mask, align=True)
        r1 = compute_rmsd(
            x_after.reshape(-1, 3), holo.reshape(-1, 3), flat_mask, align=True
        )
        rmsd_init.append(float(r0))
        rmsd_after.append(float(r1))

        init_coords_by_chain[chain_i] = x0
        after_coords_by_chain[chain_i] = x_after
        mask_by_chain[chain_i] = mask

    if not rmsd_init:
        return {"na_chains": 0.0}

    init_b = _na_bond_stats_from_coords(struct, init_coords_by_chain, mask_by_chain)
    after_b = _na_bond_stats_from_coords(struct, after_coords_by_chain, mask_by_chain)
    after_r = _na_rms_radius_from_coords(after_coords_by_chain, mask_by_chain)

    out: dict[str, float] = {
        "na_chains": float(len(rmsd_init)),
        "rmsd_init_mean": float(np.mean(rmsd_init)),
        "rmsd_after_mean": float(np.mean(rmsd_after)),
        "rmsd_delta_mean": float(np.mean(np.asarray(rmsd_after) - np.asarray(rmsd_init))),
    }
    out |= {f"init_{k}": v for k, v in init_b.items()}
    out |= {f"after_{k}": v for k, v in after_b.items()}
    out |= after_r
    return out


def _debug_print_na_bond_samples(
    struct: TokenizedStructure,
    steps: int,
    dt: float,
    res_r: float,
    ent_r: float,
    sphere_r: float,
    bond_coef: float,
    seed: int,
    n_samples: int,
) -> None:
    """
    Print raw NA bond samples (token/atom indices + distances) for one record/setting.
    """
    chain_slices = _chain_slices(struct)
    pick = None
    for chain_i, (st, end) in enumerate(chain_slices):
        if _is_na_chain(struct, chain_i):
            pick = (chain_i, st, end)
            break
    if pick is None:
        print("[debug_na_bonds] no NA chain found.")
        return

    chain_i, st, end = pick
    holo = struct.atom.coords[st:end, :, 0, :].astype(np.float32, copy=False)
    mask = struct.atom.resolved_mask[st:end].astype(bool, copy=False)
    if not np.any(mask):
        print("[debug_na_bonds] NA chain has no resolved atoms.")
        return

    rng0 = np.random.default_rng(seed)
    x0 = rng0.normal(loc=0.0, scale=sphere_r, size=holo.shape).astype(np.float32)
    x0[~mask] = 0.0

    apo_pert = ApoPerturbation(
        use_perturbation=True,
        prob_perturbation=1.0,
        metric_lmdb_path=None,
        metric_comp=None,
        random_walk=None,
        langevin={
            "num_steps": int(steps),
            "dt": float(dt),
            "res_r": float(res_r),
            "ent_r": float(ent_r),
            "sphere_r": float(sphere_r),
            "bond_coef": float(bond_coef),
        },
        seed=int(seed),
    )
    rng_ld = np.random.default_rng(seed)
    x_after = apo_pert.langevin_dynamics_perturbation(
        apo_coords=None,
        mask=mask,
        rng=rng_ld,
        struct=struct,
        chain_i=chain_i,
        record_id="<debug_na_bonds>",
        entity_id=int(struct.chain.entity_id[chain_i]),
    ).astype(np.float32, copy=False)

    bond_token = struct.bond.token_index.astype(np.int64, copy=False)
    bond_atom = struct.bond.atom_index.astype(np.int64, copy=False)
    in_slice = (bond_token[:, 0] >= st) & (bond_token[:, 0] < end)
    in_slice &= (bond_token[:, 1] >= st) & (bond_token[:, 1] < end)
    idxs = np.where(in_slice)[0]
    if idxs.size == 0:
        print("[debug_na_bonds] no bonds in NA chain slice.")
        return

    t1 = (bond_token[idxs, 0] - st).astype(np.int64)
    t2 = (bond_token[idxs, 1] - st).astype(np.int64)
    a1 = bond_atom[idxs, 0].astype(np.int64)
    a2 = bond_atom[idxs, 1].astype(np.int64)

    valid = (0 <= a1) & (a1 < 24) & (0 <= a2) & (a2 < 24)
    t1, t2, a1, a2 = t1[valid], t2[valid], a1[valid], a2[valid]
    if t1.size == 0:
        print("[debug_na_bonds] bonds exist but atom_index out of [0,24).")
        return

    valid2 = mask[t1, a1] & mask[t2, a2]
    t1, t2, a1, a2 = t1[valid2], t2[valid2], a1[valid2], a2[valid2]
    if t1.size == 0:
        print("[debug_na_bonds] no bonds with both endpoints resolved_mask==True.")
        return

    n = min(int(n_samples), int(t1.size))
    rng_s = np.random.default_rng(seed + 12345)
    pick_idx = rng_s.choice(t1.size, size=n, replace=False)

    d0 = np.linalg.norm(
        x0[t1[pick_idx], a1[pick_idx]] - x0[t2[pick_idx], a2[pick_idx]], axis=-1
    )
    d1 = np.linalg.norm(
        x_after[t1[pick_idx], a1[pick_idx]] - x_after[t2[pick_idx], a2[pick_idx]],
        axis=-1,
    )
    dh = np.linalg.norm(
        holo[t1[pick_idx], a1[pick_idx]] - holo[t2[pick_idx], a2[pick_idx]], axis=-1
    )

    print(
        f"[debug_na_bonds] chain_i={chain_i} (tokens {st}:{end}), "
        f"bonds_total_in_slice={int(idxs.size)}, bonds_used={int(t1.size)}"
    )
    print(
        f"[debug_na_bonds] params: steps={steps} dt={dt} bond_coef={bond_coef} "
        f"sphere_r={sphere_r}"
    )
    print("[debug_na_bonds] sample distances: init / after / holo")
    for k in range(n):
        print(
            f"  (t{int(t1[pick_idx[k]])},a{int(a1[pick_idx[k]])})-"
            f"(t{int(t2[pick_idx[k]])},a{int(a2[pick_idx[k]])}): "
            f"{d0[k]:.3f} / {d1[k]:.3f} / {dh[k]:.3f}"
        )
    print(
        f"[debug_na_bonds] mean(init/after/holo) = "
        f"{float(d0.mean()):.3f} / {float(d1.mean()):.3f} / {float(dh.mean()):.3f}"
    )


def _find_first_na_key(env: lmdb.Environment, max_scan: int) -> str | None:
    with env.begin(write=False) as txn:
        with txn.cursor() as cur:
            for i, (k, v) in enumerate(cur):
                if i >= max_scan:
                    break
                key = k.decode("utf-8")
                with io.BytesIO(v) as bio:
                    struct = TokenizedStructure.load_npz(bio)
                ct = struct.chain.chain_type.astype(np.int32, copy=False)
                if np.any(
                    (ct == C.chain.ChainType.DNA.value)
                    | (ct == C.chain.ChainType.RNA.value)
                ):
                    return key
    return None


def _iter_na_keys(env: lmdb.Environment, max_scan: int) -> list[str]:
    """Return NA-containing keys by scanning from the beginning up to max_scan entries."""
    keys: list[str] = []
    with env.begin(write=False) as txn:
        with txn.cursor() as cur:
            for i, (k, v) in enumerate(cur):
                if max_scan > 0 and i >= max_scan:
                    break
                key = k.decode("utf-8")
                with io.BytesIO(v) as bio:
                    struct = TokenizedStructure.load_npz(bio)
                ct = struct.chain.chain_type.astype(np.int32, copy=False)
                if np.any(
                    (ct == C.chain.ChainType.DNA.value)
                    | (ct == C.chain.ChainType.RNA.value)
                ):
                    keys.append(key)
    return keys


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True, help="Path to tokenized structure LMDB.")
    ap.add_argument("--record-id", default=None, help="LMDB key (record id).")
    ap.add_argument(
        "--list-keys", type=int, default=0, help="List first N keys and exit."
    )
    ap.add_argument(
        "--find-na",
        type=int,
        default=0,
        help="Scan up to N keys and pick the first record containing DNA/RNA chains.",
    )
    ap.add_argument(
        "--eval-na-batch",
        type=int,
        default=0,
        help=(
            "Evaluate many NA-containing records (count). Requires --find-na > 0 to scan."
        ),
    )
    ap.add_argument(
        "--steps-grid",
        type=int,
        nargs="+",
        default=None,
        help="Grid of Langevin num_steps values (e.g., 16 32 64 128).",
    )
    ap.add_argument(
        "--dt-grid",
        type=float,
        nargs="+",
        default=None,
        help="Grid of Langevin dt values (e.g., 0.25 0.1 0.05).",
    )
    ap.add_argument(
        "--bond-coef-grid",
        type=float,
        nargs="+",
        default=None,
        help="Grid of Langevin bond term coefficients (default: 2.0).",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to average over for RMSD evaluation (default: --seed only).",
    )
    ap.add_argument(
        "--out-dir", default=None, help="Optional output dir to write structures."
    )
    ap.add_argument(
        "--write", action="store_true", help="Write PDB outputs when --out-dir is set."
    )
    ap.add_argument(
        "--na-rmsd",
        action="store_true",
        help=(
            "Report RMSD(initial Gaussian X0 vs holo) and RMSD(after LD vs holo) "
            "for NA chains."
        ),
    )

    # ApoPerturbation toggles
    ap.add_argument(
        "--use-perturbation", action="store_true", help="Enable perturbation."
    )
    ap.add_argument("--prob-perturbation", type=float, default=1.0)
    ap.add_argument("--mask-nucleic-acids", action="store_true", default=False)

    # Langevin params
    ap.add_argument("--ld-steps", type=int, default=64)
    ap.add_argument("--ld-dt", type=float, default=0.25)
    ap.add_argument("--ld-res-r", type=float, default=4.0)
    ap.add_argument("--ld-ent-r", type=float, default=10.0)
    ap.add_argument("--ld-sphere-r", type=float, default=10.0)
    ap.add_argument("--ld-bond-coef", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--debug-na-bonds",
        type=int,
        default=0,
        help=(
            "Print N raw NA bond samples (init/after/holo distances) "
            "for this record/setting."
        ),
    )
    args = ap.parse_args()

    env = _open_lmdb(args.lmdb)
    if args.list_keys > 0:
        keys = _first_keys(env, args.list_keys)
        for k in keys:
            print(k)
        return

    if args.eval_na_batch > 0:
        if args.find_na <= 0:
            raise ValueError("--eval-na-batch requires --find-na > 0 (scan limit).")
        if not args.na_rmsd:
            raise ValueError("--eval-na-batch requires --na-rmsd.")

        steps_grid = (
            args.steps_grid if args.steps_grid is not None else [int(args.ld_steps)]
        )
        dt_grid = args.dt_grid if args.dt_grid is not None else [float(args.ld_dt)]
        bond_coef_grid = (
            args.bond_coef_grid
            if args.bond_coef_grid is not None
            else [float(args.ld_bond_coef)]
        )
        seeds = args.seeds if args.seeds is not None else [int(args.seed)]

        na_keys = _iter_na_keys(env, int(args.find_na))
        if not na_keys:
            raise RuntimeError(
                f"No NA-containing records found in first {args.find_na} keys."
            )
        na_keys = na_keys[: int(args.eval_na_batch)]
        print(
            f"[batch] NA keys: using {len(na_keys)} records (scan_limit={args.find_na})"
        )

        results: dict[tuple[int, float, float], dict[str, list[float]]] = {}
        for steps, dt, bc in itertools.product(steps_grid, dt_grid, bond_coef_grid):
            results[(int(steps), float(dt), float(bc))] = {
                "delta": [],
                "after_na_bond_mean": [],
                "after_na_bond_p95": [],
                "delta_na_bond_mean": [],
                "na_rms_radius": [],
                "init": [],
                "after": [],
            }

        for key in na_keys:
            struct = _load_struct(env, key)
            for steps, dt, bc in itertools.product(steps_grid, dt_grid, bond_coef_grid):
                for seed in seeds:
                    r = _na_eval_once(
                        struct,
                        steps=int(steps),
                        dt=float(dt),
                        res_r=float(args.ld_res_r),
                        ent_r=float(args.ld_ent_r),
                        sphere_r=float(args.ld_sphere_r),
                        bond_coef=float(bc),
                        seed=int(seed),
                    )
                    if r.get("na_chains", 0.0) <= 0.0:
                        continue
                    init_m = float(r["rmsd_init_mean"])
                    after_m = float(r["rmsd_after_mean"])
                    results[(int(steps), float(dt), float(bc))]["init"].append(init_m)
                    results[(int(steps), float(dt), float(bc))]["after"].append(after_m)
                    results[(int(steps), float(dt), float(bc))]["delta"].append(
                        after_m - init_m
                    )
                    results[(int(steps), float(dt), float(bc))][
                        "after_na_bond_mean"
                    ].append(float(r.get("after_na_bond_mean", 0.0)))
                    results[(int(steps), float(dt), float(bc))][
                        "after_na_bond_p95"
                    ].append(float(r.get("after_na_bond_p95", 0.0)))
                    results[(int(steps), float(dt), float(bc))][
                        "delta_na_bond_mean"
                    ].append(
                        float(
                            r.get("after_na_bond_mean", 0.0)
                            - r.get("init_na_bond_mean", 0.0)
                        )
                    )
                    results[(int(steps), float(dt), float(bc))]["na_rms_radius"].append(
                        float(r.get("na_rms_radius_mean", 0.0))
                    )

        print("\n=== NA RMSD grid summary (mean over chains per record) ===")
        print(
            "steps  dt      bondC   n_eval  "
            "init_mean  after_mean  delta_mean  delta_med  "
            "afterBondMean  afterBondP95  dBondMean  rmsRadius"
        )
        for steps, dt, bc in sorted(results.keys()):
            deltas = results[(steps, dt, bc)]["delta"]
            if not deltas:
                continue
            s_delta = _summarize(deltas)
            s_init = _summarize(results[(steps, dt, bc)]["init"])
            s_after = _summarize(results[(steps, dt, bc)]["after"])
            s_bm = _summarize(results[(steps, dt, bc)]["after_na_bond_mean"])
            s_bp95 = _summarize(results[(steps, dt, bc)]["after_na_bond_p95"])
            s_dbm = _summarize(results[(steps, dt, bc)]["delta_na_bond_mean"])
            s_rr = _summarize(results[(steps, dt, bc)]["na_rms_radius"])
            print(
                f"{steps:<5d}  {dt:<6.3f}  {bc:<6.2f}  {len(deltas):<6d}  "
                f"{s_init['mean']:<9.3f}  {s_after['mean']:<10.3f}  "
                f"{s_delta['mean']:<9.3f}  {s_delta['median']:<8.3f}  "
                f"{s_bm['mean']:<12.3f}  {s_bp95['mean']:<12.3f}  "
                f"{s_dbm['mean']:<9.3f}  {s_rr['mean']:<8.3f}"
            )

        print("\n=== Trend by steps (grouped by dt and bond_coef) ===")
        for dt in sorted(set(float(x) for x in dt_grid)):
            for bc in sorted(set(float(x) for x in bond_coef_grid)):
                rows = []
                for steps in sorted(set(int(s) for s in steps_grid)):
                    key2 = (steps, float(dt), float(bc))
                    deltas = results.get(key2, {}).get("delta", [])
                    if not deltas:
                        continue
                    rows.append(
                        (
                            steps,
                            _summarize(results[key2]["after"])["mean"],
                            _summarize(deltas)["mean"],
                            _summarize(results[key2]["after_na_bond_mean"])["mean"],
                            _summarize(results[key2]["after_na_bond_p95"])["mean"],
                            _summarize(results[key2]["na_rms_radius"])["mean"],
                        )
                    )
                if not rows:
                    continue
                print(
                    f"\n[dt={dt:.3f}, bond_coef={bc:.2f}] "
                    "steps -> afterRMSD / deltaRMSD / bondMean / bondP95 / rmsRadius"
                )
                for steps, a_rmsd, d_rmsd, b_m, b_p95, rr in rows:
                    print(
                        f"  {steps:<4d}: "
                        f"{a_rmsd:>7.3f} / {d_rmsd:>7.3f} / "
                        f"{b_m:>7.3f} / {b_p95:>7.3f} / {rr:>7.3f}"
                    )
        return

    key = args.record_id
    if key is None:
        if args.find_na > 0:
            key = _find_first_na_key(env, int(args.find_na))
            if key is None:
                raise RuntimeError(
                    f"Failed to find NA-containing record in first {args.find_na} keys."
                )
            print(f"[debug] record_id not provided; found NA record: {key}")
        else:
            keys = _first_keys(env, 1)
            if not keys:
                raise RuntimeError("LMDB has no keys.")
            key = keys[0]
            print(f"[debug] record_id not provided; using first key: {key}")

    struct = _load_struct(env, key)
    print(f"[debug] loaded {key}: {struct}")

    before = _compute_basic_stats(struct) | _bond_distance_stats(struct)

    apo_pert = ApoPerturbation(
        use_perturbation=bool(args.use_perturbation),
        prob_perturbation=float(args.prob_perturbation),
        mask_nucleic_acids=bool(args.mask_nucleic_acids),
        metric_lmdb_path=None,
        metric_comp=None,
        random_walk=None,
        langevin={
            "num_steps": int(args.ld_steps),
            "dt": float(args.ld_dt),
            "res_r": float(args.ld_res_r),
            "ent_r": float(args.ld_ent_r),
            "sphere_r": float(args.ld_sphere_r),
            "bond_coef": float(args.ld_bond_coef),
        },
        seed=int(args.seed),
    )

    out_struct = apo_pert.run(struct, rng=np.random.default_rng(args.seed))
    after = _compute_basic_stats(out_struct) | _bond_distance_stats(out_struct)

    def _fmt(d: dict[str, float]) -> str:
        return (
            f"apo_atoms={d['num_apo_atoms']:.0f}, "
            f"rms_r={d['rms_radius']:.3f}, "
            f"zero_frac={d['zero_frac']:.3f}, "
            f"bond_used={d['num_bonds_used']:.0f}, "
            f"bond_mean={d['bond_dist_mean']:.3f}, "
            f"bond_p95={d['bond_dist_p95']:.3f}"
        )

    print("[before]", _fmt(before))
    print("[after ]", _fmt(after))

    if args.debug_na_bonds > 0:
        _debug_print_na_bond_samples(
            struct,
            steps=int(args.ld_steps),
            dt=float(args.ld_dt),
            res_r=float(args.ld_res_r),
            ent_r=float(args.ld_ent_r),
            sphere_r=float(args.ld_sphere_r),
            bond_coef=float(args.ld_bond_coef),
            seed=int(args.seed),
            n_samples=int(args.debug_na_bonds),
        )

    if args.out_dir is not None and args.write:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{key}_with_apo.pdb"
        out_struct.to_pdb(str(out_path), conformer_id=0, is_predicted=True, save_apo=True)
        print(f"[debug] wrote {out_path}")


if __name__ == "__main__":
    main()
