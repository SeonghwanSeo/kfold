#!/usr/bin/env python3
"""Convert prediction trajectory NPZ files into multi-model PDBs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ATOM_RECORDS = {"ATOM", "HETATM"}
SKIP_RECORDS = {"MODEL", "ENDMDL", "END", "TER"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert predictions.npz trajectories to multi-model PDB files."
    )
    parser.add_argument(
        "--predictions_npz",
        type=Path,
        required=True,
        help="Path to a *-predictions.npz file containing a traj array.",
    )
    parser.add_argument(
        "--template_pdb",
        type=Path,
        default=None,
        help=(
            "Template PDB to copy atom records from. Defaults to "
            "<predictions_stem>-apo.pdb in the same directory."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for trajectory PDBs (defaults to predictions dir).",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help="Explicit output PDB path (only valid without --all_samples).",
    )
    parser.add_argument(
        "--traj_key",
        type=str,
        default="traj",
        help="Key name inside the NPZ for the trajectory array.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Sample index to export when the trajectory has multiple samples.",
    )
    parser.add_argument(
        "--all_samples",
        action="store_true",
        help="Export all sample trajectories in the NPZ.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Stride for trajectory frames (use >1 to downsample).",
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="Start frame index (inclusive).",
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=None,
        help="End frame index (exclusive). Defaults to the final frame.",
    )
    return parser.parse_args()


def infer_base_name(predictions_path: Path) -> str:
    stem = predictions_path.stem
    if stem.endswith("-predictions"):
        return stem[: -len("-predictions")]
    return stem


def infer_template_path(predictions_path: Path) -> Path:
    base_name = infer_base_name(predictions_path)
    return predictions_path.parent / f"{base_name}-apo.pdb"


def load_template_lines(template_path: Path) -> list[str]:
    if not template_path.exists():
        raise FileNotFoundError(f"Template PDB not found: {template_path}")
    lines = template_path.read_text().splitlines()
    return [line for line in lines if line[:6].strip().upper() not in SKIP_RECORDS]


def count_atom_lines(lines: list[str]) -> int:
    return sum(1 for line in lines if line[:6].strip().upper() in ATOM_RECORDS)


def normalize_traj(traj: np.ndarray) -> np.ndarray:
    if traj.ndim == 3:
        if traj.shape[-1] != 3:
            raise ValueError(f"Expected last dimension to be 3, got {traj.shape}.")
        return traj[:, None, :, :]
    if traj.ndim != 4 or traj.shape[-1] != 3:
        raise ValueError(
            f"Expected traj shape (T, N, A, 3) or (T, A, 3), got {traj.shape}."
        )
    return traj


def replace_coords(line: str, coords: np.ndarray) -> str:
    if len(line) < 54:
        line = line.ljust(54)
    prefix = line[:30]
    suffix = line[54:]
    x, y, z = coords.tolist()
    updated = f"{prefix}{x:8.3f}{y:8.3f}{z:8.3f}{suffix}"
    return updated.ljust(80)


def render_frame(template_lines: list[str], coords: np.ndarray) -> list[str]:
    output_lines: list[str] = []
    coord_idx = 0
    for line in template_lines:
        record = line[:6].strip().upper()
        if record in ATOM_RECORDS:
            if coord_idx >= len(coords):
                raise ValueError("Not enough coordinates for template atom lines.")
            atom_coords = coords[coord_idx]
            coord_idx += 1
            if not np.isfinite(atom_coords).all():
                continue
            output_lines.append(replace_coords(line, atom_coords))
        else:
            output_lines.append(line)
    if coord_idx != len(coords):
        raise ValueError(f"Template has {coord_idx} atoms, but coords has {len(coords)}.")
    return output_lines


def write_multimodel_pdb(
    template_lines: list[str],
    traj_coords: np.ndarray,
    out_path: Path,
) -> None:
    with out_path.open("w") as handle:
        for frame_idx, coords in enumerate(traj_coords):
            handle.write(f"MODEL     {frame_idx + 1}\n")
            frame_lines = render_frame(template_lines, coords)
            handle.write("\n".join(frame_lines) + "\n")
            if frame_idx < len(traj_coords) - 1:
                handle.write("ENDMDL\n")
        handle.write("END\n")


def main() -> None:
    args = parse_args()
    pred_path = args.predictions_npz
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions NPZ not found: {pred_path}")

    template_path = args.template_pdb or infer_template_path(pred_path)
    template_lines = load_template_lines(template_path)
    num_atoms = count_atom_lines(template_lines)

    with np.load(pred_path, allow_pickle=False) as data:
        if args.traj_key not in data:
            raise KeyError(f"traj key '{args.traj_key}' not found in {pred_path.name}.")
        traj = data[args.traj_key]

    traj = normalize_traj(traj)
    if traj.shape[2] != num_atoms:
        raise ValueError(f"Template has {num_atoms} atoms, but traj has {traj.shape[2]}.")

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1.")

    frame_end = args.frame_end if args.frame_end is not None else traj.shape[0]
    traj = traj[args.frame_start : frame_end : args.frame_stride]
    if traj.shape[0] == 0:
        raise ValueError("No frames selected after applying frame slicing.")

    out_dir = args.out_dir or pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_path and args.all_samples:
        raise ValueError("--out_path cannot be used with --all_samples.")

    if args.all_samples:
        sample_indices = range(traj.shape[1])
    else:
        if not (0 <= args.sample_index < traj.shape[1]):
            raise ValueError(
                f"--sample_index {args.sample_index} is out of range (0.."
                f"{traj.shape[1] - 1})."
            )
        sample_indices = [args.sample_index]

    base_name = infer_base_name(pred_path)
    for sample_idx in sample_indices:
        if args.out_path is not None:
            out_path = args.out_path
        else:
            out_name = f"{base_name}-traj-s{sample_idx}.pdb"
            out_path = out_dir / out_name
        write_multimodel_pdb(template_lines, traj[:, sample_idx], out_path)


if __name__ == "__main__":
    main()
