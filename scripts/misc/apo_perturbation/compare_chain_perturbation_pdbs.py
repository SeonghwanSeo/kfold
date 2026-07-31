#!/usr/bin/env python3
"""Create before/after PDBs for chain-wise perturbation of apo inputs."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from string import Template

import numpy as np
import torch
from kfold.data.utils.writer.pdb import to_pdbstring
from kfold.model.modules.structure_module.kfold_ecsi import KFoldECSI
from kfold.training.dataset.dataset import MultiTrainingDataset, TrainingDataset
from omegaconf import DictConfig

import kfold.model.modules as submodules  # noqa: F401
from kfold.config import load_config
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.tokenized import TokenizedStructure
from kfold.training.dataset.datamodule import TrainingDataModule
from kfold.training.dataset.sampler.base import Sample
from kfold.utils import errors
from kfold.utils.registry import Registry

_ASSIGN_RE = re.compile(r"^(?:export\s+)?([A-Z0-9_]+)=(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create multi-model PDBs comparing original apo coordinates to "
            "chain-wise perturbed coordinates."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Training config yaml to load directly (for example "
            "configs/train-esm2-edm-mini.yaml). If set, skips "
            "--multinode_script."
        ),
    )
    parser.add_argument(
        "--multinode_script",
        type=Path,
        default=None,
        help=(
            "Path to multinode_new.sh with the current training settings. "
            "Defaults to ./multinode_new.sh when --config is not set."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./tmp/apo_chain_perturbation"),
        help="Directory to write multi-model PDB comparisons.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of dataset samples to export.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to process per batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for sample selection and perturbation randomness.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for perturbation.",
    )
    parser.add_argument(
        "--keep_temp_config",
        action="store_true",
        help="Keep the generated temp config for debugging.",
    )
    return parser.parse_args()


def _strip_quotes(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _normalize_bash_value(value: str) -> str:
    value = _strip_quotes(value.strip())
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered
    return value


def parse_bash_assignments(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGN_RE.match(stripped)
        if not match:
            continue
        key, value = match.groups()
        values[key] = _normalize_bash_value(value)
    return values


def apply_multinode_logic(values: dict[str, str]) -> None:
    radius = values.get("CHAIN_COM_SAMPLING_RADIUS")
    translation = values.get("APO_TRANSLATION_SCALE")
    if radius is None or translation is None:
        return
    try:
        radius_val = float(radius)
    except ValueError:
        return
    if radius_val > 0:
        values["APO_TRANSLATION_SCALE"] = "0.0"


def extract_temp_config_template(lines: list[str]) -> str:
    start = None
    for i, line in enumerate(lines):
        if "cat <<EOF" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError("Could not find temp config heredoc in multinode script.")
    for j in range(start, len(lines)):
        if lines[j].strip() == "EOF":
            return "\n".join(lines[start:j]) + "\n"
    raise ValueError("Temp config heredoc does not terminate with EOF.")


def substitute_template(template: str, values: dict[str, str]) -> str:
    return Template(template).safe_substitute(values)


def parse_overrides(lines: list[str], values: dict[str, str]) -> list[str]:
    overrides: list[str] = []
    in_override = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if "--override" in stripped:
            in_override = True
            remainder = stripped.split("--override", 1)[1].strip()
            if remainder:
                overrides.extend(_parse_override_line(remainder, values))
            if not stripped.endswith("\\"):
                in_override = False
            continue

        if not in_override:
            continue

        overrides.extend(_parse_override_line(stripped, values))
        if not stripped.endswith("\\"):
            in_override = False

    return overrides


def _parse_override_line(line: str, values: dict[str, str]) -> list[str]:
    if line.startswith("--"):
        return []
    cleaned = line.rstrip("\\").rstrip().strip()
    if cleaned.endswith('"') and cleaned.count('"') % 2 == 1:
        cleaned = cleaned[:-1]
    if cleaned.endswith("'") and cleaned.count("'") % 2 == 1:
        cleaned = cleaned[:-1]
    if "=" not in cleaned:
        return []
    cleaned = cleaned.replace('\\"', '"').replace("\\'", "'")
    cleaned = Template(cleaned).safe_substitute(values)
    cleaned = cleaned.strip()
    return [cleaned] if cleaned else []


def build_config_from_multinode(
    multinode_script: Path, keep_temp_config: bool
) -> tuple[DictConfig, dict[str, str]]:
    lines = multinode_script.read_text().splitlines()
    values = parse_bash_assignments(lines)
    apply_multinode_logic(values)
    template = extract_temp_config_template(lines)
    filled_template = substitute_template(template, values)
    overrides = parse_overrides(lines, values)

    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    temp_path = None
    cfg = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="multinode_",
            dir=configs_dir,
            delete=False,
        ) as handle:
            handle.write(filled_template)
            temp_path = Path(handle.name)
        cfg = load_config(temp_path, override_args=overrides)
    finally:
        if temp_path is not None and not keep_temp_config:
            temp_path.unlink(missing_ok=True)
    return cfg, values


def build_config_from_path(config_path: Path) -> tuple[DictConfig, dict[str, str]]:
    cfg = load_config(config_path)
    return cfg, {}


def resolve_sample(
    dataset: MultiTrainingDataset, index: int
) -> tuple[TrainingDataset, Sample]:
    dataset_idx = np.searchsorted(dataset.cumulative_sizes, index, side="right")
    if dataset_idx == 0:
        sample_idx = index
    else:
        sample_idx = index - dataset.cumulative_sizes[dataset_idx - 1]
    return dataset.datasets[dataset_idx], dataset.datasets[dataset_idx].samples[
        sample_idx
    ]


def build_cropped_sample(
    dataset: TrainingDataset, sample: Sample
) -> tuple[FoldingInput, TokenizedStructure, str]:
    metadata = sample.metadata
    if dataset.seed is not None:
        rng = np.random.default_rng(dataset.seed + hash(metadata.id) % (1 << 15))
    else:
        rng = np.random.default_rng()

    ref_struct = dataset.load_ref_structure(metadata)
    ref_struct = dataset.extract_substructure(
        ref_struct, rng=rng, asym_ids=sample.asym_id
    )
    dataset.load_apo_structure(ref_struct, rng=rng)
    struct = dataset.tokenize(ref_struct, rng=rng)
    cropped_struct = dataset.crop_structure(struct, rng=rng, asym_ids=sample.asym_id)
    f_input = dataset.featurize(cropped_struct, metadata, rng=rng)
    f_input = dataset.pad_input(f_input)
    return f_input, cropped_struct, metadata.id


def format_asym_id(asym_id: int | tuple[int, int] | None) -> str:
    if asym_id is None:
        return "all"
    if isinstance(asym_id, tuple):
        return "-".join(str(v) for v in asym_id)
    return str(asym_id)


def remap_asym_ids(struct: TokenizedStructure) -> TokenizedStructure:
    chain_asym_ids = struct.chain.asym_id
    unique_ids = list(chain_asym_ids.tolist())
    mapping = {
        int(old_id): int(new_id) for new_id, old_id in enumerate(unique_ids, start=1)
    }

    def _apply_map(values: np.ndarray) -> np.ndarray:
        updated = values.copy()
        for old_id, new_id in mapping.items():
            updated[values == old_id] = new_id
        return updated

    chain = struct.chain.copy_with(asym_id=_apply_map(struct.chain.asym_id))
    residue = struct.residue.copy_with(asym_id=_apply_map(struct.residue.asym_id))
    token = struct.token.copy_with(asym_id=_apply_map(struct.token.asym_id))
    bond = struct.bond.copy_with(asym_id=_apply_map(struct.bond.asym_id))
    return struct.copy_with(chain=chain, residue=residue, token=token, bond=bond)


def write_multimodel_pdb(
    struct: TokenizedStructure,
    traj_coords: np.ndarray,
    out_path: Path,
) -> None:
    struct = remap_asym_ids(struct)
    with open(out_path, "w") as handle:
        for frame_idx, coords in enumerate(traj_coords):
            handle.write(f"MODEL     {frame_idx + 1}\n")
            trajectory_struct = struct.replace_atom_coords(
                atom_coords=coords, is_apo=True
            )
            pdb_string = to_pdbstring(trajectory_struct, save_apo=True).rstrip()
            if pdb_string.endswith("END"):
                pdb_string = pdb_string[:-3].rstrip()
            lines = [
                line
                for line in pdb_string.split("\n")
                if not line.strip().startswith("TER")
            ]
            handle.write("\n".join(lines) + "\n")
            if frame_idx < len(traj_coords) - 1:
                handle.write("ENDMDL\n")
        handle.write("END\n")


def ensure_chain_augment(structure_module: KFoldECSI) -> KFoldECSI:
    if not hasattr(structure_module, "apply_chain_random_augmentation"):
        raise AttributeError(
            "Structure module does not support chain-wise augmentation. "
            "Expected apply_chain_random_augmentation()."
        )
    return structure_module


def main() -> None:
    args = parse_args()
    if args.config is not None:
        if args.multinode_script is not None:
            raise ValueError("Use either --config or --multinode_script, not both.")
        if not args.config.exists():
            raise FileNotFoundError(f"Config not found: {args.config}")
        cfg, _values = build_config_from_path(args.config)
    else:
        multinode_script = args.multinode_script or Path("./multinode_new.sh")
        if not multinode_script.exists():
            raise FileNotFoundError(f"Multinode script not found: {multinode_script}")
        cfg, _values = build_config_from_multinode(
            multinode_script, args.keep_temp_config
        )

    data_module = TrainingDataModule(cfg.train.data)
    train_dataset = data_module.construct_train_dataset()

    rng = np.random.default_rng(args.seed)
    total_samples = len(train_dataset)
    target_samples = min(args.num_samples, total_samples)
    if target_samples < args.num_samples:
        print(
            f"Requested {args.num_samples} samples, but dataset has {total_samples}. "
            f"Using {target_samples}."
        )

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")

    indices = rng.permutation(total_samples).tolist()[:target_samples]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    score_model = Registry.instantiate(
        cfg.model.score_model, kernel_config=cfg.model.kernel
    )
    structure_module = Registry.instantiate(
        cfg.model.structure_module, score_model=score_model
    )
    structure_module: KFoldECSI = ensure_chain_augment(structure_module)

    if hasattr(structure_module, "coordinate_augmentation") and not getattr(
        structure_module, "coordinate_augmentation", False
    ):
        print("Warning: coordinate_augmentation is disabled; before/after may match.")

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    written = 0
    for batch_start in range(0, target_samples, args.batch_size):
        batch_indices = indices[batch_start : batch_start + args.batch_size]
        batch_inputs: list[FoldingInput] = []
        batch_structs: list[TokenizedStructure] = []
        batch_ids: list[str] = []
        batch_asym_ids: list[str] = []
        batch_sample_indices: list[int] = []

        for idx in batch_indices:
            try:
                dataset, sample = resolve_sample(train_dataset, idx)
                f_input, struct, sample_id = build_cropped_sample(dataset, sample)
            except Exception as exc:
                print(f"Skipping index {idx} due to load error: {exc}")
                continue

            batch_inputs.append(f_input)
            batch_structs.append(struct)
            batch_ids.append(sample_id)
            batch_asym_ids.append(format_asym_id(sample.asym_id))
            batch_sample_indices.append(idx)

        if not batch_inputs:
            continue

        f_input = FoldingInput.from_list(batch_inputs, pad_to_max=True).to(device)

        if args.seed is not None:
            torch.manual_seed(args.seed + batch_start)

        apo_coords = f_input.atom.apo_coords
        apo_mask = f_input.atom.apo_mask
        before_coords = apo_coords.clone()
        after_coords = structure_module.apply_chain_random_augmentation(
            apo_coords, apo_mask, f_input
        )

        valid_mask = f_input.atom.pad_mask & f_input.atom.apo_mask
        batch_items = zip(
            batch_structs,
            batch_ids,
            batch_asym_ids,
            batch_sample_indices,
            strict=True,
        )
        for batch_idx, (struct, sample_id, asym_id, sample_idx) in enumerate(batch_items):
            before_np = before_coords[batch_idx].cpu().numpy()
            after_np = after_coords[batch_idx].cpu().numpy()
            traj_np = np.stack([before_np, after_np], axis=0)

            invalid_mask = ~valid_mask[batch_idx].cpu().numpy()
            traj_np[:, invalid_mask] = np.nan

            out_name = f"{sample_id}_idx{sample_idx}_asym{asym_id}-chain-perturb.pdb"
            out_path = args.out_dir / out_name

            try:
                write_multimodel_pdb(struct, traj_np, out_path)
            except errors.PDBWriterMaxChainError as exc:
                print(f"Skipping {sample_id} (too many chains for PDB): {exc}")
                continue

            written += 1
            if written % 10 == 0 or written == target_samples:
                print(f"Wrote {written}/{target_samples} comparisons.")

    print(f"Done. Generated {written} comparison PDBs in {args.out_dir}.")


if __name__ == "__main__":
    main()
