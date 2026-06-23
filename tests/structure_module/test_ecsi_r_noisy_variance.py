from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path

import torch
from kfold.data.pipelines._apo_perturbation import ApoPerturbationConfig
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.model.modules.structure_module.kfold_ecsi import KFoldECSI
from omegaconf import DictConfig, OmegaConf

from kfold.config import load_config
from kfold.data.types.model_input import FoldingInput
from kfold.training.dataset.datamodule import TrainingDataModule


class DummyScoreModel(BaseScoreModel):
    """Minimal score model stub for structure module initialization."""

    def __init__(self) -> None:
        super().__init__(cfg=None, kernel_config=None)

    def forward(  # type: ignore[override]
        self,
        r_noisy: torch.Tensor,
        c_noise: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache: dict | None = None,
    ) -> torch.Tensor:
        return torch.zeros_like(r_noisy[..., :3])


@dataclasses.dataclass
class ScalarStats:
    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.double().reshape(-1)
        self.count += values.numel()
        self.sum += values.sum().item()
        self.sumsq += (values * values).sum().item()

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def var(self) -> float:
        if self.count == 0:
            return 0.0
        mean = self.mean()
        var = self.sumsq / self.count - mean * mean
        return max(var, 0.0)


@dataclasses.dataclass
class PairStats:
    count: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def clear(self) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.double().reshape(-1)
        y = y.double().reshape(-1)
        if x.numel() != y.numel():
            raise ValueError("x and y must have the same number of elements.")
        self.count += x.numel()
        self.sum_x += x.sum().item()
        self.sum_x2 += (x * x).sum().item()
        self.sum_y += y.sum().item()
        self.sum_y2 += (y * y).sum().item()
        self.sum_xy += (x * y).sum().item()

    def mean_x(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_x / self.count

    def mean_y(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_y / self.count

    def var_x(self) -> float:
        if self.count == 0:
            return 0.0
        mean_x = self.mean_x()
        var = self.sum_x2 / self.count - mean_x * mean_x
        return max(var, 0.0)

    def var_y(self) -> float:
        if self.count == 0:
            return 0.0
        mean_y = self.mean_y()
        var = self.sum_y2 / self.count - mean_y * mean_y
        return max(var, 0.0)

    def sigma_x(self) -> float:
        return math.sqrt(self.var_x())

    def sigma_y(self) -> float:
        return math.sqrt(self.var_y())

    def cov_xy(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_xy / self.count - self.mean_x() * self.mean_y()


@dataclasses.dataclass
class VarianceStats:
    total: ScalarStats = dataclasses.field(default_factory=ScalarStats)
    axis: list[ScalarStats] = dataclasses.field(
        default_factory=lambda: [ScalarStats() for _ in range(3)]
    )

    def update(self, coords: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.bool()
        mask_expanded = mask[:, None, :, None].expand_as(coords)
        self.total.update(coords[mask_expanded])
        mask_axis = mask[:, None, :].expand(
            coords.shape[0], coords.shape[1], coords.shape[2]
        )
        for axis_idx in range(3):
            self.axis[axis_idx].update(coords[..., axis_idx][mask_axis])

    def variances(self) -> tuple[float, float, float, float]:
        return (
            self.total.var(),
            self.axis[0].var(),
            self.axis[1].var(),
            self.axis[2].var(),
        )


def _parse_times(
    raw_times: str | None, time_min: float, time_max: float, num_times: int
) -> list[float]:
    if raw_times:
        return [float(value.strip()) for value in raw_times.split(",") if value.strip()]
    return torch.linspace(time_min, time_max, steps=num_times).tolist()


def _build_data_module(global_config: DictConfig) -> TrainingDataModule:
    global_config.train.data.train_batch_size = 1
    global_config.train.data.val_batch_size = 1
    global_config.train.data.num_workers = 0
    global_config.train.data.safe_load = False
    global_config.train.data.train_datasets[0].apo_init.chain_com_sampling_radius = 30
    global_config.train.data.val_datasets[0].apo_init.chain_com_sampling_radius = 30
    data_module = TrainingDataModule(global_config.train.data)
    return data_module


def _build_structure_module(global_config: DictConfig) -> KFoldECSI:
    score_model = DummyScoreModel()
    return KFoldECSI(global_config.model.structure_module, score_model)


def _clone_config(config_obj: object) -> DictConfig:
    if dataclasses.is_dataclass(config_obj):
        return OmegaConf.create(dataclasses.asdict(config_obj))
    return OmegaConf.create(OmegaConf.to_container(config_obj, resolve=True))


def _get_apo_perturbation_template(global_config: DictConfig, source: str) -> DictConfig:
    if source == "minimal":
        return _clone_config(ApoPerturbationConfig())

    if source == "train":
        train_datasets = global_config.train.data.train_datasets
        if train_datasets:
            train_apo = train_datasets[0].apo_init.apo_perturbation
            if train_apo is not None:
                return _clone_config(train_apo)

    default_path = Path("configs/dataset/default.yaml")
    if default_path.exists():
        default_cfg = load_config(default_path)
        default_apo = default_cfg.apo_init.apo_perturbation
        if default_apo is not None:
            return _clone_config(default_apo)

    return _clone_config(ApoPerturbationConfig())


def _maybe_disable_rieprody(apo_init: DictConfig, data_path: str | Path) -> None:
    apo_perturbation = apo_init.apo_perturbation
    if apo_perturbation is None:
        return
    if "rieprody" not in apo_perturbation or apo_perturbation.rieprody is None:
        return
    metric_path = Path(data_path) / "rieprody_metric.lmdb"
    if not metric_path.exists():
        apo_perturbation.rieprody = None


def _maybe_enable_val_perturbation(
    global_config: DictConfig,
    use_perturbation: bool,
    prob_perturbation: float,
    source: str,
    override: bool,
) -> None:
    if not use_perturbation:
        return

    val_datasets = global_config.train.data.val_datasets
    if not val_datasets:
        raise ValueError("val_datasets is empty; cannot enable perturbation.")

    template = _get_apo_perturbation_template(global_config, source)
    for dataset_cfg in val_datasets:
        apo_init = dataset_cfg.apo_init
        apo_init.use_perturbation = True
        apo_init.prob_perturbation = prob_perturbation
        apo_init.prob_replace_to_holo = 0.0
        if override or apo_init.apo_perturbation is None:
            apo_init.apo_perturbation = _clone_config(template)
        _maybe_disable_rieprody(apo_init, dataset_cfg.data_path)


def _update_pair_stats(
    pair_stats: PairStats,
    label_coords: torch.Tensor,
    prior_coords: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    mask_expanded = mask[:, None, :, None].expand_as(label_coords)
    pair_stats.update(label_coords[mask_expanded], prior_coords[mask_expanded])


def _bridge_c_in(
    structure_module: KFoldECSI,
    t_hat: torch.Tensor,
    sigma_data: float,
    sigma_data_end: float,
    cov_xy: float,
) -> torch.Tensor:
    t_exp = t_hat[..., None, None]
    alpha_t = structure_module.si_coeffs.alpha(t_exp)
    beta_t = structure_module.si_coeffs.beta(t_exp)
    gamma_t = structure_module.si_coeffs.gamma(t_exp)
    a_term = alpha_t**2 * sigma_data**2
    b_term = beta_t**2 * sigma_data_end**2
    cov_term = 2 * alpha_t * beta_t * cov_xy
    denom = a_term + b_term + cov_term + gamma_t**2
    return 1.0 / torch.sqrt(denom + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Measure r_noisy variance across interpolation times for KFoldECSI.")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train-esm2-ecsi-mini.yaml"),
        help="Config path to load model and data settings.",
    )
    parser.add_argument(
        "--num_diffusion_samples",
        type=int,
        default=64,
        help="Number of diffusion samples for interpolation.",
    )
    parser.add_argument(
        "--num_times",
        type=int,
        default=11,
        help=(
            "Number of interpolation times between sampling_time_min and "
            "sampling_time_max."
        ),
    )
    parser.add_argument(
        "--times",
        type=str,
        default=None,
        help="Comma-separated list of interpolation times (overrides --num_times).",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=8,
        help="Number of validation batches to use for statistics.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("val", "train"),
        default="val",
        help="Dataset split to use for statistics.",
    )
    parser.add_argument(
        "--val_use_perturbation",
        action="store_true",
        help="Enable apo perturbation for validation datasets.",
    )
    parser.add_argument(
        "--val_prob_perturbation",
        type=float,
        default=1.0,
        help="Probability of apo perturbation in validation datasets.",
    )
    parser.add_argument(
        "--val_perturbation_source",
        type=str,
        choices=("train", "default", "minimal"),
        default="train",
        help="Source config for validation apo perturbation settings.",
    )
    parser.add_argument(
        "--val_perturbation_override",
        action="store_true",
        help="Override existing validation apo_perturbation config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run interpolation on (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Optional CSV output path for the variance table.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    global_config = load_config(args.config)
    if args.split == "val":
        _maybe_enable_val_perturbation(
            global_config,
            args.val_use_perturbation,
            args.val_prob_perturbation,
            args.val_perturbation_source,
            args.val_perturbation_override,
        )
    data_module = _build_data_module(global_config)
    if args.split == "train":
        data_module.setup("fit")
        dataloader = data_module.train_dataloader()
    else:
        data_module.setup("validate")
        dataloader = data_module.val_dataloader()
    structure_module = _build_structure_module(global_config)

    time_values = _parse_times(
        args.times,
        structure_module.sampling.time_min,
        structure_module.sampling.time_max,
        args.num_times,
    )

    pair_stats = PairStats()
    torch.manual_seed(args.seed)
    empirical_sigma_data = []
    empirical_sigma_data_end = []
    empirical_cov_xy = []
    for batch_idx, data in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        f_input, _ = data
        f_input = f_input.to(device)

        mask = f_input.atom.pad_mask
        num_samples = args.num_diffusion_samples
        label_coords = structure_module.sample_label(f_input, num_samples)
        prior_coords = structure_module.sample_prior(f_input, num_samples, label_coords)

        if structure_module.normalize_coordinate:
            label_coords = label_coords / structure_module.sigma_data
            prior_coords = prior_coords / structure_module.sigma_data_end

        _update_pair_stats(pair_stats, label_coords, prior_coords, mask)
        sigma_x = pair_stats.sigma_x()
        sigma_y = pair_stats.sigma_y()
        cov_xy = pair_stats.cov_xy()
        empirical_sigma_data.append(sigma_x)
        empirical_sigma_data_end.append(sigma_y)
        empirical_cov_xy.append(cov_xy)
        pair_stats.clear()

    if len(empirical_sigma_data) == 0:
        raise RuntimeError("No valid atoms found while computing empirical stats.")

    empirical_sigma_data = sum(empirical_sigma_data) / len(empirical_sigma_data)
    empirical_sigma_data_end = sum(empirical_sigma_data_end) / len(
        empirical_sigma_data_end
    )
    empirical_cov_xy = sum(empirical_cov_xy) / len(empirical_cov_xy)

    structure_module.sigma_data = 16
    structure_module.sigma_data_end = 16
    structure_module.cov_xy = 120

    print(
        f"Config stats: sigma_data={structure_module.sigma_data:.6f} "
        f"sigma_data_end={structure_module.sigma_data_end:.6f} "
        f"cov_xy={structure_module.cov_xy:.6f}"
    )
    print(
        f"Empirical stats: sigma_data={empirical_sigma_data:.6f} "
        f"sigma_data_end={empirical_sigma_data_end:.6f} "
        f"cov_xy={empirical_cov_xy:.6f} "
        f"count={pair_stats.count}"
    )
    print(
        "Suggested overrides: "
        f"model.structure_module.sigma_data={empirical_sigma_data:.6f} "
        f"model.structure_module.sigma_data_end={empirical_sigma_data_end:.6f} "
        f"model.structure_module.cov_xy={empirical_cov_xy:.6f}"
    )

    sum_vars_config = [[0.0] * 4 for _ in time_values]
    sum_vars_empirical = [[0.0] * 4 for _ in time_values]
    num_batches_processed = 0

    torch.manual_seed(args.seed)
    for batch_idx, data in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        num_batches_processed += 1

        f_input, _ = data
        f_input = f_input.to(device)

        mask = f_input.atom.pad_mask
        num_samples = args.num_diffusion_samples
        label_coords = structure_module.sample_label(f_input, num_samples)
        prior_coords = structure_module.sample_prior(f_input, num_samples, label_coords)

        if structure_module.normalize_coordinate:
            label_coords = label_coords / structure_module.sigma_data
            prior_coords = prior_coords / structure_module.sigma_data_end

        for time_idx, t_value in enumerate(time_values):
            t_hat = torch.full(
                (f_input.batch_size, num_samples),
                float(t_value),
                device=f_input.device,
                dtype=label_coords.dtype,
            )
            noised_coords = structure_module.interpolate(
                prior_coords, label_coords, t_hat, mask
            )
            r_noisy = structure_module.c_in(t_hat[..., None, None]) * noised_coords

            batch_stats_cfg = VarianceStats()
            batch_stats_cfg.update(r_noisy, mask)
            for i, val in enumerate(batch_stats_cfg.variances()):
                sum_vars_config[time_idx][i] += val

            c_in_emp = _bridge_c_in(
                structure_module,
                t_hat,
                empirical_sigma_data,
                empirical_sigma_data_end,
                empirical_cov_xy,
            )
            r_noisy_empirical = c_in_emp * noised_coords

            batch_stats_emp = VarianceStats()
            batch_stats_emp.update(r_noisy_empirical, mask)
            for i, val in enumerate(batch_stats_emp.variances()):
                sum_vars_empirical[time_idx][i] += val

    avg_vars_config = []
    avg_vars_empirical = []
    if num_batches_processed > 0:
        for i in range(len(time_values)):
            avg_vars_config.append(
                tuple(val / num_batches_processed for val in sum_vars_config[i])
            )
            avg_vars_empirical.append(
                tuple(val / num_batches_processed for val in sum_vars_empirical[i])
            )
    else:
        avg_vars_config = [(0.0,) * 4] * len(time_values)
        avg_vars_empirical = [(0.0,) * 4] * len(time_values)

    header = [
        "t",
        "var_cfg",
        "var_emp",
        "var_x_cfg",
        "var_x_emp",
        "var_y_cfg",
        "var_y_emp",
        "var_z_cfg",
        "var_z_emp",
    ]
    print("\t".join(header))
    for time_idx, t_value in enumerate(time_values):
        var_cfg, var_x_cfg, var_y_cfg, var_z_cfg = avg_vars_config[time_idx]
        var_emp, var_x_emp, var_y_emp, var_z_emp = avg_vars_empirical[time_idx]
        print(
            f"{t_value:.6f}\t{var_cfg:.6f}\t{var_emp:.6f}\t"
            f"{var_x_cfg:.6f}\t{var_x_emp:.6f}\t"
            f"{var_y_cfg:.6f}\t{var_y_emp:.6f}\t"
            f"{var_z_cfg:.6f}\t{var_z_emp:.6f}"
        )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            for time_idx, t_value in enumerate(time_values):
                var_cfg, var_x_cfg, var_y_cfg, var_z_cfg = avg_vars_config[time_idx]
                var_emp, var_x_emp, var_y_emp, var_z_emp = avg_vars_empirical[time_idx]
                writer.writerow(
                    [
                        f"{t_value:.6f}",
                        f"{var_cfg:.6f}",
                        f"{var_emp:.6f}",
                        f"{var_x_cfg:.6f}",
                        f"{var_x_emp:.6f}",
                        f"{var_y_cfg:.6f}",
                        f"{var_y_emp:.6f}",
                        f"{var_z_cfg:.6f}",
                        f"{var_z_emp:.6f}",
                    ]
                )


if __name__ == "__main__":
    main()
