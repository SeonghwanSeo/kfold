"""Module for apo structure perturbation using RieProDy."""

import dataclasses
import os
import pickle
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch

import kfold.constants as C

from .rigid_align import compute_rmsd

ATOM37_ORDER: dict[str, int] = C.atom.protein_atom37_order


@dataclasses.dataclass(kw_only=True)
class MetricCompConfig:
    # Metric computation parameters
    apo_internal_coord_metric_calculation_device: str = "cpu"
    apo_internal_coord_metric_calculation_precision: str = "float32"
    apo_internal_coord_metric_save_precision: str = "float32"
    # RieProDy's __init__ accesses this path (we don't call preprocess()).
    apo_internal_coord_metric_information_path: str = str(Path("."))


@dataclasses.dataclass(kw_only=True)
class RandomWalkConfig:
    # Random walk parameters
    apo_internal_coord_random_walk_device: str = "cpu"
    apo_internal_coord_random_walk_precision: str = "float32"
    consider_side_chain_in_metric: bool = False
    total_time: float = 15.0
    total_time_min: float = 0.0
    fixed_bb_angles: tuple[str, ...] = ()
    num_steps: int = 1
    metric_calculation_period: int = 100
    metric: str = "lagrangian"
    use_precomputed_metric: bool = True
    with_christoffel_term: bool = False
    kabsch_aligned_traj: bool = False


@dataclasses.dataclass(kw_only=True)
class LangevinConfig:
    # Langevin dynamics parameters
    num_steps: int = 64
    dt: float = 0.25
    res_r: float = 4.0
    ent_r: float = 10.0
    sphere_r: float = 10.0


@dataclasses.dataclass
class RieProdyModuleConfig:
    metric_comp: MetricCompConfig = dataclasses.field(default_factory=MetricCompConfig)
    random_walk: RandomWalkConfig = dataclasses.field(default_factory=RandomWalkConfig)
    langevin: LangevinConfig = dataclasses.field(default_factory=LangevinConfig)
    seed: int | None = None
    rmsd_threshold: float = 10.0
    metric_lmdb_path: Path | str | None = None
    log_stats: bool = False
    log_stats_interval: int = 1000


# Create a simple config-like object from dict
class SimpleConfig:
    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)


class RieProdyModule:
    """Class to handle apo structure perturbation with RieProDy."""

    def __init__(self, config: RieProdyModuleConfig) -> None:
        """Initialize ApoPerturbation.

        Parameters
        ----------
        metric_comp : Configuration for metric computation.
        random_walk : Configuration for random walk.
        langevin : Configuration for Langevin-dynamics-based perturbation.
            Used as a fallback for proteins when RieProDy perturbation is unavailable,
            and as the default perturbation for nucleic acids when perturbation is
            enabled.
        rmsd_threshold : float, optional
            RMSD threshold in Angstroms. If perturbation causes RMSD > threshold,
            original coordinates are used instead. Default: 15.0
        metric_lmdb_path : Path | str | None, optional
            Path to LMDB file containing pre-computed metric information.
        seed : int | None, optional
            Random seed for stochastic operations.
        log_stats : bool, optional
            Whether to log perturbation statistics periodically. Default: False
        log_stats_interval : int, optional
            Log statistics every N chain perturbations. Default: 1000
        """
        from rieprody.proteins.protein_perturbation import ProteinPerturbationModule
        from rieprody.proteins.protein_vocab import (
            ONE_TO_THREE,
            RESTYPE_NAME_TO_ATOM14_NAMES,
        )

        self.config = config
        self.seed: int | None = config.seed
        self.rmsd_threshold: float = config.rmsd_threshold
        self.log_stats: bool = config.log_stats
        self.log_stats_interval: int = config.log_stats_interval

        self.rng = np.random.default_rng(self.seed)

        self.metric_comp: MetricCompConfig = config.metric_comp
        self.random_walk: RandomWalkConfig = config.random_walk
        self.langevin: LangevinConfig = config.langevin

        module_config = SimpleConfig(
            {
                "metric_comp": self.metric_comp,
                "random_walk": self.random_walk,
                "dataset": {
                    # Not used in read_computed_data/solve_riemannian_brownian_motion
                    "data_dir": Path("."),
                    # Not used in read_computed_data/solve_riemannian_brownian_motion
                    "cache_path": Path("."),
                    # Not used in preprocess() here
                    "preprocess_num_workers": 1,
                },
            }
        )
        self._module = ProteinPerturbationModule(module_config, records=None)
        self._atom_vocab: dict[str, list[str]] = {
            aa: RESTYPE_NAME_TO_ATOM14_NAMES[ONE_TO_THREE[aa]]
            for aa in ONE_TO_THREE.keys()
        }

        # Lazy initialization of LMDB
        self._lmdb_env: lmdb.Environment | None = None

        # Debug / diagnostics (prints only on exceptions)
        self._debug_apo_perturbation: bool = (
            os.environ.get("KFOLD_APO_PERTURB_DEBUG", "0") == "1"
        )
        try:
            self._debug_apo_perturbation_max_errors: int = int(
                os.environ.get("KFOLD_APO_PERTURB_DEBUG_MAX", "3")
            )
        except ValueError:
            self._debug_apo_perturbation_max_errors = 3
        self._debug_apo_perturbation_error_count: int = 0

        self._stats_total_perturbations: int = 0
        self._stats_rmsd_filtered: int = 0
        self._stats_shape_mismatch: int = 0
        self._stats_success: int = 0
        self._stats_langevin_used: int = 0

    @staticmethod
    def log(*args, **kwargs):
        """Utility print function for debugging."""
        # FIXME: remove debug prints later
        logging_debug = True

        if logging_debug:
            print("[RieProDyModule]", *args, **kwargs)

    def perturb_protein_apo_structure(
        self,
        apo_coords: np.ndarray,
        mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        key: str | None = None,
    ) -> np.ndarray:
        """Apply perturbation to apo structure coordinates.

        Parameters
        ----------
        apo_coords : np.ndarray
            Apo protein structure coordinates of shape [L, 14, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, 14].
        rng : np.random.Generator
            Random number generator for stochastic operations.
        name : str | None
            Name for lmdb lookup / logging.

        Returns
        -------
        perturbed_apo_coords : np.ndarray
            Perturbed apo structure coordinates of shape [L, 37, 3].
        """
        rng = rng or self.rng

        assert apo_coords.ndim == 3 and apo_coords.shape[1] == 37, (
            f"Expected apo_coords shape [L, 37, 3], got {apo_coords.shape}"
        )
        if mask is None:
            # Create mask based on finite coordinates
            # WARN: assumes that missing atoms are represented by NaN/Inf
            mask: np.ndarray = np.isfinite(apo_coords).all(axis=-1)

        # Try metric-based RieProDy perturbation only when we can look up LMDB.
        if key is None:
            self.log("No LMDB key provided for RieProDy perturbation.")
            return self.langevin_dynamics_perturbation(apo_coords, mask, rng)

        metric_data = self._load_metric_from_lmdb(key)

        if metric_data is None:
            self.log("Failed to load metric data from LMDB for key:", key)
            return self.langevin_dynamics_perturbation(apo_coords, mask, rng)

        # Apply RieProDy perturbation
        perturbed_coords = self.rieprody_perturbation(
            x_init=apo_coords,
            mask=mask,
            rng=rng,
            metric_data=metric_data,
            name=key,
        )
        return perturbed_coords

    def rieprody_perturbation(
        self,
        x_init: np.ndarray,
        mask: np.ndarray,
        metric_data: dict,
        rng: np.random.Generator,
        name: str | None,
    ) -> np.ndarray:
        """Metric-based perturbation using pre-computed LMDB metric + RieProDy RBM.

        Parameters
        ----------
        x_init : np.ndarray
            Coordinates of shape [L, 37, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, 37].
        metric_data : dict
            Pre-computed metric data from LMDB.
        rng : np.random.Generator
            Random number generator for stochastic operations.
        name : str | None
            Name identifier for logging/debugging.
        """
        self._stats_total_perturbations += 1

        if (
            self.log_stats
            and self.log_stats_interval > 0
            and self._stats_total_perturbations % self.log_stats_interval == 0
        ):
            self._log_perturbation_stats()

        try:
            # Convert data to RieProDy format
            rieprody_data = self._prepare_rieprody_data(metric_data)
            self._module.read_computed_data(rieprody_data)

            # NOTE: perturbation via RieProDy's Riemannian Brownian Motion.
            params = {
                "total_time": self._sample_random_walk_total_time(rng),
                "num_steps": self.random_walk.num_steps,
                "metric": self.random_walk.metric,
                "with_christoffel_term": self.random_walk.with_christoffel_term,
                "metric_calculation_period": self.random_walk.metric_calculation_period,
                "use_precomputed_metric": self.random_walk.use_precomputed_metric,
                "kabsch_aligned_traj": self.random_walk.kabsch_aligned_traj,
                "return_as_N3": False,
                "return_q": False,
            }
            trajectory_atom14: np.ndarray = (
                self._module.solve_riemannian_brownian_motion(**params)
                .detach()
                .cpu()
                .numpy()
            )  # [num_steps+1, L, 14, 3]
            if not np.isfinite(trajectory_atom14).all():
                self.log(f"NaN/Inf detected in RieProDy output! (name={name})")
                return x_init  # Fallback to original coords

            if trajectory_atom14.ndim != 4:
                warnings.warn(
                    f"Unexpected trajectory shape: {trajectory_atom14.shape}. "
                    "Using original coordinates.",
                    UserWarning,
                )
                return x_init

            sequence = metric_data["sequence"]
            Nstep, L = trajectory_atom14.shape[:2]
            trajectory = np.full((Nstep, L, 37, 3), np.nan, dtype=np.float32)
            for res_idx, aa in enumerate(sequence):
                atom_orders = self._atom_vocab[aa]
                for i, atom_name in enumerate(atom_orders):
                    if atom_name == "":
                        break  # Skip dummy
                    j = ATOM37_ORDER[atom_name]
                    trajectory[:, res_idx, j, :] = trajectory_atom14[:, res_idx, i, :]

            # Filter by RMSD threshold
            align_mask = mask & np.isfinite(trajectory[0]).all(axis=-1)
            true_mask = np.ones(align_mask.sum(), dtype=bool)
            for step_idx in range(Nstep - 1, -1, -1):
                # Use the last valid perturbation within RMSD threshold
                perturbed_coords = trajectory[step_idx]  # [L, 37, 3]
                rmsd = compute_rmsd(
                    perturbed_coords[align_mask],
                    x_init[align_mask],
                    mask=true_mask,
                    align=True,
                )
                if rmsd < self.rmsd_threshold:
                    self._stats_success += 1
                    out_coords = perturbed_coords
                    break
            else:
                self._stats_rmsd_filtered += 1
                self.log(f"RieProDy perturbation exceeded RMSD threshold (name={name})")
                out_coords = x_init  # All steps exceeded RMSD threshold

            # Restore mask
            return out_coords

        except Exception as e:
            self.log(f"Exception during RieProDy perturbation (name={name}): {e}")
            raise e
            return x_init  # Fallback to original coords

    def sample_prior_from_langevin(
        self,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample prior using Langevin dynamics.

        Parameters
        ----------
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, Natom].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        perturbed_coords : np.ndarray
            Sampled coordinates of shape [L, Natom, 3].
        """
        self._stats_total_perturbations += 1
        self._stats_langevin_used += 1

        # Hyperparameters (Algorithm S3 defaults)
        cfg = self.langevin
        sphere_r = cfg.sphere_r

        # Initialize X0
        # Start from random atomic positions
        # Scale by sphere_r so global compactness term has the right magnitude.
        L, Natoms = mask.shape
        x_init = rng.normal(loc=0.0, scale=sphere_r, size=(L, Natoms, 3)).astype(
            np.float32
        )
        x_init[~mask] = 0.0
        return self.langevin_dynamics_perturbation(x_init, mask, rng)

    def langevin_dynamics_perturbation(
        self,
        x_init: np.ndarray,
        mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Langevin-dynamics-based perturbation.

        Parameters
        ----------
        x_init : np.ndarray
            Starting coordinates of shape [L, Natom, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, Natom].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        perturbed_coords : np.ndarray
            Sampled coordinates of shape [L, Natom, 3].
        """
        rng = rng or self.rng

        self._stats_total_perturbations += 1
        self._stats_langevin_used += 1

        # === Hyperparameters (Algorithm S3 defaults) ===
        cfg = self.langevin
        num_steps = cfg.num_steps
        dt = cfg.dt
        res_r = cfg.res_r
        ent_r = cfg.ent_r
        sphere_r = cfg.sphere_r

        assert num_steps >= 0 and dt > 0.0, (
            f"Invalid Langevin dynamics parameters: num_steps={num_steps}, dt={dt}"
        )

        # === Initialize X0 ===
        if mask is None:
            mask: np.ndarray = np.isfinite(x_init).all(axis=-1)

        L, Natoms = mask.shape
        x = x_init.astype(np.float32, copy=True)  # Safe copy
        x[~mask] = 0.0

        if mask.sum() == 0:
            # No valid atoms, return original coords
            self.log("No valid atoms in mask for Langevin perturbation.")
            return x_init

        w_all = mask.astype(np.float32)[..., None]  # [L,Natom,1]
        w_res = w_all.sum(axis=1).clip(min=1)  # [L,1]
        denom_all = float(w_all.sum())  # Scalar

        # === Langevin dynamics (Algorithm S3) ===
        res_r2 = res_r * res_r
        ent_r2 = ent_r * ent_r
        sphere_r2 = sphere_r * sphere_r
        noise_scale = float(2.0 * np.sqrt(dt))

        for _ in range(num_steps):
            # Pull atoms towards chain center
            mean_chain = (x * w_all).sum(axis=(0, 1)) / denom_all  # [3]
            d_ent = mean_chain.reshape(1, 1, 3) - x  # [L,Natom,3]

            # Pull atoms towards residue center
            mean_res = (x * w_all).sum(axis=1) / w_res  # [L,3]
            d_res = mean_res.reshape(L, 1, 3) - x  # [L,Natom,3]

            # drift
            drift = (d_ent / ent_r2) + (d_res / res_r2) - (x / sphere_r2)

            eps = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float32)
            x = x + dt * drift + noise_scale * eps
            x[~mask] = 0.0
            assert x.dtype == np.float32  # For debugging purposes

        if not np.isfinite(x).all():
            # Safety fallback
            self.log("NaN/Inf detected in Langevin dynamics output!")
            return x_init

        # Centering (masked mean to origin)
        center = (x * w_all).sum(axis=(0, 1)) / denom_all
        x = x - center.reshape(1, 1, 3)

        # Restore mask
        x[~mask] = np.nan

        self._stats_success += 1
        return x

    # === Internal methods === #
    def _sample_random_walk_total_time(self, rng: np.random.Generator) -> float:
        """Sample random-walk total_time for Riemannian Brownian motion.

        Interpretation:
        - `random_walk.total_time` is treated as the maximum value (max_time).
        - Optionally, `random_walk.total_time_min` can be provided as the minimum value.
        - If max_time == min_time, the value is treated as fixed.
        - Otherwise, we sample uniformly from [min_time, max_time).
        """
        max_time = self.random_walk.total_time
        min_time = self.random_walk.total_time_min
        if max_time < min_time:
            warnings.warn(
                (
                    f"random_walk.total_time ({max_time}) < "
                    f"random_walk.total_time_min ({min_time}). Swapping the bounds."
                ),
                UserWarning,
            )
            min_time, max_time = max_time, min_time

        if max_time == min_time:
            return max_time

        return float(rng.uniform(min_time, max_time))

    @property
    def lmdb_env(self) -> lmdb.Environment | None:
        """Lazy initialization of LMDB environment."""
        lmdb_path = self.config.metric_lmdb_path
        if self._lmdb_env is None and lmdb_path is not None:
            assert Path(lmdb_path).exists(), f"LMDB path does not exist: {lmdb_path}"
            self._lmdb_env = lmdb.open(
                str(lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        return self._lmdb_env

    def __del__(self):
        """Cleanup LMDB environment on deletion."""
        if self._lmdb_env is not None:
            self._lmdb_env.close()
            self._lmdb_env = None

    def get_perturbation_stats(self) -> dict[str, float]:
        """Get perturbation statistics as a dictionary with counts and percentages."""
        total = self._stats_total_perturbations
        if total == 0:
            return {
                "total": 0,
                "success": 0,
                "rmsd_filtered": 0,
                "shape_mismatch": 0,
                "success_pct": 0.0,
                "rmsd_filtered_pct": 0.0,
                "shape_mismatch_pct": 0.0,
            }
        return {
            "total": total,
            "success": self._stats_success,
            "rmsd_filtered": self._stats_rmsd_filtered,
            "shape_mismatch": self._stats_shape_mismatch,
            "success_pct": 100.0 * self._stats_success / total,
            "rmsd_filtered_pct": 100.0 * self._stats_rmsd_filtered / total,
            "shape_mismatch_pct": 100.0 * self._stats_shape_mismatch / total,
        }

    def reset_perturbation_stats(self) -> None:
        """Reset all perturbation statistics counters."""
        self._stats_total_perturbations = 0
        self._stats_rmsd_filtered = 0
        self._stats_shape_mismatch = 0
        self._stats_success = 0

    def _log_perturbation_stats(self) -> None:
        stats = self.get_perturbation_stats()
        self.log(
            f"n={stats['total']}: "
            f"success={stats['success_pct']:.1f}%, "
            f"rmsd_filtered={stats['rmsd_filtered_pct']:.1f}%, "
            f"shape_mismatch={stats['shape_mismatch_pct']:.1f}%"
        )

    def _load_metric_from_lmdb(self, key: str) -> dict | None:
        """Load pre-computed metric data from LMDB.

        Parameters
        ----------
        key : str
            Key to look up in LMDB.

        Returns
        -------
        dict | None
            Metric data dictionary or None if not found.
        """
        if self.lmdb_env is None:
            return None

        try:
            with self.lmdb_env.begin(write=False) as txn:
                value_bytes = txn.get(key.encode("utf-8"))
                if value_bytes is None:
                    return None
                data = pickle.loads(value_bytes)
                return data
        except Exception as e:
            # Log error but don't raise - return None to skip perturbation
            raise e
            warnings.warn(
                f"Failed to load metric data for {key}: {e}",
                UserWarning,
            )
            return None

    def _prepare_rieprody_data(self, metric_data: dict) -> dict:
        """Prepare data dictionary for RieProDy from metric data."""

        # Convert numpy arrays to torch tensors
        # Note: RieProDy expects torch tensors
        def to_torch(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            if isinstance(arr, torch.Tensor):
                return arr
            return torch.from_numpy(arr).to(dtype)

        # Create a Data-like object compatible with RieProDy.
        # RieProDy expects `data["receptor"].to(device)` to exist.
        class _RieProDyReceptorData:
            def to(self, device: torch.device | str):
                for k, v in self.__dict__.items():
                    if isinstance(v, torch.Tensor):
                        setattr(self, k, v.to(device))
                return self

        receptor_data = _RieProDyReceptorData()

        # Set sequence from metric_data (pre-computed)
        receptor_data.sequence = metric_data["sequence"]
        num_res = len(receptor_data.sequence)

        # Use metric data fields (already in correct format from LMDB)
        # We need consider_side_chain early to interpret q-dim.
        consider_side_chain = True
        if self.random_walk is not None:
            consider_side_chain = bool(self.random_walk.consider_side_chain_in_metric)

        receptor_data.internal_coords_mask_R8_to_q = to_torch(
            metric_data["internal_coords_mask_R8_to_q"], dtype=torch.bool
        )

        # Match RDBDock: unpack packed metric_inv_cholesky via RieProDy utility.
        # RieProDy expects `metric_inv_cholesky` shaped like [R, 8, R, 8].
        metric_inv_cholesky = to_torch(metric_data["metric_inv_cholesky"]).to(
            torch.float32
        )
        if metric_inv_cholesky.ndim in (1, 2):
            metric_inv_cholesky = self._module.unpack_metric_inv_cholesky(
                metric_inv_cholesky,
                num_res,
                side_chain=consider_side_chain,
            )

        receptor_data.metric_inv_cholesky = metric_inv_cholesky
        receptor_data.mp_nerf_image_R = to_torch(metric_data["mp_nerf_image_R"])
        receptor_data.initial_q_R8 = to_torch(metric_data["initial_q_R8"])
        receptor_data.mp_nerf_non_empty_atom_mask_R14 = to_torch(
            metric_data["mp_nerf_non_empty_atom_mask_R14"], dtype=torch.bool
        )
        receptor_data.rot_atom_mask = to_torch(
            metric_data["rot_atom_mask"], dtype=torch.bool
        )

        # Additional required fields from metric_data
        # cloud_mask / reverse_cloud_mask
        if "cloud_mask" in metric_data and metric_data["cloud_mask"] is not None:
            cloud_mask = to_torch(metric_data["cloud_mask"], dtype=torch.bool)
            if cloud_mask.ndim == 1 and cloud_mask.numel() == num_res * 14:
                cloud_mask = cloud_mask.view(num_res, 14)
            receptor_data.cloud_mask = cloud_mask

            if (
                "reverse_cloud_mask" in metric_data
                and metric_data["reverse_cloud_mask"] is not None
            ):
                receptor_data.reverse_cloud_mask = to_torch(
                    metric_data["reverse_cloud_mask"], dtype=torch.long
                )
            else:
                counts = cloud_mask.to(torch.long).sum(dim=-1)
                receptor_data.reverse_cloud_mask = torch.repeat_interleave(
                    torch.arange(len(counts), device=counts.device), counts
                )

            # R3 masks are required when consider_side_chain_in_metric=False
            if not consider_side_chain:
                if (
                    "cloud_mask_R3" in metric_data
                    and metric_data["cloud_mask_R3"] is not None
                ):
                    cloud_mask_r3 = to_torch(
                        metric_data["cloud_mask_R3"], dtype=torch.bool
                    )
                    if cloud_mask_r3.ndim == 1 and cloud_mask_r3.numel() == num_res * 3:
                        cloud_mask_r3 = cloud_mask_r3.view(num_res, 3)
                else:
                    cloud_mask_r3 = cloud_mask[:, :3].clone()
                receptor_data.cloud_mask_R3 = cloud_mask_r3

                if (
                    "reverse_cloud_mask_R3" in metric_data
                    and metric_data["reverse_cloud_mask_R3"] is not None
                ):
                    receptor_data.reverse_cloud_mask_R3 = to_torch(
                        metric_data["reverse_cloud_mask_R3"], dtype=torch.long
                    )
                else:
                    counts3 = cloud_mask_r3.to(torch.long).sum(dim=-1)
                    receptor_data.reverse_cloud_mask_R3 = torch.repeat_interleave(
                        torch.arange(len(counts3), device=counts3.device), counts3
                    )

        if "point_ref_mask" in metric_data:
            receptor_data.point_ref_mask = to_torch(
                metric_data["point_ref_mask"], dtype=torch.long
            )
        if "bond_mask" in metric_data:
            receptor_data.bond_mask = to_torch(metric_data["bond_mask"])
        if "apo_angles_mask" in metric_data:
            receptor_data.apo_angles_mask = to_torch(metric_data["apo_angles_mask"])
        elif "angles_mask" in metric_data:
            receptor_data.apo_angles_mask = to_torch(metric_data["angles_mask"])
        if "chi2rot_atom_mask" in metric_data:
            receptor_data.chi2rot_atom_mask = to_torch(metric_data["chi2rot_atom_mask"])
        if "chi2ref_atom_mask" in metric_data:
            receptor_data.chi2ref_atom_mask = to_torch(metric_data["chi2ref_atom_mask"])

        return {"receptor": receptor_data}
