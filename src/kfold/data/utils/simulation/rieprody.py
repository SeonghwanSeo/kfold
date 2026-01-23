"""Module for apo structure perturbation using RieProDy.
Fallback to Langevin dynamics if RieProDy perturbation is unavailable.
"""

import dataclasses
import os
from io import BytesIO
from pathlib import Path
from typing import Self

import lmdb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import kfold.constants as C
from kfold.utils.geometry.rigid_align import compute_rmsd, rigid_align


# fmt: off
class RieProdyPerturbationError(Exception):
    """Custom exception for RieProDy perturbation errors."""
class ShapeMismatchError(RieProdyPerturbationError): ...
class SimulationError(RieProdyPerturbationError): ...
class NanInfInOutputError(RieProdyPerturbationError): ...
# fmt: on


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
class RieProdyConfig:
    metric_comp: MetricCompConfig = dataclasses.field(default_factory=MetricCompConfig)
    random_walk: RandomWalkConfig = dataclasses.field(default_factory=RandomWalkConfig)
    rmsd_threshold: float = 10.0
    metric_lmdb_path: Path | str | None = None
    log_stats: bool = False
    log_stats_interval: int = 1000
    disable_log: bool = False

    @classmethod
    def from_config(cls, config: DictConfig | Self) -> Self:
        """Create RieProdyConfig using omegaconf merge"""
        base_cfg = OmegaConf.structured(cls)
        merged_cfg = OmegaConf.merge(base_cfg, config)
        return OmegaConf.to_object(merged_cfg)


# Create a simple config-like object from dict
class SimpleConfig:
    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)


class RieProdyPerturbation:
    """Class to handle apo structure perturbation with RieProDy."""

    def __init__(self, config: RieProdyConfig) -> None:
        """Initialize RieProDy perturbation module.

        Parameters
        ----------
        metric_comp : Configuration for metric computation.
        random_walk : Configuration for random walk.
        rmsd_threshold : float, optional
            RMSD threshold in Angstroms. If perturbation causes RMSD > threshold,
            original coordinates are used instead. Default: 15.0
        metric_lmdb_path : Path | str | None, optional
            Path to LMDB file containing pre-computed metric information.
        log_stats : bool, optional
            Whether to log perturbation statistics periodically. Default: False
        log_stats_interval : int, optional
            Log statistics every N chain perturbations. Default: 1000
        """
        # Import RieProDy modules here to avoid hard dependency at the top-level.
        from rieprody.proteins.protein_perturbation import ProteinPerturbationModule
        from rieprody.proteins.protein_vocab import (
            ONE_TO_THREE,
            RESTYPE_NAME_TO_ATOM14_NAMES,
        )

        # Initialize configuration
        config = RieProdyConfig.from_config(config)
        self.config: RieProdyConfig = config
        self.rmsd_threshold: float = config.rmsd_threshold
        self.log_stats: bool = config.log_stats
        self.log_stats_interval: int = config.log_stats_interval
        self._disable_log: bool = config.disable_log

        if os.environ.get("RIEPRODY_DISABLE_LOG", "0") == "1":
            self._disable_log = True

        # Initialize RieProDy ProteinPerturbationModule
        self.metric_comp: MetricCompConfig = config.metric_comp
        self.random_walk: RandomWalkConfig = config.random_walk

        # Check random_walk parameters
        if self.random_walk.num_steps <= 0:
            raise ValueError(
                f"random_walk.num_steps must be positive, got "
                f"{self.random_walk.num_steps}."
            )
        if self.random_walk.total_time_min > self.random_walk.total_time:
            raise ValueError(
                f"random_walk.total_time_min ({self.random_walk.total_time_min}) "
                f"> random_walk.total_time ({self.random_walk.total_time})."
            )

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

        # Atom vocab from atom37 conversion
        ATOM37_ORDER: dict[str, int] = C.atom.protein_atom37_order
        self._atom_vocab: dict[str, list[str]] = {
            aa: RESTYPE_NAME_TO_ATOM14_NAMES[ONE_TO_THREE[aa]]
            for aa in ONE_TO_THREE.keys()
        }
        self._atom37_indices: dict[str, list[int]] = {}
        for aa in ONE_TO_THREE.keys():
            atom_names: list[str] = self._atom_vocab[aa]
            self._atom37_indices[aa] = [
                ATOM37_ORDER[name] for name in atom_names if name != ""
            ]

        # Lazy initialization of LMDB
        assert config.metric_lmdb_path is not None, (
            "metric_lmdb_path must be provided for RieProDyModule."
        )
        self.lmdb_path: str = str(config.metric_lmdb_path)
        assert Path(self.lmdb_path).exists(), (
            f"LMDB path does not exist: {self.lmdb_path}"
        )
        self._lmdb_env: lmdb.Environment | None = None

        # === Perturbation statistics === #
        self._stats_total_perturbations: int = 0
        self._stats_rmsd_filtered: int = 0
        self._stats_nan_inf_in_output: int = 0
        self._stats_exception: int = 0
        self._stats_shape_mismatch: int = 0
        self._stats_success: int = 0

    # === Main perturbation methods === #
    def run(
        self,
        coords: np.ndarray,
        mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        key: str | None = None,
    ) -> np.ndarray | None:
        """Apply perturbation to apo structure coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Apo protein structure coordinates of shape [L, 37, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, 37].
        rng : np.random.Generator
            Random number generator for stochastic operations.
        key : str | None
            Key for lmdb lookup / logging for rieprody perturbation.

        Returns
        -------
        perturbed_coords : np.ndarray | None
            Perturbed apo structure coordinates of shape [L, 37, 3].
            Returns None if perturbation failed.
        """
        # Validate input shapes
        if coords.ndim != 3 or coords.shape[1:] != (37, 3):
            raise ValueError(
                f"Input coords must have shape [L, 37, 3], got {coords.shape}"
            )
        if mask is None:
            # Create mask based on finite coordinates
            # WARN: assumes that missing atoms are represented by NaN/Inf
            mask: np.ndarray = np.isfinite(coords).all(axis=-1)
        else:
            if mask.shape != coords.shape[:2]:
                raise ValueError(f"Input mask must have shape [L, 37], got {mask.shape}")

        # Try metric-based RieProDy perturbation only when we can look up LMDB.
        if key is None:
            raise NotImplementedError("On-the-fly perturbation is not implemented yet.")
        else:
            # Sample RieProDy-perturbed coordinates
            rng = rng or np.random.default_rng()
            metric_data = self._load_metric_from_lmdb(key)
            if metric_data is None:
                self.log(f"LMDB load failure (key={key})")
                return None
            # Convert data to RieProDy format
            rieprody_data = self._prepare_rieprody_data(metric_data)

        try:
            perturbed_coords: np.ndarray = self.run_simulation(rieprody_data, rng)
        except ShapeMismatchError as e:
            self.log(f"Output shape mismatch (key={key}), {e}")
            self._stats_shape_mismatch += 1
            return None
        except NanInfInOutputError as e:
            self.log(f"NaN/Inf detected! (key={key}), {e}")
            self._stats_nan_inf_in_output += 1
            return None
        except SimulationError as e:
            self.log(f"Simulation failure (key={key}), {e}")
            self._stats_exception += 1
            return None
        except Exception as e:
            raise e

        # Align perturbed coords to original coords
        perturbed_mask: np.ndarray = np.isfinite(perturbed_coords).all(axis=-1)
        align_mask: np.ndarray = mask & perturbed_mask
        aligned_coords = rigid_align(
            perturbed_coords.reshape(-1, 3),
            coords.reshape(-1, 3),
            align_mask.reshape(-1),
        ).reshape(perturbed_coords.shape)
        aligned_coords[~perturbed_mask] = np.nan  # Mask out invalid atoms

        # Compute RMSD between aligned perturbed coords and original coords
        rmsd = compute_rmsd(
            aligned_coords.reshape(-1, 3),
            coords.reshape(-1, 3),
            mask=align_mask.reshape(-1),
            align=False,
        ).item()

        if rmsd > self.rmsd_threshold:
            self._stats_rmsd_filtered += 1
            self.log(
                f"RieProDy perturbation exceeded RMSD threshold "
                f"(key={key}, rmsd={rmsd:.3f}A > {self.rmsd_threshold}A)"
            )
            return None

        self._stats_success += 1
        return aligned_coords

    def sample(
        self,
        coords: np.ndarray,
        mask: np.ndarray | None = None,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
        key: str | None = None,
    ) -> list[np.ndarray]:
        """Apply perturbation to apo structure coordinates and
        sample multiple perturbed structures.

        Parameters
        ----------
        coords : np.ndarray
            Apo protein structure coordinates of shape [L, 37, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, 37].
        rng : np.random.Generator
            Random number generator for stochastic operations.
        key : str | None
            Key for lmdb lookup / logging for rieprody perturbation.

        Returns
        -------
        perturbed_coords_list : list[np.ndarray]
            List of perturbed apo structure coordinates of shape [L, 37, 3].
            Returns empty list if no valid perturbations were sampled.
        """
        # Validate input shapes
        if coords.ndim != 3 or coords.shape[1:] != (37, 3):
            raise ValueError(
                f"Input coords must have shape [L, 37, 3], got {coords.shape}"
            )
        if mask is None:
            # Create mask based on finite coordinates
            # WARN: assumes that missing atoms are represented by NaN/Inf
            mask: np.ndarray = np.isfinite(coords).all(axis=-1)
        else:
            if mask.shape != coords.shape[:2]:
                raise ValueError(f"Input mask must have shape [L, 37], got {mask.shape}")

        # Try metric-based RieProDy perturbation only when we can look up LMDB.
        if key is None:
            raise NotImplementedError("On-the-fly perturbation is not implemented yet.")
        else:
            # Sample RieProDy-perturbed coordinates
            metric_data = self._load_metric_from_lmdb(key)
            if metric_data is None:
                self.log(f"LMDB load failure (key={key})")
                return []
            # Convert data to RieProDy format
            rieprody_data = self._prepare_rieprody_data(metric_data)

        perturbed_coords_list: list[np.ndarray] = []
        rng = rng or np.random.default_rng()
        for _ in range(num_samples):
            try:
                out = self.run_simulation(rieprody_data, rng)
            except ShapeMismatchError as e:
                self.log(f"Output shape mismatch (key={key}), {e}")
                self._stats_shape_mismatch += 1
                continue
            except NanInfInOutputError as e:
                self.log(f"NaN/Inf detected! (key={key}), {e}")
                self._stats_nan_inf_in_output += 1
                continue
            except SimulationError as e:
                self.log(f"Simulation failure (key={key}), {e}")
                self._stats_exception += 1
                continue
            except Exception as e:
                raise e

            perturbed_coords: np.ndarray = out

            # Align perturbed coords to original coords
            perturbed_mask: np.ndarray = np.isfinite(perturbed_coords).all(axis=-1)
            align_mask: np.ndarray = mask & perturbed_mask
            aligned_coords = rigid_align(
                perturbed_coords.reshape(-1, 3),
                coords.reshape(-1, 3),
                align_mask.reshape(-1),
            ).reshape(perturbed_coords.shape)
            aligned_coords[~perturbed_mask] = np.nan  # Mask out invalid atoms

            # Compute RMSD between aligned perturbed coords and original coords
            rmsd = compute_rmsd(
                aligned_coords.reshape(-1, 3),
                coords.reshape(-1, 3),
                mask=align_mask.reshape(-1),
                align=False,
            ).item()

            if rmsd > self.rmsd_threshold:
                # Log RMSD exceed and continue
                self.log(
                    f"RieProDy perturbation exceeded RMSD threshold "
                    f"(key={key}, rmsd={rmsd:.3f}A > {self.rmsd_threshold}A)"
                )
                self._stats_rmsd_filtered += 1
                continue

            self._stats_success += 1
            perturbed_coords_list.append(aligned_coords)

        return perturbed_coords_list

    # === Main simulation method === #
    def run_simulation(
        self,
        rieprody_data: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Metric-based perturbation.

        Parameters
        ----------
        rieprody_data : dict
            RieProDy-formatted data dictionary.
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        perturbed_coords : np.ndarray
            Perturbed coordinates of shape [L, 37, 3].
        """
        self._stats_total_perturbations += 1
        if (
            self.log_stats
            and self.log_stats_interval > 0
            and self._stats_total_perturbations % self.log_stats_interval == 0
        ):
            # Log perturbation statistics periodically
            self._log_perturbation_stats()

        # Sample random walk total time
        total_time = self._sample_random_walk_total_time(rng)

        try:
            # NOTE: perturbation via RieProDy's Riemannian Brownian Motion.
            self._module.read_computed_data(rieprody_data)
            trajectory_atom14: np.ndarray = self._module.solve_riemannian_brownian_motion(
                total_time=total_time,
                num_steps=self.random_walk.num_steps,
                metric=self.random_walk.metric,
                with_christoffel_term=self.random_walk.with_christoffel_term,
                metric_calculation_period=self.random_walk.metric_calculation_period,
                use_precomputed_metric=self.random_walk.use_precomputed_metric,
                kabsch_aligned_traj=self.random_walk.kabsch_aligned_traj,
                return_as_N3=False,
                return_q=False,
            )  # [num_steps+1, L, 14, 3]
            if trajectory_atom14.ndim != 4 or trajectory_atom14.shape[2:] != (14, 3):
                raise ShapeMismatchError(
                    f"Rieprody output shape mismatch: "
                    f"expected [num-step, length, 14, 3], "
                    f"got {trajectory_atom14.shape}",
                )

            # TODO: Handle multiple steps later
            perturbed_atom14: np.ndarray = (
                trajectory_atom14[-1].detach().cpu().numpy()
            )  # [L, 14, 3]
            if not np.isfinite(perturbed_atom14).all():
                raise NanInfInOutputError("NaN/Inf detected in RieProDy output.")

            # Convert atom14 to atom37
            # WARN: RieProDy atom14 ordering is different from standard atom14 ordering.
            sequence: str = rieprody_data["receptor"].sequence
            L = perturbed_atom14.shape[0]
            perturbed: np.ndarray = np.full((L, 37, 3), np.nan, dtype=np.float32)
            for res_i, aa in enumerate(sequence):
                atom37_idcs = self._atom37_indices[aa]
                natoms = len(atom37_idcs)
                perturbed[res_i, atom37_idcs, :] = perturbed_atom14[res_i, :natoms, :]
            return perturbed

        except Exception as e:
            raise SimulationError(f"Exception during RieProDy perturbation: {e}") from e

    # === Internal methods === #
    def log(self, *args, **kwargs):
        """Utility print function for debugging."""
        # FIXME: remove debug prints later
        if not self._disable_log:
            print("[RieProDy]", *args, **kwargs)

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
        if max_time == min_time:
            return max_time
        return float(rng.uniform(min_time, max_time))

    @property
    def lmdb_env(self) -> lmdb.Environment:
        """Lazy initialization of LMDB environment."""
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                str(self.lmdb_path),
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
        self._stats_nan_inf_in_output = 0
        self._stats_exception = 0
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
        with self.lmdb_env.begin(write=False) as txn:
            try:
                value_bytes = txn.get(key.encode("utf-8"))
                if value_bytes is None:
                    return None
                # npz deserialization
                with BytesIO(value_bytes) as byte_io:
                    with np.load(byte_io, allow_pickle=True) as npz_file:
                        data = {k: npz_file[k] for k in npz_file.files}
                return data
            except Exception:
                # Failed to load from LMDB
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
        receptor_data.sequence = metric_data["sequence"].item()
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
            receptor_data.chi2rot_atom_mask = to_torch(
                metric_data["chi2rot_atom_mask"]
            ).float()
        if "chi2ref_atom_mask" in metric_data:
            receptor_data.chi2ref_atom_mask = to_torch(
                metric_data["chi2ref_atom_mask"]
            ).float()

        return {"receptor": receptor_data}
