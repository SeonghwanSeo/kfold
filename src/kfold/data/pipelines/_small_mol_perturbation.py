"""Small molecule apo perturbation (Gaussian / Langevin)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from kfold.data.types.structure import Chain


@dataclasses.dataclass(kw_only=True)
class SmallMolPerturbationConfig:
    """Configuration for small molecule apo perturbation."""

    mode: str = "none"  # none, gaussian, langevin, gaussian_langevin
    gaussian_sigma: float = 0.3
    langevin_num_steps: int = 200
    langevin_dt: float = 0.005
    langevin_k_bond: float = 50.0
    langevin_k_angle: float = 10.0
    langevin_temperature: float = 1.0
    recenter: bool = True


class SmallMolPerturbation:
    """Apply small molecule perturbation on apo coordinates."""

    def __init__(self, config: SmallMolPerturbationConfig) -> None:
        self.config = config
        valid_modes = {"none", "gaussian", "langevin", "gaussian_langevin"}
        if self.config.mode not in valid_modes:
            raise ValueError(
                "Invalid small molecule perturbation mode: "
                f"{self.config.mode}. Expected one of {sorted(valid_modes)}."
            )

    def __call__(
        self,
        coords: np.ndarray,
        chain: Chain,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return self.run(coords, chain, rng=rng)

    def run(
        self,
        coords: np.ndarray,
        chain: Chain,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        cfg = self.config
        if cfg.mode == "none":
            return coords
        rng = rng or np.random.default_rng()

        base_coords = coords.astype(np.float32, copy=True)
        mask = np.isfinite(base_coords).all(axis=-1)
        if not mask.any():
            return coords
        ref_center = base_coords[mask].mean(axis=0)

        bonds = self._build_chain_bond_indices(chain)
        if bonds.size > 0:
            valid = mask[bonds[:, 0]] & mask[bonds[:, 1]]
            bonds = bonds[valid]
        angles = self._build_angle_indices(bonds, base_coords.shape[0])
        if angles.size > 0:
            valid = mask[angles[:, 0]] & mask[angles[:, 1]] & mask[angles[:, 2]]
            angles = angles[valid]
        d0 = self._compute_bond_lengths(base_coords, bonds)
        theta0 = self._compute_angles(base_coords, angles)

        if cfg.mode == "gaussian":
            return self._apply_gaussian(base_coords, mask, ref_center, rng)
        if cfg.mode == "langevin":
            return self._run_langevin(
                base_coords, bonds, angles, d0, theta0, mask, ref_center, rng
            )
        if cfg.mode == "gaussian_langevin":
            init = self._sample_gaussian_init(base_coords, mask, rng, ref_center)
            return self._run_langevin(
                init, bonds, angles, d0, theta0, mask, ref_center, rng
            )
        return base_coords

    def _build_chain_bond_indices(self, chain: Chain) -> np.ndarray:
        atom_index_map: dict[tuple[int, str], int] = {}
        for res_i in range(chain.num_residues):
            residue_index = res_i + 1  # 1-based
            for atom_i in chain.residue.iter_residue_atoms(residue_index):
                atom_index_map[(residue_index, chain.atom.name[atom_i])] = atom_i

        bonds: list[tuple[int, int]] = []
        for bond_i in range(chain.num_bonds):
            ridx1, ridx2 = chain.bond.residue_index[bond_i]
            atom1, atom2 = chain.bond.atom_name[bond_i].tolist()
            idx1 = atom_index_map.get((int(ridx1), atom1))
            idx2 = atom_index_map.get((int(ridx2), atom2))
            if idx1 is None or idx2 is None:
                continue
            bonds.append((idx1, idx2))
        if len(bonds) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array(bonds, dtype=np.int64)

    def _build_angle_indices(self, bonds: np.ndarray, num_atoms: int) -> np.ndarray:
        if bonds.size == 0:
            return np.zeros((0, 3), dtype=np.int64)
        neighbors: list[list[int]] = [[] for _ in range(num_atoms)]
        for i, j in bonds:
            neighbors[int(i)].append(int(j))
            neighbors[int(j)].append(int(i))
        angles: list[tuple[int, int, int]] = []
        for j in range(num_atoms):
            neigh = neighbors[j]
            if len(neigh) < 2:
                continue
            for a in range(len(neigh) - 1):
                for b in range(a + 1, len(neigh)):
                    i = neigh[a]
                    k = neigh[b]
                    angles.append((i, j, k))
        if len(angles) == 0:
            return np.zeros((0, 3), dtype=np.int64)
        return np.array(angles, dtype=np.int64)

    def _compute_bond_lengths(self, coords: np.ndarray, bonds: np.ndarray) -> np.ndarray:
        if bonds.size == 0:
            return np.zeros((0,), dtype=np.float32)
        vec = coords[bonds[:, 0]] - coords[bonds[:, 1]]
        return np.linalg.norm(vec, axis=1)

    def _compute_angles(self, coords: np.ndarray, angles: np.ndarray) -> np.ndarray:
        if angles.size == 0:
            return np.zeros((0,), dtype=np.float32)
        v1 = coords[angles[:, 0]] - coords[angles[:, 1]]
        v2 = coords[angles[:, 2]] - coords[angles[:, 1]]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        denom = np.clip(n1 * n2, 1e-8, None)
        cos_theta = (v1 * v2).sum(axis=1) / denom
        cos_theta = np.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        return np.arccos(cos_theta)

    def _bond_gradients(
        self, coords: np.ndarray, bonds: np.ndarray, d0: np.ndarray
    ) -> np.ndarray:
        if bonds.size == 0:
            return np.zeros_like(coords)
        grad = np.zeros_like(coords)
        eps = 1e-8
        for idx, (i, j) in enumerate(bonds):
            rij = coords[i] - coords[j]
            dist = np.linalg.norm(rij)
            if dist < eps:
                continue
            diff = dist - float(d0[idx])
            coeff = self.config.langevin_k_bond * diff / dist
            g = coeff * rij
            grad[i] += g
            grad[j] -= g
        return grad

    def _angle_gradients(
        self, coords: np.ndarray, angles: np.ndarray, theta0: np.ndarray
    ) -> np.ndarray:
        if angles.size == 0:
            return np.zeros_like(coords)
        grad = np.zeros_like(coords)
        eps = 1e-8
        for idx, (i, j, k) in enumerate(angles):
            v1 = coords[i] - coords[j]
            v2 = coords[k] - coords[j]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < eps or n2 < eps:
                continue
            u = v1 / n1
            v = v2 / n2
            cos_theta = float(np.dot(u, v))
            cos_theta = float(np.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
            theta = math.acos(cos_theta)
            sin_theta = math.sqrt(max(1.0 - cos_theta * cos_theta, eps))
            diff = theta - float(theta0[idx])
            if abs(diff) < 1e-12:
                continue
            dtheta_dv1 = (cos_theta * u - v) / (n1 * sin_theta)
            dtheta_dv2 = (cos_theta * v - u) / (n2 * sin_theta)
            g1 = self.config.langevin_k_angle * diff * dtheta_dv1
            g3 = self.config.langevin_k_angle * diff * dtheta_dv2
            grad[i] += g1
            grad[k] += g3
            grad[j] -= g1 + g3
        return grad

    def _apply_gaussian(
        self,
        coords: np.ndarray,
        mask: np.ndarray,
        ref_center: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = coords.astype(np.float32, copy=True)
        noise = rng.normal(scale=self.config.gaussian_sigma, size=out.shape).astype(
            np.float32
        )
        out[mask] = out[mask] + noise[mask]
        if self.config.recenter and mask.any():
            center = out[mask].mean(axis=0)
            out[mask] = out[mask] - center + ref_center
        return out

    def _sample_gaussian_init(
        self,
        coords: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        ref_center: np.ndarray,
    ) -> np.ndarray:
        out = coords.astype(np.float32, copy=True)
        noise = rng.normal(scale=self.config.gaussian_sigma, size=out.shape).astype(
            np.float32
        )
        out[mask] = noise[mask]
        if self.config.recenter and mask.any():
            center = out[mask].mean(axis=0)
            out[mask] = out[mask] - center + ref_center
        return out

    def _run_langevin(
        self,
        coords: np.ndarray,
        bonds: np.ndarray,
        angles: np.ndarray,
        d0: np.ndarray,
        theta0: np.ndarray,
        mask: np.ndarray,
        ref_center: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = coords.astype(np.float32, copy=True)
        noise_scale = math.sqrt(
            2.0 * self.config.langevin_temperature * self.config.langevin_dt
        )
        for _ in range(self.config.langevin_num_steps):
            grad = self._bond_gradients(out, bonds, d0)
            grad += self._angle_gradients(out, angles, theta0)
            noise = rng.normal(scale=noise_scale, size=out.shape).astype(np.float32)
            out = out - self.config.langevin_dt * grad + noise
            out[~mask] = coords[~mask]
        if self.config.recenter and mask.any():
            center = out[mask].mean(axis=0)
            out[mask] = out[mask] - center + ref_center
        return out
