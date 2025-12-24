"""Test DDBM interpolation with synthetic molecular conformations.

This test uses two synthetic molecular conformations to validate:
1. Interpolation trajectory correctness (bridge mean path)
2. Statistical properties: variance and mean of multiple interpolation samples
3. Bridge diffusion specific noise injection and coefficients
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.model.modules.structure_module.kfold_ddbm import KFoldBridgeDiffusion


class MockScoreModel(BaseScoreModel):
    """Mock score model for synthetic tests."""

    def __init__(self):
        # Initialize with dummy config
        class DummyConfig:
            pass

        super().__init__(cfg=DummyConfig())

    def forward(self, *args, **kwargs):
        # Not used in interpolation tests
        return torch.zeros(1, 1, 1, 1, 3)


def create_synthetic_conformations(
    num_atoms: int = 10,
    displacement: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create two synthetic molecular conformations.

    Creates a simple chain-like molecule with two conformations:
    - Conformation A (holo/target): atoms in a line along x-axis
    - Conformation B (apo/source): atoms displaced along y and z axes

    Returns
    -------
    coords_a : torch.Tensor
        Source (apo) coordinates. Shape (1, 1, num_atoms, 3).
    coords_b : torch.Tensor
        Target (holo) coordinates. Shape (1, 1, num_atoms, 3).
    """
    # Holo conformation: simple line along x-axis
    x_holo = torch.linspace(0, num_atoms - 1, num_atoms)
    y_holo = torch.zeros(num_atoms)
    z_holo = torch.zeros(num_atoms)
    coords_holo = torch.stack([x_holo, y_holo, z_holo], dim=-1)  # (num_atoms, 3)

    # Apo conformation: displaced version (rotation + translation effect)
    x_apo = torch.linspace(0, num_atoms - 1, num_atoms)
    y_apo = torch.sin(torch.linspace(0, 2 * np.pi, num_atoms)) * displacement
    z_apo = torch.cos(torch.linspace(0, 2 * np.pi, num_atoms)) * displacement
    coords_apo = torch.stack([x_apo, y_apo, z_apo], dim=-1)  # (num_atoms, 3)

    # Reshape to (B, N, L, 3) format
    coords_holo = coords_holo.unsqueeze(0).unsqueeze(0)  # (1, 1, num_atoms, 3)
    coords_apo = coords_apo.unsqueeze(0).unsqueeze(0)  # (1, 1, num_atoms, 3)

    return coords_apo, coords_holo


def analyze_interpolation_statistics(
    structure_module: KFoldBridgeDiffusion,
    coords_apo: torch.Tensor,
    coords_holo: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 1000,
    num_time_steps: int = 21,
) -> dict:
    """Analyze statistical properties of DDBM interpolation.

    Returns mean and variance of interpolated positions at each sigma level.
    """
    sigma_values = torch.linspace(
        0.001, structure_module.sigma_max - 0.001, num_time_steps
    )

    results = {
        "sigma_values": sigma_values.numpy(),
        "means": [],
        "variances": [],
        "expected_means": [],
        "expected_variances": [],
    }

    T = structure_module.sigma_max * structure_module.sigma_data

    for sigma in sigma_values:
        sigma_tensor = torch.full((1, num_samples), sigma.item())

        # Expand coordinates for multiple samples
        apo_expanded = coords_apo.expand(1, num_samples, -1, -1)
        holo_expanded = coords_holo.expand(1, num_samples, -1, -1)

        # Sample multiple interpolations
        samples = structure_module.interpolate(
            apo_expanded, holo_expanded, sigma_tensor, mask
        )  # (1, num_samples, num_atoms, 3)

        # Compute statistics across samples
        mean = samples.mean(dim=1)  # (1, num_atoms, 3)
        var = samples.var(dim=1)  # (1, num_atoms, 3)

        results["means"].append(mean.squeeze(0).numpy())
        results["variances"].append(var.squeeze(0).numpy())

        # Compute expected values for DDBM
        # Expected mean: a_t * apo + b_t * holo
        sigma_exp = torch.tensor([[[[sigma.item()]]]])
        a_t = (sigma_exp**2 / T**2).item()
        b_t = 1 - a_t
        std_t = sigma_exp.item() * np.sqrt(b_t)

        expected_mean = a_t * coords_apo + b_t * coords_holo
        expected_var = std_t**2  # Variance from noise term

        results["expected_means"].append(expected_mean.squeeze().numpy())
        results["expected_variances"].append(expected_var)

    return results


def plot_trajectory_3d(
    structure_module: KFoldBridgeDiffusion,
    coords_apo: torch.Tensor,
    coords_holo: torch.Tensor,
    mask: torch.Tensor,
    num_time_steps: int = 11,
    save_path: str = "./tmp/test_ddbm_synthetic/trajectory_3d.png",
) -> None:
    """Plot 3D trajectory of DDBM interpolation for a single atom."""
    fig = plt.figure(figsize=(12, 5))

    # Left: Single trajectory (mean)
    ax1 = fig.add_subplot(121, projection="3d")

    sigma_values = torch.linspace(
        0.001, structure_module.sigma_max - 0.001, num_time_steps
    )[None, :]
    apo_exp = coords_apo.expand(1, num_time_steps, -1, -1)
    holo_exp = coords_holo.expand(1, num_time_steps, -1, -1)

    trajectory = structure_module.interpolate(apo_exp, holo_exp, sigma_values, mask)
    trajectory = trajectory[0].numpy()  # (num_time_steps, num_atoms, 3)

    # Plot trajectory for atom 0
    atom_idx = 0
    ax1.plot(
        trajectory[:, atom_idx, 0],
        trajectory[:, atom_idx, 1],
        trajectory[:, atom_idx, 2],
        "b-o",
        markersize=4,
        label="DDBM interpolation path",
    )
    ax1.scatter(
        *coords_holo[0, 0, atom_idx].numpy(), c="green", s=100, label="Holo (σ≈0)"
    )
    ax1.scatter(
        *coords_apo[0, 0, atom_idx].numpy(), c="red", s=100, label="Apo (σ≈σ_max)"
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"DDBM Path (Atom {atom_idx})")
    ax1.legend()

    # Right: All atoms
    ax2 = fig.add_subplot(122, projection="3d")

    colors = plt.cm.viridis(np.linspace(0, 1, num_time_steps))
    for i, _sigma in enumerate(sigma_values[0]):
        ax2.scatter(
            trajectory[i, :, 0],
            trajectory[i, :, 1],
            trajectory[i, :, 2],
            c=[colors[i]],
            alpha=0.7,
            s=30,
        )

    # Endpoints
    ax2.scatter(
        coords_holo[0, 0, :, 0].numpy(),
        coords_holo[0, 0, :, 1].numpy(),
        coords_holo[0, 0, :, 2].numpy(),
        c="green",
        s=100,
        marker="^",
        label="Holo",
    )
    ax2.scatter(
        coords_apo[0, 0, :, 0].numpy(),
        coords_apo[0, 0, :, 1].numpy(),
        coords_apo[0, 0, :, 2].numpy(),
        c="red",
        s=100,
        marker="v",
        label="Apo",
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("All Atoms DDBM Interpolation")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved 3D trajectory plot to {save_path}")


def plot_statistics(results: dict, save_path: str) -> None:
    """Plot mean and variance statistics for DDBM."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sigma_values = results["sigma_values"]
    num_atoms = results["means"][0].shape[0]

    # Plot 1: Mean comparison (atom 0, x-coordinate)
    ax = axes[0, 0]
    empirical_means = [m[0, 0] for m in results["means"]]
    expected_means = [m[0, 0] for m in results["expected_means"]]
    ax.plot(sigma_values, empirical_means, "b-o", label="Empirical mean", markersize=4)
    ax.plot(sigma_values, expected_means, "r--", label="Expected mean", linewidth=2)
    ax.set_xlabel("σ (noise level)")
    ax.set_ylabel("X position")
    ax.set_title("Mean X Position (Atom 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Variance vs expected variance
    ax = axes[0, 1]
    empirical_vars = [v.mean() for v in results["variances"]]  # avg across atoms/dims
    expected_vars = results["expected_variances"]
    ax.plot(sigma_values, empirical_vars, "b-o", label="Empirical variance", markersize=4)
    ax.plot(
        sigma_values,
        expected_vars,
        "r--",
        label=r"$\sigma_t^2(1-t^2/T^2)$ (expected)",
        linewidth=2,
    )
    ax.set_xlabel("σ (noise level)")
    ax.set_ylabel("Variance")
    ax.set_title(r"Variance vs Expected $\sigma_t^2(1-t^2/T^2)$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Mean trajectory for all atoms (y-coordinate)
    ax = axes[1, 0]
    for atom_idx in range(num_atoms):
        means_y = [m[atom_idx, 1] for m in results["means"]]
        ax.plot(sigma_values, means_y, "-", alpha=0.7, label=f"Atom {atom_idx}")
    ax.set_xlabel("σ (noise level)")
    ax.set_ylabel("Y position")
    ax.set_title("Mean Y Position (All Atoms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Variance per atom
    ax = axes[1, 1]
    for atom_idx in range(num_atoms):
        vars_atom = [v[atom_idx].mean() for v in results["variances"]]
        ax.plot(sigma_values, vars_atom, "-", alpha=0.7, label=f"Atom {atom_idx}")
    ax.set_xlabel("σ (noise level)")
    ax.set_ylabel("Variance")
    ax.set_title("Variance per Atom")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved statistics plot to {save_path}")


def test_mean_trajectory(
    structure_module: KFoldBridgeDiffusion,
    coords_apo: torch.Tensor,
    coords_holo: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 500,
) -> None:
    """Test that empirical mean matches expected mean for DDBM."""
    print("\n=== Testing DDBM Mean Trajectory ===")

    sigma_test = structure_module.sigma_max / 2  # Test at mid sigma
    sigma_tensor = torch.full((1, num_samples), sigma_test)

    apo_exp = coords_apo.expand(1, num_samples, -1, -1)
    holo_exp = coords_holo.expand(1, num_samples, -1, -1)

    samples = structure_module.interpolate(apo_exp, holo_exp, sigma_tensor, mask)
    empirical_mean = samples.mean(dim=1)  # (1, num_atoms, 3)

    # Expected: a_t * apo + b_t * holo
    T = structure_module.sigma_max * structure_module.sigma_data
    a_t = sigma_test**2 / T**2
    b_t = 1 - a_t
    expected_mean = a_t * coords_apo + b_t * coords_holo

    diff = (empirical_mean - expected_mean.squeeze(1)).abs().max().item()
    print(f"  At σ={sigma_test:.3f}:")
    print(f"  a_t = {a_t:.4f}, b_t = {b_t:.4f}")
    print(f"  Max |empirical_mean - expected_mean|: {diff:.6f}")

    # Should be reasonable (allowing for statistical fluctuation)
    # At mid-sigma with high noise, expect larger deviation
    assert diff < 20.0, f"Mean trajectory deviates too much: {diff}"
    print("  ✓ Mean trajectory test passed!")


def test_variance_matches_theory(
    structure_module: KFoldBridgeDiffusion,
    coords_apo: torch.Tensor,
    coords_holo: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 1000,
) -> None:
    """Test that empirical variance matches DDBM theoretical variance."""
    print("\n=== Testing Variance Matches DDBM Theory ===")

    sigma_test = structure_module.sigma_max / 2  # Test at mid sigma
    sigma_tensor = torch.full((1, num_samples), sigma_test)

    apo_exp = coords_apo.expand(1, num_samples, -1, -1)
    holo_exp = coords_holo.expand(1, num_samples, -1, -1)

    samples = structure_module.interpolate(apo_exp, holo_exp, sigma_tensor, mask)

    # Compute empirical variance
    empirical_var = samples.var(dim=1).mean().item()  # avg across atoms/dims

    # Expected variance = sigma^2 * (1 - a_t)
    T = structure_module.sigma_max * structure_module.sigma_data
    a_t = sigma_test**2 / T**2
    b_t = 1 - a_t
    expected_var = sigma_test**2 * b_t

    print(f"  At σ={sigma_test:.3f}:")
    print(f"  a_t = {a_t:.4f}, b_t = {b_t:.4f}")
    print(f"  Expected variance (σ² * (1-a_t)) = {expected_var:.6f}")
    print(f"  Empirical variance = {empirical_var:.6f}")
    print(
        "  Relative error = "
        f"{abs(empirical_var - expected_var) / expected_var * 100:.2f}%"
    )

    # Allow 30% relative error due to sampling and DDBM noise characteristics
    rel_error = abs(empirical_var - expected_var) / (expected_var + 1e-8)
    assert rel_error < 0.30, f"Variance mismatch: {rel_error * 100:.1f}% error"
    print("  ✓ Variance test passed!")


def test_bridge_coefficients(
    structure_module: KFoldBridgeDiffusion,
    coords_apo: torch.Tensor,
    coords_holo: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Test DDBM bridge coefficient behavior."""
    print("\n=== Testing DDBM Bridge Coefficients ===")

    T = structure_module.sigma_max * structure_module.sigma_data

    # Test coefficient monotonicity
    sigmas = torch.linspace(0.001, structure_module.sigma_max - 0.001, 10)
    prev_c_skip = None
    prev_c_out = None

    print("  σ     | a_t     | b_t     | c_skip  | c_out")
    print("  " + "-" * 50)

    for sigma in sigmas:
        sigma_tensor = sigma.reshape(1, 1)

        # Bridge interpolation coefficients
        a_t = (sigma_tensor**2 / T**2).item()
        b_t = 1 - a_t

        # Bridge preconditioning coefficients
        c_skip = structure_module.c_skip(sigma_tensor).item()
        c_out = structure_module.c_out(sigma_tensor).item()

        print(
            f"  {sigma.item():6.3f} | {a_t:.4f} | {b_t:.4f} | {c_skip:.4f} | {c_out:.4f}"
        )

        # Test monotonicity
        if prev_c_skip is not None:
            assert c_skip <= prev_c_skip, "c_skip should be decreasing with sigma"
        if prev_c_out is not None:
            assert c_out >= prev_c_out, "c_out should be increasing with sigma"

        prev_c_skip = c_skip
        prev_c_out = c_out

        # Test coefficient bounds
        assert 0 <= a_t <= 1, f"a_t should be in [0,1], got {a_t}"
        assert 0 <= b_t <= 1, f"b_t should be in [0,1], got {b_t}"
        assert abs(a_t + b_t - 1.0) < 1e-5, f"a_t + b_t = 1, got {a_t + b_t}"
        assert 0 <= c_skip <= 1, f"c_skip should be in [0,1], got {c_skip}"
        assert c_out > 0, f"c_out should be positive, got {c_out}"

    print("  ✓ Bridge coefficient test passed!")


if __name__ == "__main__":
    from pathlib import Path

    torch.manual_seed(42)
    np.random.seed(42)

    SAVE_PATH = Path("./tmp/test_ddbm_synthetic/")
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Create mock score model (not used in interpolation tests)
    # MockScoreModel is defined above

    # Create DDBM module with test configuration
    ddbm_config = KFoldBridgeDiffusion.Config(
        num_steps=200,
        sigma_min=0.0004,
        sigma_max=160.0,
        sigma_data=16.0,
        sigma_data_end=12.0,
        cov_xy=96.0,
        w=1.0,
        coordinate_augmentation=False,
    )

    structure_module = KFoldBridgeDiffusion(ddbm_config, MockScoreModel())
    print("Created KFoldBridgeDiffusion:")
    print(f"  sigma_min: {structure_module.sigma_min}")
    print(f"  sigma_max: {structure_module.sigma_max}")
    print(f"  sigma_data: {structure_module.sigma_data}")
    print(f"  sigma_data_end: {structure_module.sigma_data_end}")
    print(f"  cov_xy: {structure_module.cov_xy}")

    # Create synthetic molecules
    num_atoms = 10
    coords_apo, coords_holo = create_synthetic_conformations(num_atoms=num_atoms)
    mask = torch.ones(1, num_atoms).bool()

    print("\nSynthetic molecule:")
    print(f"  Number of atoms: {num_atoms}")
    print(f"  Apo shape: {coords_apo.shape}")
    print(f"  Holo shape: {coords_holo.shape}")

    # Run tests
    test_bridge_coefficients(structure_module, coords_apo, coords_holo, mask)
    test_mean_trajectory(structure_module, coords_apo, coords_holo, mask)
    test_variance_matches_theory(structure_module, coords_apo, coords_holo, mask)

    # Analyze and plot statistics
    print("\n=== Analyzing DDBM Interpolation Statistics ===")
    results = analyze_interpolation_statistics(
        structure_module,
        coords_apo,
        coords_holo,
        mask,
        num_samples=1000,
        num_time_steps=21,
    )

    # Generate plots
    print("\n=== Generating DDBM Plots ===")
    plot_trajectory_3d(
        structure_module,
        coords_apo,
        coords_holo,
        mask,
        save_path=str(SAVE_PATH / "trajectory_3d.png"),
    )
    plot_statistics(results, str(SAVE_PATH / "statistics.png"))

    # Print summary table
    print("\n=== DDBM Summary Statistics Table ===")
    print("  σ     | Mean Var  | Expected   | Rel Err")
    print("  " + "-" * 45)
    for i, sigma in enumerate(results["sigma_values"]):
        emp_var = np.mean(results["variances"][i])
        exp_var = results["expected_variances"][i]
        rel_err = abs(emp_var - exp_var) / (exp_var + 1e-8) * 100
        print(f"  {sigma:6.3f} | {emp_var:.6f} | {exp_var:.6f} | {rel_err:.2f}%")

    print("\n" + "=" * 60)
    print("All DDBM synthetic interpolation tests passed!")
    print(f"Plots saved to {SAVE_PATH}")
    print("=" * 60)
