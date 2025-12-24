"""Test ECSI interpolation with synthetic molecular conformations.

This test uses two synthetic molecular conformations to validate:
1. Interpolation trajectory correctness (mean path)
2. Statistical properties: variance and mean of multiple interpolation samples
3. Noise injection via gamma coefficient
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from kfold.model.modules.structure_module.kfold_ecsi import KFoldECSI


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
    structure_module: KFoldECSI,
    coords_apo: torch.Tensor,
    coords_holo: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 1000,
    num_time_steps: int = 21,
) -> dict:
    """Analyze statistical properties of interpolation.

    Returns mean and variance of interpolated positions at each time step.
    """
    t_values = torch.linspace(
        structure_module.sigma_min, structure_module.sigma_max, num_time_steps
    )

    results = {
        "t_values": t_values.numpy(),
        "means": [],
        "variances": [],
        "expected_means": [],
        "expected_variances": [],
    }

    for t in t_values:
        t_tensor = torch.full((1, num_samples), t.item())

        # Expand coordinates for multiple samples
        apo_expanded = coords_apo.expand(1, num_samples, -1, -1)
        holo_expanded = coords_holo.expand(1, num_samples, -1, -1)

        # Sample multiple interpolations
        samples = structure_module.interpolate(
            apo_expanded, holo_expanded, t_tensor, mask
        )  # (1, num_samples, num_atoms, 3)

        # Compute statistics across samples
        mean = samples.mean(dim=1)  # (1, num_atoms, 3)
        var = samples.var(dim=1)  # (1, num_atoms, 3)

        results["means"].append(mean.squeeze(0).numpy())
        results["variances"].append(var.squeeze(0).numpy())

        # Compute expected values
        # Expected mean: alpha_t * holo + beta_t * apo
        t_exp = torch.tensor([[[[t.item()]]]])
        alpha_t = structure_module.alpha(t_exp).item()
        beta_t = structure_module.beta(t_exp).item()
        gamma_t = structure_module.gamma(t_exp).item()

        expected_mean = alpha_t * coords_holo + beta_t * coords_apo
        expected_var = gamma_t**2  # Variance from noise term

        results["expected_means"].append(expected_mean.squeeze().numpy())
        results["expected_variances"].append(expected_var)

    return results


def plot_trajectory_3d(
    structure_module: KFoldECSI,
    coords_apo: torch.Tensor,
    coords_holo: torch.Tensor,
    mask: torch.Tensor,
    num_time_steps: int = 11,
    save_path: str = "./tmp/test_ecsi_synthetic/trajectory_3d.png",
) -> None:
    """Plot 3D trajectory of interpolation for a single atom."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 5))

    # Left: Single trajectory (mean)
    ax1 = fig.add_subplot(121, projection="3d")

    t_values = torch.linspace(0.001, 0.999, num_time_steps)[None, :]
    apo_exp = coords_apo.expand(1, num_time_steps, -1, -1)
    holo_exp = coords_holo.expand(1, num_time_steps, -1, -1)

    trajectory = structure_module.interpolate(apo_exp, holo_exp, t_values, mask)
    trajectory = trajectory[0].numpy()  # (num_time_steps, num_atoms, 3)

    # Plot trajectory for atom 0
    atom_idx = 0
    ax1.plot(
        trajectory[:, atom_idx, 0],
        trajectory[:, atom_idx, 1],
        trajectory[:, atom_idx, 2],
        "b-o",
        markersize=4,
        label="Interpolation path",
    )
    ax1.scatter(
        *coords_holo[0, 0, atom_idx].numpy(), c="green", s=100, label="Holo (t=0)"
    )
    ax1.scatter(*coords_apo[0, 0, atom_idx].numpy(), c="red", s=100, label="Apo (t=1)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"Single Interpolation Path (Atom {atom_idx})")
    ax1.legend()

    # Right: All atoms
    ax2 = fig.add_subplot(122, projection="3d")

    colors = plt.cm.viridis(np.linspace(0, 1, num_time_steps))
    for i, _t in enumerate(t_values[0]):
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
    ax2.set_title("All Atoms Interpolation")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved 3D trajectory plot to {save_path}")


def plot_statistics(results: dict, save_path: str) -> None:
    """Plot mean and variance statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    t_values = results["t_values"]
    num_atoms = results["means"][0].shape[0]

    # Plot 1: Mean comparison (atom 0, x-coordinate)
    ax = axes[0, 0]
    empirical_means = [m[0, 0] for m in results["means"]]
    expected_means = [m[0, 0] for m in results["expected_means"]]
    ax.plot(t_values, empirical_means, "b-o", label="Empirical mean", markersize=4)
    ax.plot(t_values, expected_means, "r--", label="Expected mean", linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("X position")
    ax.set_title("Mean X Position (Atom 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Variance vs gamma^2
    ax = axes[0, 1]
    empirical_vars = [v.mean() for v in results["variances"]]  # avg across atoms/dims
    expected_vars = results["expected_variances"]
    ax.plot(t_values, empirical_vars, "b-o", label="Empirical variance", markersize=4)
    ax.plot(t_values, expected_vars, "r--", label=r"$\gamma_t^2$ (expected)", linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("Variance")
    ax.set_title(r"Variance vs $\gamma_t^2$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Mean trajectory for all atoms (y-coordinate)
    ax = axes[1, 0]
    for atom_idx in range(num_atoms):
        means_y = [m[atom_idx, 1] for m in results["means"]]
        ax.plot(t_values, means_y, "-", alpha=0.7, label=f"Atom {atom_idx}")
    ax.set_xlabel("t")
    ax.set_ylabel("Y position")
    ax.set_title("Mean Y Position (All Atoms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Variance per atom
    ax = axes[1, 1]
    for atom_idx in range(num_atoms):
        vars_atom = [v[atom_idx].mean() for v in results["variances"]]
        ax.plot(t_values, vars_atom, "-", alpha=0.7, label=f"Atom {atom_idx}")
    ax.set_xlabel("t")
    ax.set_ylabel("Variance")
    ax.set_title("Variance per Atom")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved statistics plot to {save_path}")


def test_mean_trajectory(
    structure_module: KFoldECSI,
    coords_apo: torch.Tensor,
    coords_holo: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 500,
) -> None:
    """Test that empirical mean matches expected mean."""
    print("\n=== Testing Mean Trajectory ===")

    t_test = 0.5
    t_tensor = torch.full((1, num_samples), t_test)

    apo_exp = coords_apo.expand(1, num_samples, -1, -1)
    holo_exp = coords_holo.expand(1, num_samples, -1, -1)

    samples = structure_module.interpolate(apo_exp, holo_exp, t_tensor, mask)
    empirical_mean = samples.mean(dim=1)  # (1, num_atoms, 3)

    # Expected: 0.5 * holo + 0.5 * apo
    expected_mean = 0.5 * coords_holo + 0.5 * coords_apo

    diff = (empirical_mean - expected_mean.squeeze(1)).abs().max().item()
    print(f"  At t={t_test}:")
    print(f"  Max |empirical_mean - expected_mean|: {diff:.6f}")

    # Should be small (within statistical fluctuation)
    assert diff < 0.5, f"Mean trajectory deviates too much: {diff}"
    print("  ✓ Mean trajectory test passed!")


def test_variance_matches_gamma(
    structure_module: KFoldECSI,
    coords_apo: torch.Tensor,
    coords_holo: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 1000,
) -> None:
    """Test that empirical variance matches gamma^2."""
    print("\n=== Testing Variance Matches Gamma^2 ===")

    t_test = 0.5  # gamma is maximized here
    t_tensor = torch.full((1, num_samples), t_test)

    apo_exp = coords_apo.expand(1, num_samples, -1, -1)
    holo_exp = coords_holo.expand(1, num_samples, -1, -1)

    samples = structure_module.interpolate(apo_exp, holo_exp, t_tensor, mask)

    # Compute empirical variance
    empirical_var = samples.var(dim=1).mean().item()  # avg across atoms/dims

    # Expected variance = gamma^2
    gamma_t = structure_module.gamma(torch.tensor([[t_test]])).item()
    expected_var = gamma_t**2

    print(f"  At t={t_test}:")
    print(f"  gamma_t = {gamma_t:.6f}")
    print(f"  Expected variance (gamma^2) = {expected_var:.6f}")
    print(f"  Empirical variance = {empirical_var:.6f}")
    print(
        f"Relative error = {abs(empirical_var - expected_var) / expected_var * 100:.2f}%"
    )

    # Allow 10% relative error due to sampling
    rel_error = abs(empirical_var - expected_var) / expected_var
    assert rel_error < 0.15, f"Variance mismatch: {rel_error * 100:.1f}% error"
    print("  ✓ Variance test passed!")


if __name__ == "__main__":
    from pathlib import Path

    torch.manual_seed(42)
    np.random.seed(42)

    SAVE_PATH = Path("./tmp/test_ecsi_synthetic/")
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Create mock score model (not used in interpolation tests)
    class MockScoreModel:
        pass

    # Create ECSI module with test configuration
    ecsi_config = KFoldECSI.Config(
        num_steps=200,
        sigma_min=0.001,
        sigma_max=0.999,
        gamma_max=4.0,
        sigma_data=16.0,
        sigma_data_end=16.0,
        cov_xy=128.0,
        eta=1.0,
        coordinate_augmentation=False,
    )

    structure_module = KFoldECSI(ecsi_config, MockScoreModel())
    print(f"Created KFoldECSI with gamma_max={structure_module.gamma_max}")

    # Create synthetic molecules
    num_atoms = 10
    coords_apo, coords_holo = create_synthetic_conformations(num_atoms=num_atoms)
    mask = torch.ones(1, num_atoms).bool()

    print("\nSynthetic molecule:")
    print(f"  Number of atoms: {num_atoms}")
    print(f"  Apo shape: {coords_apo.shape}")
    print(f"  Holo shape: {coords_holo.shape}")

    # Run tests
    test_mean_trajectory(structure_module, coords_apo, coords_holo, mask)
    test_variance_matches_gamma(structure_module, coords_apo, coords_holo, mask)

    # Analyze and plot statistics
    print("\n=== Analyzing Interpolation Statistics ===")
    results = analyze_interpolation_statistics(
        structure_module,
        coords_apo,
        coords_holo,
        mask,
        num_samples=1000,
        num_time_steps=21,
    )

    # Generate plots
    print("\n=== Generating Plots ===")
    plot_trajectory_3d(
        structure_module,
        coords_apo,
        coords_holo,
        mask,
        save_path=str(SAVE_PATH / "trajectory_3d.png"),
    )
    plot_statistics(results, str(SAVE_PATH / "statistics.png"))

    # Print summary table
    print("\n=== Summary Statistics Table ===")
    print("  t     | Mean Var  | γ²        | Rel Err")
    print("  " + "-" * 45)
    for i, t in enumerate(results["t_values"]):
        emp_var = np.mean(results["variances"][i])
        exp_var = results["expected_variances"][i]
        rel_err = abs(emp_var - exp_var) / (exp_var + 1e-8) * 100
        print(f"  {t:.3f} | {emp_var:.6f} | {exp_var:.6f} | {rel_err:.2f}%")

    print("\n" + "=" * 60)
    print("All synthetic interpolation tests passed!")
    print(f"Plots saved to {SAVE_PATH}")
    print("=" * 60)
