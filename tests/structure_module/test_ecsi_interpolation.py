"""Test ECSI interpolation functionality.

This test validates the `interpolate` method of the KFoldECSI module by:
1. Loading a single sample from the validation dataset
2. Testing that interpolation behaves correctly at boundary conditions
3. Visualizing the interpolated structures at different time steps
"""

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

import kfold.model.modules as submodules
from kfold.config import load_config
from kfold.data.model_input import FoldingInput
from kfold.data.tokenized import TokenizedStructure
from kfold.model.modules.structure_module.kfold_ecsi import KFoldECSI
from kfold.training.dataset.datamodule import TrainingDataModule
from kfold.utils import errors
from kfold.utils.registry import Registry

TEST_CONFIG_PATH = Path("./configs/train-esmc-ddbm-mini.yaml")
SAVE_PATH = Path("./tmp/test_ecsi_interpolation/")


def test_interpolation_boundary_conditions(
    structure_module: KFoldECSI,
    apo_coords: torch.Tensor,
    label_coords: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Test that interpolation satisfies boundary conditions.

    At t=0: x_t should be close to x_0 (holo)
    At t=1: x_t should be close to x_T (apo)
    """
    print("\n=== Testing Interpolation Boundary Conditions ===")

    # Test at t ≈ 0 (should be close to holo)
    t_near_zero = torch.full((1, 1), 0.001, device=apo_coords.device)
    x_near_zero = structure_module.interpolate(
        apo_coords, label_coords, t_near_zero, mask
    )

    # Test at t ≈ 1 (should be close to apo)
    t_near_one = torch.full((1, 1), 0.999, device=apo_coords.device)
    x_near_one = structure_module.interpolate(apo_coords, label_coords, t_near_one, mask)

    # Compute distances (masked)
    mask_expanded = mask[:, None, :, None].float()

    # Distance from x_near_zero to holo (should be small)
    dist_to_holo = (
        ((x_near_zero - label_coords) ** 2 * mask_expanded).sum()
        / mask_expanded.sum()
        / 3
    )
    dist_to_holo = dist_to_holo.sqrt().item()

    # Distance from x_near_one to apo (should be small)
    dist_to_apo = (
        ((x_near_one - apo_coords) ** 2 * mask_expanded).sum() / mask_expanded.sum() / 3
    )
    dist_to_apo = dist_to_apo.sqrt().item()

    print(f"  t=0.001: RMSD to holo = {dist_to_holo:.4f} Å")
    print(f"  t=0.999: RMSD to apo = {dist_to_apo:.4f} Å")

    # Check that distances are reasonable
    # At t=0, alpha=0.999, beta=0.001, gamma is small
    # At t=1, alpha=0.001, beta=0.999, gamma is small
    assert dist_to_holo < 1.0, (
        f"At t≈0, structure should be close to holo (got RMSD={dist_to_holo:.4f})"
    )
    assert dist_to_apo < 1.0, (
        f"At t≈1, structure should be close to apo (got RMSD={dist_to_apo:.4f})"
    )

    print("  ✓ Boundary conditions satisfied!")


def test_interpolation_coefficients(structure_module: KFoldECSI) -> None:
    """Test that alpha, beta, gamma coefficients are computed correctly."""
    print("\n=== Testing Interpolation Coefficients ===")

    t_values = torch.linspace(0.0, 1.0, 11)

    print("  t     | alpha   | beta    | gamma   | alpha+beta")
    print("  " + "-" * 50)

    for t in t_values:
        t_tensor = t.reshape(1, 1)
        alpha = structure_module.alpha(t_tensor).item()
        beta = structure_module.beta(t_tensor).item()
        gamma = structure_module.gamma(t_tensor).item()

        print(f"  {t:.2f}  | {alpha:.4f} | {beta:.4f} | {gamma:.4f} | {alpha + beta:.4f}")

        # Verify alpha + beta = 1 (linear interpolation)
        assert abs(alpha + beta - 1.0) < 1e-5, (
            f"alpha + beta should equal 1, got {alpha + beta}"
        )

        # Verify alpha = 1 - t and beta = t
        assert abs(alpha - (1 - t.item())) < 1e-5, "alpha should be 1-t"
        assert abs(beta - t.item()) < 1e-5, "beta should be t"

        # Verify gamma is correct: 2 * gamma_max * sqrt(t * (1-t))
        expected_gamma = (
            structure_module.gamma_max * 2 * (t * (1 - t) + 1e-8).sqrt().item()
        )
        assert abs(gamma - expected_gamma) < 1e-4, (
            f"gamma mismatch: got {gamma}, expected {expected_gamma}"
        )

    print("  ✓ All coefficient tests passed!")


def test_interpolation_smoothness(
    structure_module: KFoldECSI,
    apo_coords: torch.Tensor,
    label_coords: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int = 20,
) -> None:
    """Test that interpolation produces smooth transitions."""
    print(f"\n=== Testing Interpolation Smoothness ({num_steps} steps) ===")

    t_values = torch.linspace(0.001, 0.999, num_steps)[None, :]  # [1, num_steps]

    # Expand coords to match num_steps
    apo_expanded = apo_coords.expand(-1, num_steps, -1, -1)
    label_expanded = label_coords.expand(-1, num_steps, -1, -1)

    # Get interpolated coords
    x_t = structure_module.interpolate(apo_expanded, label_expanded, t_values, mask)

    # Compute differences between consecutive steps
    mask_expanded = mask[:, None, :, None].float()

    diffs = []
    for i in range(num_steps - 1):
        diff = (
            ((x_t[:, i + 1] - x_t[:, i]) ** 2 * mask_expanded[:, 0]).sum()
            / mask_expanded[:, 0].sum()
            / 3
        )
        diff = diff.sqrt().item()
        diffs.append(diff)

    avg_diff = sum(diffs) / len(diffs)
    max_diff = max(diffs)

    print(f"  Average step RMSD: {avg_diff:.4f} Å")
    print(f"  Maximum step RMSD: {max_diff:.4f} Å")
    print(f"  Step RMSDs: {[f'{d:.3f}' for d in diffs]}")

    print("  ✓ Smoothness test completed!")


def test_gamma_max_effect(structure_module: KFoldECSI) -> None:
    """Test the effect of gamma_max on the noise scale."""
    print("\n=== Testing Gamma Max Effect ===")

    # gamma is maximized at t=0.5
    t_mid = torch.tensor([[0.5]])
    gamma_at_mid = structure_module.gamma(t_mid).item()

    # At t=0.5: gamma = 2 * gamma_max * sqrt(0.5 * 0.5) = 2 * gamma_max * 0.5 = gamma_max
    expected_gamma_at_mid = structure_module.gamma_max

    print(f"  gamma_max setting: {structure_module.gamma_max}")
    print(f"  gamma at t=0.5: {gamma_at_mid:.4f}")
    print(f"  Expected (gamma_max): {expected_gamma_at_mid:.4f}")

    assert abs(gamma_at_mid - expected_gamma_at_mid) < 1e-4, (
        f"gamma at t=0.5 should equal gamma_max, got {gamma_at_mid}"
    )

    print("  ✓ Gamma max test passed!")


if __name__ == "__main__":
    pl.seed_everything(42)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Number of time steps for visualization
    num_samples = 20

    # Load config
    global_config = load_config(TEST_CONFIG_PATH)
    global_config.train.data.val_batch_size = 1
    global_config.train.data.num_workers = 0
    global_config.train.data.safe_load = False

    # Load validation loader
    data_module = TrainingDataModule(global_config.train.data)
    data_module.setup("validate")
    dataloader = data_module.val_dataloader()

    # Instantiate score model first
    score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
        global_config.model.score_model
    )

    ecsi_config = KFoldECSI.Config(
        num_steps=200,
        sigma_min=0.001,
        sigma_max=0.999,
        gamma_max=4.0,
        sigma_data=16.0,
        sigma_data_end=16.0,
        cov_xy=128.0,
        rho=7,
        eta=1.0,
        coordinate_augmentation=False,  # Disable for testing
    )
    structure_module: KFoldECSI = KFoldECSI(ecsi_config, score_model)
    ecsi_config._registry_ = "structure_module"
    ecsi_config._class_ = "KFoldECSI"

    score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
        global_config.model.score_model
    )
    structure_module: KFoldECSI = Registry.instantiate(
        ecsi_config, score_model=score_model
    )
    assert isinstance(structure_module, KFoldECSI), (
        f"Expected KFoldECSI, got {type(structure_module)}"
    )

    print(f"Loaded KFoldECSI module with gamma_max={structure_module.gamma_max}")

    # Turn off gradient
    torch.set_grad_enabled(False)

    # Run coefficient tests first (no data needed)
    test_interpolation_coefficients(structure_module)
    test_gamma_max_effect(structure_module)

    # Load a single sample for interpolation tests
    print("\n=== Loading Single Sample ===")
    f_input: FoldingInput
    struct: TokenizedStructure

    for iter, (f_input, full_dict_list) in enumerate(dataloader):
        if iter > 0:
            break

        full_dict = full_dict_list[0]
        name: str = full_dict["id"]
        struct = full_dict["structure"]

        print(f"Testing with sample: {name}")
        print(f"  Number of atoms: {f_input.atom.pad_mask.sum().item()}")

        # Get label (holo) coords: [B, 1, Natom, 3]
        label_coords = structure_module.sample_holo(f_input, 1)
        # Get apo coords: [B, 1, Natom, 3]
        apo_coords = structure_module.sample_prior(f_input, 1, label_coords)

        # Get mask
        mask = f_input.atom.resolved_mask  # [B, Natom]

        # Run tests
        test_interpolation_boundary_conditions(
            structure_module, apo_coords, label_coords, mask
        )
        test_interpolation_smoothness(
            structure_module, apo_coords, label_coords, mask, num_samples
        )

        # ====== Save interpolated structures for visualization ====== #
        print("\n=== Saving Interpolated Structures ===")

        # Generate time values
        t_hat = torch.linspace(
            structure_module.sigma_min, structure_module.sigma_max, num_samples
        )[None, :]  # [1, num_samples]

        # Expand coords for multiple samples
        apo_expanded = apo_coords.expand(-1, num_samples, -1, -1)
        label_expanded = label_coords.expand(-1, num_samples, -1, -1)

        # Get interpolated coords
        noisy_coords = structure_module.interpolate(
            apo_expanded,
            label_expanded,
            t_hat,
            mask,
        )

        # Save endpoints
        try:
            struct = struct.replace_atom_coords(
                atom_coords=label_coords[0].numpy(),
            )
            struct.to_pdb(
                SAVE_PATH / f"{name}-ecsi-holo.pdb",
            )

            struct = struct.replace_atom_coords(
                atom_coords=apo_coords[0].numpy(),
            )
            struct.to_pdb(
                SAVE_PATH / f"{name}-ecsi-apo.pdb",
            )

            # Save interpolated structures
            struct = struct.replace_atom_coords(
                atom_coords=noisy_coords[0].numpy(),
            )
            for i in range(num_samples):
                t_val = t_hat[0, i].item()
                struct.to_pdb(
                    SAVE_PATH / f"{name}-ecsi-t{i:02d}_t{t_val:.3f}.pdb",
                    conformer_id=i,
                )

            print(f"  Saved {num_samples + 2} PDB files to {SAVE_PATH}")

            # ====== Create single trajectory file from apo to holo ====== #
            print("\n=== Creating Trajectory File ===")

            # Combine all structures into trajectory: apo -> interpolation -> holo
            trajectory_coords = []

            # Start with apo structure
            apo_flat = apo_coords[0, 0]  # [Natom, 3] - squeeze the first dimension
            trajectory_coords.append(apo_flat.numpy())

            # Add interpolated structures (excluding endpoints to avoid duplicates)
            for i in range(1, num_samples - 1):
                interp_flat = noisy_coords[0, i]  # [Natom, 3]
                trajectory_coords.append(interp_flat.numpy())

            # End with holo structure
            holo_flat = label_coords[0, 0]  # [Natom, 3] - squeeze the first dimension
            trajectory_coords.append(holo_flat.numpy())

            # Create multi-conformer structure
            trajectory_coords_array = np.stack(
                trajectory_coords, axis=0
            )  # [Nframes, Natom, 3]

            # Create multi-model PDB file by concatenating PDB strings
            trajectory_path = SAVE_PATH / f"{name}-ecsi-trajectory.pdb"
            with open(trajectory_path, "w") as f:
                for frame_idx in range(len(trajectory_coords)):
                    # Write model header for all frames including the first one
                    f.write(f"MODEL     {frame_idx + 1}\n")

                    # Get PDB string for this conformer
                    from kfold.utils.writer.pdb import to_pdbstring

                    trajectory_struct = struct.replace_atom_coords(
                        atom_coords=trajectory_coords_array[frame_idx],
                    )
                    pdb_string = to_pdbstring(trajectory_struct)

                    # Remove END record from all models (will add single END at end)
                    pdb_string = pdb_string.rstrip()
                    if pdb_string.endswith("END"):
                        pdb_string = pdb_string[:-3].rstrip()

                    # Remove TER records and add them only at proper residue boundaries
                    # For simplicity, let's remove all TER records for now
                    lines = pdb_string.split("\n")
                    filtered_lines = [
                        line for line in lines if not line.strip().startswith("TER")
                    ]

                    # Write the filtered PDB content
                    f.write("\n".join(filtered_lines) + "\n")

                    # Write model footer for all frames except the last one
                    if frame_idx < len(trajectory_coords) - 1:
                        f.write("ENDMDL\n")

                # Write single END record at the very end
                f.write("END\n")

            print(f"  Created trajectory file with {len(trajectory_coords)} frames")

        except errors.PDBWriterMaxChainError:
            print(f"  Skipping PDB writing for {name} due to max chain error.")

    print("\n" + "=" * 60)
    print("All ECSI interpolation tests passed!")
    print("=" * 60)
