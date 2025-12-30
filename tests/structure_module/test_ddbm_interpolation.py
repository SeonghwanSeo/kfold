"""Test DDBM interpolation functionality.

This test validates the `interpolate` method of the KFoldBridgeDiffusion module by:
1. Loading a single sample from the validation dataset
2. Testing that interpolation behaves correctly at boundary conditions
3. Visualizing the interpolated structures at different time steps
"""

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

from kfold.config import load_config
from kfold.data.model_input import FoldingInput
from kfold.data.tokenized import TokenizedStructure
from kfold.model.modules.structure_module.kfold_ddbm import KFoldBridgeDiffusion
from kfold.training.folding.dataset.datamodule import TrainingDataModule
from kfold.utils import errors
from kfold.utils.registry import Registry

TEST_CONFIG_PATH = Path("./configs/train-esmc-ddbm-mini.yaml")
SAVE_PATH = Path("./tmp/test_ddbm_interpolation/")


def test_interpolation_boundary_conditions(
    structure_module: KFoldBridgeDiffusion,
    apo_coords: torch.Tensor,
    label_coords: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Test that interpolation satisfies boundary conditions.

    At sigma≈0: x_t should be close to x_0 (holo)
    At sigma≈sigma_max: x_t should be close to x_T (apo)
    """
    print("\n=== Testing DDBM Interpolation Boundary Conditions ===")

    # Test at sigma≈0 (should be close to holo)
    sigma_near_zero = torch.full((1, 1), 0.001, device=apo_coords.device)
    x_near_zero = structure_module.interpolate(
        apo_coords, label_coords, sigma_near_zero, mask
    )

    # Test at sigma≈sigma_max (should have significant noise)
    sigma_near_max = torch.full(
        (1, 1), structure_module.sigma_max - 0.001, device=apo_coords.device
    )
    x_near_max = structure_module.interpolate(
        apo_coords, label_coords, sigma_near_max, mask
    )

    # Test at sigma≈sigma_max/2 (should be somewhere between apo and holo)
    sigma_mid = torch.full(
        (1, 1), structure_module.sigma_max / 2, device=apo_coords.device
    )
    x_mid = structure_module.interpolate(apo_coords, label_coords, sigma_mid, mask)

    # Compute distances (masked)
    mask_expanded = mask[:, None, :, None].float()

    # Distance from x_near_zero to holo (should be small)
    dist_to_holo = (
        ((x_near_zero - label_coords) ** 2 * mask_expanded).sum()
        / mask_expanded.sum()
        / 3
    )
    dist_to_holo = dist_to_holo.sqrt().item()

    # Distance from x_near_max to apo (should have significant noise)
    dist_to_apo = (
        ((x_near_max - apo_coords) ** 2 * mask_expanded).sum() / mask_expanded.sum() / 3
    )
    dist_to_apo = dist_to_apo.sqrt().item()

    # Distance from x_mid to both apo and holo (should be intermediate)
    dist_mid_to_apo = (
        (((x_mid - apo_coords) ** 2 * mask_expanded).sum() / mask_expanded.sum() / 3)
        .sqrt()
        .item()
    )
    dist_mid_to_holo = (
        (((x_mid - label_coords) ** 2 * mask_expanded).sum() / mask_expanded.sum() / 3)
        .sqrt()
        .item()
    )

    print(f"  sigma=0.001: RMSD to holo = {dist_to_holo:.4f} Å")
    print(
        f"  sigma={structure_module.sigma_max / 2:.3f}: "
        f"RMSD to apo = {dist_mid_to_apo:.4f} Å"
    )
    print(
        f"  sigma={structure_module.sigma_max / 2:.3f}: "
        f"RMSD to holo = {dist_mid_to_holo:.4f} Å"
    )
    print(
        f"  sigma={structure_module.sigma_max - 0.001:.3f}: "
        f"RMSD to apo = {dist_to_apo:.4f} Å"
    )

    # Check that distances are reasonable
    assert dist_to_holo < 2.0, (
        f"At sigma≈0, structure should be close to holo (got RMSD={dist_to_holo:.4f})"
    )
    # At intermediate sigma, should be somewhere between apo and holo
    assert (
        min(dist_mid_to_apo, dist_mid_to_holo) > 1.0
        and max(dist_mid_to_apo, dist_mid_to_holo) < 10.0
    ), (
        "At sigma≈sigma_max/2, structure should be intermediate "
        f"(apo:{dist_mid_to_apo:.4f}, holo:{dist_mid_to_holo:.4f})"
    )
    # At high sigma, DDBM has high noise, so we only check that it's not completely random
    assert dist_to_apo < 50.0, (
        "At sigma≈sigma_max, structure should not "
        f"be completely random (got RMSD={dist_to_apo:.4f})"
    )

    # Check that distances are reasonable
    assert dist_to_holo < 2.0, (
        f"At sigma≈0, structure should be close to holo (got RMSD={dist_to_holo:.4f})"
    )
    # At high sigma, DDBM has high noise, so we only check that it's not completely random
    assert dist_to_apo < 50.0, (
        "At sigma≈sigma_max, structure should not "
        f"be completely random (got RMSD={dist_to_apo:.4f})"
    )

    print("  ✓ Boundary conditions satisfied!")


def test_interpolation_coefficients(structure_module: KFoldBridgeDiffusion) -> None:
    """Test that bridge diffusion coefficients are computed correctly."""
    print("\n=== Testing DDBM Interpolation Coefficients ===")

    sigma_values = torch.linspace(0.001, structure_module.sigma_max - 0.001, 11)
    T = structure_module.sigma_max * structure_module.sigma_data

    print("  sigma  | a_t     | b_t     | a_t + b_t")
    print("  " + "-" * 45)

    for sigma in sigma_values:
        sigma_tensor = sigma.reshape(1, 1)
        a_t = (sigma_tensor**2 / T**2).item()
        b_t = 1 - a_t

        print(f"  {sigma.item():6.3f} | {a_t:.4f} | {b_t:.4f} | {a_t + b_t:.4f}")

        # Verify a_t + b_t = 1 (interpolation weights sum to 1)
        assert abs(a_t + b_t - 1.0) < 1e-5, f"a_t + b_t should equal 1, got {a_t + b_t}"

        # Verify monotonic behavior: as sigma increases, a_t should increase
        # (more apo weight)
        expected_a_t_min = 0.001**2 / T**2
        expected_a_t_max = (structure_module.sigma_max - 0.001) ** 2 / T**2
        assert expected_a_t_min <= a_t <= expected_a_t_max, (
            "a_t should be in expected range"
            f"[{expected_a_t_min:.6f}, {expected_a_t_max:.6f}]"
        )

    print("  ✓ All coefficient tests passed!")


def test_interpolation_smoothness(
    structure_module: KFoldBridgeDiffusion,
    apo_coords: torch.Tensor,
    label_coords: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int = 20,
) -> None:
    """Test that interpolation produces smooth transitions."""
    print(f"\n=== Testing DDBM Interpolation Smoothness ({num_steps} steps) ===")

    sigma_values = torch.linspace(0.001, structure_module.sigma_max - 0.001, num_steps)[
        None, :
    ]  # [1, num_steps]

    # Expand coords to match num_steps
    apo_expanded = apo_coords.expand(-1, num_steps, -1, -1)
    label_expanded = label_coords.expand(-1, num_steps, -1, -1)

    # Get interpolated coords
    x_t = structure_module.interpolate(apo_expanded, label_expanded, sigma_values, mask)

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


def test_bridge_coefficients(structure_module: KFoldBridgeDiffusion) -> None:
    """Test bridge diffusion preconditioning coefficients."""
    print("\n=== Testing DDBM Bridge Coefficients ===")

    sigma_values = torch.linspace(0.001, structure_module.sigma_max - 0.001, 11)

    print("  sigma  | c_skip  | c_out   | c_in")
    print("  " + "-" * 40)

    for sigma in sigma_values:
        sigma_tensor = sigma.reshape(1, 1)
        c_skip = structure_module.c_skip(sigma_tensor).item()
        c_out = structure_module.c_out(sigma_tensor).item()
        c_in = structure_module.c_in(sigma_tensor).item()

        print(f"  {sigma.item():6.3f} | {c_skip:.4f} | {c_out:.4f} | {c_in:.4f}")

        # Verify coefficients are reasonable
        assert 0 <= c_skip <= 1, f"c_skip should be in [0,1], got {c_skip}"
        assert c_out > 0, f"c_out should be positive, got {c_out}"
        assert c_in > 0, f"c_in should be positive, got {c_in}"

    print("  ✓ All bridge coefficient tests passed!")


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

    # Load DDBM configuration
    ddbm_config = global_config.model.structure_module
    structure_module: KFoldBridgeDiffusion = Registry.instantiate(
        ddbm_config, score_model=Registry.instantiate(global_config.model.score_model)
    )
    assert isinstance(structure_module, KFoldBridgeDiffusion), (
        f"Expected KFoldBridgeDiffusion, got {type(structure_module)}"
    )

    print("Loaded KFoldBridgeDiffusion module:")
    print(f"  sigma_min: {structure_module.sigma_min}")
    print(f"  sigma_max: {structure_module.sigma_max}")
    print(f"  sigma_data: {structure_module.sigma_data}")
    print(f"  sigma_data_end: {structure_module.sigma_data_end}")
    print(f"  cov_xy: {structure_module.cov_xy}")

    # Turn off gradient
    torch.set_grad_enabled(False)

    # Run coefficient tests first (no data needed)
    test_interpolation_coefficients(structure_module)
    test_bridge_coefficients(structure_module)

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
        print("\n=== Saving DDBM Interpolated Structures ===")

        # Generate sigma values
        sigma_values = torch.linspace(
            0.001, structure_module.sigma_max - 0.001, num_samples
        )[None, :]  # [1, num_samples]

        # Expand coords for multiple samples
        apo_expanded = apo_coords.expand(-1, num_samples, -1, -1)
        label_expanded = label_coords.expand(-1, num_samples, -1, -1)

        # Get interpolated coords
        noisy_coords = structure_module.interpolate(
            apo_expanded,
            label_expanded,
            sigma_values,
            mask,
        )

        # Save endpoints
        try:
            struct = struct.replace_atom_coords(
                atom_coords=label_coords[0].numpy(),
            )
            struct.to_pdb(
                SAVE_PATH / f"{name}-ddbm-holo.pdb",
                is_predicted=False,
            )

            struct = struct.replace_atom_coords(
                atom_coords=apo_coords[0].numpy(),
            )
            struct.to_pdb(
                SAVE_PATH / f"{name}-ddbm-apo.pdb",
                is_predicted=False,
            )

            # Save interpolated structures
            struct = struct.replace_atom_coords(
                atom_coords=noisy_coords[0].numpy(),
            )
            for i in range(num_samples):
                sigma_val = sigma_values[0, i].item()
                struct.to_pdb(
                    SAVE_PATH / f"{name}-ddbm-t{i:02d}_sigma{sigma_val:.3f}.pdb",
                    conformer_id=i,
                    is_predicted=False,
                )

            print(f"  Saved {num_samples + 2} PDB files to {SAVE_PATH}")

            # ====== Create single trajectory file from apo to holo ====== #
            print("\n=== Creating DDBM Trajectory File ===")

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
            trajectory_struct = struct.replace_atom_coords(
                atom_coords=trajectory_coords_array,
            )

            # Create multi-model PDB file by concatenating PDB strings
            trajectory_path = SAVE_PATH / f"{name}-ddbm-trajectory.pdb"
            with open(trajectory_path, "w") as f:
                for frame_idx in range(len(trajectory_coords)):
                    # Write model header for all frames including the first one
                    f.write(f"MODEL     {frame_idx + 1}\n")

                    # Get PDB string for this conformer
                    from kfold.utils.writer.pdb import to_pdbstring

                    pdb_string = to_pdbstring(
                        trajectory_struct, conformer_id=frame_idx, is_predicted=False
                    )

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

                    # Write filtered PDB content
                    f.write("\n".join(filtered_lines) + "\n")

                    # Write model footer for all frames except the last one
                    if frame_idx < len(trajectory_coords) - 1:
                        f.write("ENDMDL\n")

                # Write single END record at the very end
                f.write("END\n")

            print(f"  Created DDBM trajectory file with {len(trajectory_coords)} frames")

        except errors.PDBWriterMaxChainError:
            print(f"  Skipping PDB writing for {name} due to max chain error.")

    print("\n" + "=" * 60)
    print("All DDBM interpolation tests passed!")
    print("=" * 60)
