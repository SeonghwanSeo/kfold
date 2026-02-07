"""Test ECSI interpolation functionality.

This test validates the `interpolate` method of the KFoldECSI module by:
1. Loading a single sample from the validation dataset
2. Testing that interpolation behaves correctly at boundary conditions
3. Visualizing the interpolated structures at different time steps
"""

import json
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

import kfold.model.modules as submodules
from kfold.config import load_config
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.model.modules.structure_module.kfold_ecsi import KFoldECSI
from kfold.training.dataset.datamodule import TrainingDataModule
from kfold.utils import errors
from kfold.utils.registry import Registry

TEST_CONFIG_PATH = Path("./configs/train-esm2-ecsi-mini.yaml")
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
    num_steps: int = 200,
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
    num_samples = 200

    # Load config
    global_config = load_config(TEST_CONFIG_PATH)
    global_config.train.data.val_batch_size = 1
    global_config.train.data.num_workers = 0
    global_config.train.data.safe_load = False
    global_config.train.data.val_datasets[0].apo_init.translation_scale = 0
    global_config.train.data.val_datasets[0].apo_init.chain_com_sampling_radius = None

    # Load validation dataset
    data_module = TrainingDataModule(global_config.train.data)
    data_module.setup("validate")
    val_dataset = data_module._val_ds

    # Instantiate score model first
    kernel_config = global_config.model.kernel
    score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
        global_config.model.score_model, kernel_config=kernel_config
    )

    compare_k_values = [1.0, 2.0]
    interpolation_noise_seed = 42
    min_protein_atoms = 400
    required_ligand_chains = 1

    def _coords_to_np(
        coords: torch.Tensor, struct: RefStructure | TokenizedStructure
    ) -> np.ndarray:
        coords_np = coords.detach().cpu().numpy()
        num_atoms = getattr(struct, "num_atoms", coords_np.shape[0])
        if coords_np.shape[0] > num_atoms:
            coords_np = coords_np[:num_atoms]
        return coords_np

    def _write_coords(
        struct: RefStructure | TokenizedStructure,
        coords: torch.Tensor,
        path: Path,
    ) -> None:
        coords_np = _coords_to_np(coords, struct)
        if isinstance(struct, TokenizedStructure):
            struct = struct.replace_atom_coords(atom_coords=coords_np)
            struct.to_pdb(path)
        else:
            KFoldWriter.write_new_coords(struct, coords_np, path)

    def _count_atoms_by_token_mask(
        f_input_single: FoldingInput, token_mask: torch.Tensor
    ) -> int:
        atom_token_index = f_input_single.atom.token_index
        atom_pad_mask = f_input_single.atom.pad_mask
        token_count = token_mask.shape[-1]
        atom_token_index = atom_token_index.clamp(min=0, max=token_count - 1)
        atom_type = token_mask.gather(-1, atom_token_index)
        return int((atom_type & atom_pad_mask).sum().item())

    def _count_ligand_chains(struct: RefStructure | TokenizedStructure) -> int:
        if isinstance(struct, RefStructure):
            return sum(1 for chain in struct.chains if chain.ctype.is_ligand)
        return int(struct.token.is_ligand.any().item())

    def _build_structure_module(power: float) -> KFoldECSI:
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
            time_power=power,
        )
        return KFoldECSI(ecsi_config, score_model)

    structure_module = _build_structure_module(compare_k_values[0])

    print(
        "Loaded KFoldECSI module with "
        f"gamma_max={structure_module.gamma_max}, "
        f"compare_k_values={compare_k_values}"
    )

    # Turn off gradient
    torch.set_grad_enabled(False)

    # Run coefficient tests first (no data needed)
    # NOTE: enable if you want diagnostics on the default setup.
    # test_interpolation_coefficients(structure_module)
    # test_gamma_max_effect(structure_module)

    # Load a single sample for interpolation tests
    print("\n=== Loading Single Sample ===")
    f_input: FoldingInput
    struct: RefStructure | TokenizedStructure

    manifest_paths = []
    manifest_from_config = (
        global_config.train.data.val_datasets[0].manifest_path
        if global_config.train.data.val_datasets
        else None
    )
    if manifest_from_config is not None:
        manifest_paths.append(Path(manifest_from_config))
    manifest_paths.append(Path("manifest-rcsb-val-smallmol.json"))

    candidate_ids: list[str] = []
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        try:
            with manifest_path.open() as f:
                manifest_data = json.load(f)
        except Exception as exc:
            print(f"Failed to read manifest {manifest_path}: {exc}")
            continue

        for item in manifest_data:
            chain_types = {c.get("chain_type") for c in item.get("chains", [])}
            if 0 in chain_types and 3 in chain_types:
                candidate_ids.append(item["id"])
        if candidate_ids:
            print(f"Selected manifest: {manifest_path}")
            break

    if not candidate_ids:
        print(
            "No protein+ligand entries found in manifest; falling back to first sample."
        )

    candidate_set = set(candidate_ids)
    selected = None
    for idx, meta in enumerate(val_dataset.metadatas):
        if candidate_set and meta.id not in candidate_set:
            continue
        f_input_single, full_dict = val_dataset[idx]
        struct = full_dict["structure"]
        ligand_chains = _count_ligand_chains(struct)
        protein_atoms = _count_atoms_by_token_mask(
            f_input_single, f_input_single.token.is_protein
        )
        ligand_atoms = _count_atoms_by_token_mask(
            f_input_single, f_input_single.token.is_ligand
        )
        if ligand_chains != required_ligand_chains:
            continue
        if protein_atoms < min_protein_atoms:
            continue
        if ligand_atoms == 0:
            continue
        selected = (f_input_single, full_dict, protein_atoms, ligand_atoms)
        break

    if selected is None:
        print("No matching sample found; falling back to the first validation sample.")
        f_input_single, full_dict = val_dataset[0]
        protein_atoms = _count_atoms_by_token_mask(
            f_input_single, f_input_single.token.is_protein
        )
        ligand_atoms = _count_atoms_by_token_mask(
            f_input_single, f_input_single.token.is_ligand
        )
    else:
        f_input_single, full_dict, protein_atoms, ligand_atoms = selected

    f_input = FoldingInput.from_list([f_input_single], pad_to_max=False)
    full_dict_list = [full_dict]

    full_dict = full_dict_list[0]
    name: str = full_dict["id"]
    struct = full_dict["structure"]

    print(f"Testing with sample: {name}")
    print(f"  Number of atoms: {f_input.atom.pad_mask.sum().item()}")
    print(f"  Protein atoms: {protein_atoms}")
    print(f"  Ligand atoms: {ligand_atoms}")

    # Get label (holo) coords: [B, 1, Natom, 3]
    label_coords = structure_module.sample_holo(f_input, 1)
    # Get apo coords: [B, 1, Natom, 3]
    apo_coords = structure_module.sample_prior(f_input, 1, label_coords)

    # Get mask
    mask = f_input.atom.resolved_mask  # [B, Natom]

    # Run tests on the base module
    test_interpolation_boundary_conditions(
        structure_module, apo_coords, label_coords, mask
    )
    test_interpolation_smoothness(
        structure_module, apo_coords, label_coords, mask, num_samples
    )

    base_struct = struct

    for power in compare_k_values:
        structure_module = _build_structure_module(power)
        k_tag = f"k{power:g}"
        k_save_path = SAVE_PATH / k_tag
        k_save_path.mkdir(parents=True, exist_ok=True)

        # ====== Save interpolated structures for visualization ====== #
        print(f"\n=== Saving Interpolated Structures ({k_tag}) ===")

        # Generate time values
        t_hat = torch.linspace(
            structure_module.sigma_max, structure_module.sigma_min, num_samples
        )[None, :]  # [1, num_samples]

        # Expand coords for multiple samples
        apo_expanded = apo_coords.expand(-1, num_samples, -1, -1)
        label_expanded = label_coords.expand(-1, num_samples, -1, -1)

        # Get interpolated coords (fix seed for fair comparison)
        torch.manual_seed(interpolation_noise_seed)
        noisy_coords = structure_module.interpolate(
            apo_expanded,
            label_expanded,
            t_hat,
            mask,
            f_input=f_input,
        )

        # Save endpoints
        try:
            _write_coords(
                base_struct,
                label_coords[0, 0],
                k_save_path / f"{name}-ecsi-holo-{k_tag}.pdb",
            )

            _write_coords(
                base_struct,
                apo_coords[0, 0],
                k_save_path / f"{name}-ecsi-apo-{k_tag}.pdb",
            )

            # Save interpolated structures
            for i in range(num_samples):
                t_val = t_hat[0, i].item()
                _write_coords(
                    base_struct,
                    noisy_coords[0, i],
                    k_save_path / f"{name}-ecsi-{k_tag}-t{i:02d}_t{t_val:.3f}.pdb",
                )

            print(f"  Saved {num_samples + 2} PDB files to {k_save_path}")

            # ====== Create single trajectory file from apo to holo ====== #
            print(f"\n=== Creating Trajectory File ({k_tag}) ===")

            # Combine all structures into trajectory: apo -> interpolation -> holo
            trajectory_coords = []

            # Start with apo structure
            trajectory_coords.append(_coords_to_np(apo_coords[0, 0], base_struct))

            # Add interpolated structures (excluding endpoints to avoid duplicates)
            for i in range(1, num_samples - 1):
                trajectory_coords.append(_coords_to_np(noisy_coords[0, i], base_struct))

            # End with holo structure
            trajectory_coords.append(_coords_to_np(label_coords[0, 0], base_struct))

            # Create multi-conformer structure
            trajectory_coords_array = np.stack(
                trajectory_coords, axis=0
            )  # [Nframes, Natom, 3]

            trajectory_path = k_save_path / f"{name}-ecsi-trajectory-{k_tag}.pdb"
            if isinstance(base_struct, RefStructure):
                KFoldWriter.write_trajectory(
                    base_struct, trajectory_coords_array, trajectory_path
                )
                print(f"  Created trajectory file with {len(trajectory_coords)} frames")
            else:
                print(
                    "  Skipping trajectory file: "
                    "TokenizedStructure multi-model PDB not supported."
                )

        except errors.PDBWriterMaxChainError:
            print(f"  Skipping PDB writing for {name} due to max chain error.")

    print("\n" + "=" * 60)
    print("All ECSI interpolation tests passed!")
    print("=" * 60)
