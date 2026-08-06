"""Mask semantics around holo-unresolved atoms.

Two masks disagree on those atoms: `pad_mask` covers them, `resolved_mask` does
not. Training hides them from the coordinate stack; inference, which has no
`resolved_mask`, keeps using `pad_mask`.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from kfold.model.modules.structure.ecsi import KFoldECSI, custom_rigid_align

torch.set_float32_matmul_precision("highest")

# Atom layout shared by the `sample_train_input` tests below.
NUM_RESOLVED = 8  # resolved in the holo label and present in the prior
NUM_UNRESOLVED = 2  # present in the prior, unresolved in the holo label
NUM_PAD = 2  # padding
NUM_ATOMS = NUM_RESOLVED + NUM_UNRESOLVED + NUM_PAD
UNRESOLVED = slice(NUM_RESOLVED, NUM_RESOLVED + NUM_UNRESOLVED)
PADDING = slice(NUM_RESOLVED + NUM_UNRESOLVED, NUM_ATOMS)


@pytest.fixture
def coords() -> torch.Tensor:
    """A non-degenerate point cloud, shape [1, 8, 3]."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn((1, 8, 3), generator=generator) * 10.0


@pytest.fixture
def rot_matrix() -> torch.Tensor:
    """45 degree rotation matrix around the Z axis."""
    cos_a, sin_a = math.cos(math.pi / 4), math.sin(math.pi / 4)
    return torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def test_output_mask_defaults_to_fit_mask(coords: torch.Tensor):
    """Without `output_mask` the fit mask still governs the output."""
    mask = torch.ones((1, 8), dtype=torch.bool)
    mask[0, -2:] = False

    aligned = custom_rigid_align(coords, coords, mask, rotation_only=True)

    assert torch.allclose(aligned[0, -2:], torch.zeros(2, 3), atol=1e-6)


def test_output_mask_preserves_atoms_outside_the_fit(
    coords: torch.Tensor, rot_matrix: torch.Tensor
):
    """Atoms outside the fit mask are rotated, not zeroed."""
    # x_0 is only defined on the first six atoms; x_T is defined on all eight.
    fit_mask = torch.ones((1, 8), dtype=torch.bool)
    fit_mask[0, -2:] = False
    keep_mask = torch.ones((1, 8), dtype=torch.bool)

    rotated = coords @ rot_matrix.T

    aligned = custom_rigid_align(
        rotated, coords, fit_mask, rotation_only=True, output_mask=keep_mask
    )

    # No atom is discarded, and none lands on the origin.
    assert not torch.allclose(aligned[0, -2:], torch.zeros(2, 3), atol=1e-6)

    # The unfitted atoms undergo exactly the transform fitted on the others,
    # so the recovered cloud matches the original up to the residual translation.
    recovered = aligned - aligned.mean(dim=-2, keepdim=True)
    expected = coords - coords.mean(dim=-2, keepdim=True)
    assert torch.allclose(recovered, expected, atol=1e-4)


def test_fit_ignores_coordinates_outside_the_fit_mask(
    coords: torch.Tensor, rot_matrix: torch.Tensor
):
    """Corrupting an unfitted atom must not perturb the fitted transform."""
    fit_mask = torch.ones((1, 8), dtype=torch.bool)
    fit_mask[0, -1] = False
    keep_mask = torch.ones((1, 8), dtype=torch.bool)

    rotated = coords @ rot_matrix.T
    corrupted = rotated.clone()
    corrupted[0, -1] += 1000.0

    aligned = custom_rigid_align(
        rotated, coords, fit_mask, rotation_only=True, output_mask=keep_mask
    )
    aligned_corrupted = custom_rigid_align(
        corrupted, coords, fit_mask, rotation_only=True, output_mask=keep_mask
    )

    assert torch.allclose(aligned[0, :-1], aligned_corrupted[0, :-1], atol=1e-4)


def test_translation_is_applied_only_to_kept_atoms(coords: torch.Tensor):
    """With `rotation_only=False` the masked atoms stay at zero, not at +T."""
    fit_mask = torch.ones((1, 8), dtype=torch.bool)
    fit_mask[0, -2:] = False
    keep_mask = fit_mask.clone()

    translated = coords + torch.tensor([10.0, -5.0, 3.0])

    aligned = custom_rigid_align(
        translated, coords, fit_mask, rotation_only=False, output_mask=keep_mask
    )

    assert torch.allclose(aligned[0, :-2], coords[0, :-2], atol=1e-4)
    assert torch.allclose(aligned[0, -2:], torch.zeros(2, 3), atol=1e-6)


# ==========================================
# `sample_train_input` end-to-end
# ==========================================


def make_folding_input() -> SimpleNamespace:
    """Minimal stand-in for the fields `sample_train_input` reads."""
    generator = torch.Generator().manual_seed(0)
    label_coords = torch.randn((1, NUM_ATOMS, 3), generator=generator) * 10.0
    prior_coords = torch.randn((1, NUM_ATOMS, 1, 3), generator=generator) * 10.0

    resolved_mask = torch.zeros((1, NUM_ATOMS), dtype=torch.bool)
    resolved_mask[0, :NUM_RESOLVED] = True
    pad_mask = torch.zeros((1, NUM_ATOMS), dtype=torch.bool)
    pad_mask[0, : NUM_RESOLVED + NUM_UNRESOLVED] = True

    label_coords = label_coords.masked_fill(~resolved_mask[..., None], 0.0)
    prior_coords = prior_coords.masked_fill(~pad_mask[..., None, None], 0.0)

    return SimpleNamespace(
        batch_size=1,
        device=torch.device("cpu"),
        atom=SimpleNamespace(
            label_coords=label_coords,
            prior_coords=prior_coords,
            resolved_mask=resolved_mask,
            pad_mask=pad_mask,
        ),
    )


def test_unresolved_atoms_are_hidden_from_the_coordinate_stack():
    """Unresolved atoms leave the training mask and carry no coordinate."""
    module = KFoldECSI(KFoldECSI.Config(), None)
    f_input = make_folding_input()

    out = module.sample_train_input(f_input, diffusion_batch_size=2)

    expected = f_input.atom.pad_mask & f_input.atom.resolved_mask
    assert torch.equal(out["atom_mask"], expected)
    assert not out["atom_mask"][:, UNRESOLVED].any()
    assert not out["atom_mask"][:, PADDING].any()

    # Hidden atoms sit at the origin in every channel, so a path that ignores the
    # mask sees 0 rather than a prior-scale offset.
    for key in ("x_0", "x_t", "x_T"):
        assert torch.allclose(out[key][:, :, UNRESOLVED], torch.zeros(1), atol=1e-6)
        assert torch.allclose(out[key][:, :, PADDING], torch.zeros(1), atol=1e-6)


def test_resolved_atoms_keep_their_apo_geometry():
    """Hiding the unresolved atoms must not disturb the ones that remain."""
    module = KFoldECSI(KFoldECSI.Config(), None)
    f_input = make_folding_input()
    prior = f_input.atom.prior_coords[0, :, 0, :].clone()
    kept = f_input.atom.resolved_mask[0]

    out = module.sample_train_input(f_input, diffusion_batch_size=2)

    assert out["x_T"][:, :, :NUM_RESOLVED].abs().max() > 1.0

    # x_T is only rotated, so distances among the surviving atoms are unchanged.
    expected = torch.cdist(prior[kept], prior[kept])
    for sample in out["x_T"][0]:
        assert torch.allclose(
            torch.cdist(sample[kept], sample[kept]), expected, atol=1e-3
        )


def test_interpolation_is_exact_on_resolved_atoms():
    """The interpolant is untouched where both endpoints are defined."""
    # gamma_max = 0 removes the noise term, making the interpolant deterministic.
    module = KFoldECSI(KFoldECSI.Config(gamma_max=0.0), None)
    f_input = make_folding_input()

    out = module.sample_train_input(f_input, diffusion_batch_size=2)
    t = out["t"][:, :, None, None]

    assert torch.allclose(
        out["x_t"][:, :, :NUM_RESOLVED],
        (1 - t) * out["x_0"][:, :, :NUM_RESOLVED] + t * out["x_T"][:, :, :NUM_RESOLVED],
        atol=1e-4,
    )


def test_inference_prior_still_covers_every_padded_atom():
    """The narrowing is training-only; `resolved_mask` does not exist at inference."""
    module = KFoldECSI(KFoldECSI.Config(), None)
    f_input = make_folding_input()

    x_T = module.sample_prior(f_input, num_samples=2)

    # Atoms hidden during training are live here, and only padding is at the origin.
    assert x_T[:, :, UNRESOLVED].abs().max() > 1.0
    assert torch.allclose(x_T[:, :, PADDING], torch.zeros(1), atol=1e-6)


def test_narrowed_mask_reaches_the_score_model():
    """`training_step` hands the coordinate stack the narrowed mask, not `pad_mask`."""
    seen = {}

    def train_step(f_input, r_noisy, c_noise, s_inputs, z, atom_mask=None):
        seen["atom_mask"] = atom_mask
        return torch.zeros(*r_noisy.shape[:-1], 3)

    module = KFoldECSI(KFoldECSI.Config(), SimpleNamespace(train_step=train_step))
    f_input = make_folding_input()

    module.training_step(
        f_input,
        s_inputs=torch.zeros((1, 1, 8)),
        z=torch.zeros((1, 1, 1, 8)),
        diffusion_batch_size=2,
    )

    got = seen["atom_mask"]
    assert got is not None, "atom_mask was not forwarded"
    assert torch.equal(got, f_input.atom.pad_mask & f_input.atom.resolved_mask)
    assert not torch.equal(got, f_input.atom.pad_mask)
