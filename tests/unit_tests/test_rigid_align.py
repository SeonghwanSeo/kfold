import numpy as np
import pytest
import torch

from kfold.data.utils.io.structure import read_protein_structure
from kfold.utils.geometry.rigid_align import (
    compute_rmsd,
    rigid_align,
    weighted_rigid_align,
)

EXAMPLE_PDB = "./examples/apo/H1106-protein-A.pdb"

torch.set_float32_matmul_precision("highest")


@pytest.fixture
def p() -> np.ndarray:
    """Load example protein structure and return its coordinates as a flat array."""
    _, coords = read_protein_structure(EXAMPLE_PDB)  # [L, 37, 3]
    return coords.reshape(-1, 3)  # [Natom, 3]


@pytest.fixture
def mask(p: np.ndarray) -> np.ndarray:
    """Create a mask that marks all finite coordinates as valid."""
    return np.isfinite(p).all(axis=-1)  # [Natom]


@pytest.fixture
def rot_matrix(p: np.ndarray) -> np.ndarray:
    """Create 45 degree rotation matrix around Z axis."""
    angle = np.pi / 4
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return rot_matrix


def to_torch(*arrays: np.ndarray) -> list[torch.Tensor]:
    return [torch.from_numpy(arr) for arr in arrays]


# ==========================================
# 2. Test Cases
# ==========================================


def test_rmsd_identity(p: np.ndarray, mask: np.ndarray):
    # NumPy
    rmsd_np = compute_rmsd(p, p, mask, align=False)
    assert rmsd_np == pytest.approx(0.0, abs=1e-5)
    rmsd_np = compute_rmsd(p, p, mask, align=True)
    assert rmsd_np == pytest.approx(0.0, abs=1e-5)
    rmsd_np = compute_rmsd(p, p, mask, align=False, no_svd=True)
    assert rmsd_np == pytest.approx(0.0, abs=1e-5)
    rmsd_np = compute_rmsd(p, p, mask, align=True, no_svd=True)
    assert rmsd_np == pytest.approx(0.0, abs=1e-5)
    p_aligned = rigid_align(p, p, mask)
    assert np.abs((p_aligned - p)[mask]).mean() == pytest.approx(0.0, abs=1e-5)

    # Torch
    p_t, mask_t = to_torch(p, mask)
    rmsd_th = compute_rmsd(p_t, p_t, mask=mask_t, align=False)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)
    rmsd_th = compute_rmsd(p_t, p_t, mask=mask_t, align=True)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)
    rmsd_th = compute_rmsd(p_t, p_t, mask=mask_t, align=False, no_svd=True)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)
    rmsd_th = compute_rmsd(p_t, p_t, mask=mask_t, align=True, no_svd=True)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)
    p_t_aligned = rigid_align(p_t, p_t, mask=mask_t)
    assert (p_t_aligned - p_t)[mask].abs().mean().item() == pytest.approx(0.0, abs=1e-5)


def test_rmsd_translation(p: np.ndarray, mask: np.ndarray):
    q = p + np.array([5.0, -3.0, 2.0], dtype=np.float32)
    expected = np.linalg.norm(np.array([5.0, -3.0, 2.0], dtype=np.float32))

    # NumPy
    rmsd_np = compute_rmsd(p, q, mask, align=False)
    assert rmsd_np == pytest.approx(expected, abs=1e-5)
    rmsd_np = compute_rmsd(p, q, mask, align=True)
    assert rmsd_np == pytest.approx(0.0, abs=1e-5)
    rmsd_np = compute_rmsd(p, q, mask, align=False, no_svd=True)
    assert rmsd_np == pytest.approx(expected, abs=1e-5)
    rmsd_np = compute_rmsd(p, q, mask, align=True, no_svd=True)
    assert rmsd_np == pytest.approx(0.0, abs=1e-5)
    q_aligned = rigid_align(q, p, mask)
    assert np.abs((q_aligned - p)[mask]).mean() == pytest.approx(0.0, abs=1e-5)

    # Torch
    p_t, q_t, mask_t = to_torch(p, q, mask)
    rmsd_th = compute_rmsd(p_t, q_t, mask=mask_t, align=False)
    assert rmsd_th.item() == pytest.approx(expected, abs=1e-5)
    rmsd_th = compute_rmsd(p_t, q_t, mask=mask_t, align=True)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)
    rmsd_th = compute_rmsd(p_t, q_t, mask=mask_t, align=False, no_svd=True)
    assert rmsd_th.item() == pytest.approx(expected, abs=1e-5)
    rmsd_th = compute_rmsd(p_t, q_t, mask=mask_t, align=True, no_svd=True)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)
    q_t_aligned = rigid_align(q_t, p_t, mask=mask_t)
    assert (q_t_aligned - p_t)[mask_t].abs().mean().item() == pytest.approx(0.0, abs=1e-5)


def test_rmsd_rotation_with_align(
    p: np.ndarray, mask: np.ndarray, rot_matrix: np.ndarray
):
    # Apply rotation
    q = p @ rot_matrix.T

    # NumPy
    rmsd_np = compute_rmsd(p, q, mask, align=True)
    assert rmsd_np == pytest.approx(0.0, abs=1e-5)
    rmsd_np = compute_rmsd(p, q, mask, align=True, no_svd=True)
    assert rmsd_np == pytest.approx(0.0, abs=1e-5)
    q_aligned = rigid_align(q, p, mask)
    assert np.abs((q_aligned - p)[mask]).mean() == pytest.approx(0.0, abs=1e-5)

    # Torch
    p_t, q_t, mask_t = to_torch(p, q, mask)
    rmsd_th = compute_rmsd(p_t, q_t, mask=mask_t, align=True)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)
    rmsd_th = compute_rmsd(p_t, q_t, mask=mask_t, align=True, no_svd=True)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)
    q_t_aligned = rigid_align(q_t, p_t, mask=mask_t)
    assert (q_t_aligned - p_t)[mask_t].abs().mean().item() == pytest.approx(0.0, abs=1e-5)


def test_masking_logic(p: np.ndarray, mask: np.ndarray, rot_matrix: np.ndarray):
    q, mask = p.copy(), mask.copy()

    # Create a "bad" target by adding large noise to masked positions
    q[0] += np.array([100.0, 100.0, 100.0], dtype=np.float32)
    mask[0] = False  # Invalidate the first point

    # Translate and rotate
    q = q @ rot_matrix.T + np.array([10.0, -5.0, 3.0], dtype=q.dtype)

    # NumPy
    rmsd_np = compute_rmsd(q, p, mask, align=True)
    assert rmsd_np == pytest.approx(0.0, abs=1e-4)
    rmsd_np = compute_rmsd(q, p, mask, align=True, no_svd=True)
    assert rmsd_np == pytest.approx(0.0, abs=1e-5)

    # Torch
    p_t, q_t, mask_t = to_torch(p, q, mask)
    rmsd_th = compute_rmsd(q_t, p_t, mask=mask_t, align=True)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)
    rmsd_th = compute_rmsd(q_t, p_t, mask=mask_t, align=True, no_svd=True)
    assert rmsd_th.item() == pytest.approx(0.0, abs=1e-5)


def test_batch_processing(p: np.ndarray, mask: np.ndarray):
    offset = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    p_batch = np.stack([p, p - offset, p + offset])
    q_batch = np.stack([p, p, p])
    mask_batch = np.stack([mask, mask, mask])

    rmsd_aligned = compute_rmsd(p_batch, q_batch, mask_batch, align=True)
    assert np.allclose(rmsd_aligned, 0.0, atol=1e-5)


def test_weighted_alignment():
    coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 10, 0]], dtype=np.float32)
    target = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [2, 10, 0]], dtype=np.float32)

    weights = np.array([1, 1, 1, 0], dtype=np.float32)

    # NumPy
    aligned_np = weighted_rigid_align(coords, target, weights, None)
    assert np.allclose(aligned_np[:3], target[:3], atol=1e-5)

    # Torch
    t_c, t_t, t_w = to_torch(coords, target, weights)
    aligned_th = weighted_rigid_align(t_c, t_t, t_w, None)
    assert torch.allclose(aligned_th[:3], t_t[:3], atol=1e-5)


def test_reflection_handling(p: np.ndarray, mask: np.ndarray):
    """Check that reflection cases are handled correctly."""
    p_mirror = p.copy()
    p_mirror[:, 0] *= -1  # X-axis reflection

    # NumPy
    rmsd_val = compute_rmsd(p, p_mirror, mask, align=True)
    assert rmsd_val > 0.1

    # Torch
    t_sq, t_mir, t_mask = to_torch(p, p_mirror, mask)
    rmsd_th = compute_rmsd(t_sq, t_mir, mask=t_mask, align=True)
    assert rmsd_th.item() == pytest.approx(rmsd_val, abs=1e-5)


def test_anchor_index_alignment():
    coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    target = coords + 1.0

    # Use the first point as anchor
    anchor_idx = np.array([0])

    aligned = rigid_align(coords, target, None, anchor_index=anchor_idx)
    assert np.allclose(aligned[0], target[0], atol=1e-6)
    assert np.allclose(aligned, target, atol=1e-6)
