from types import SimpleNamespace

import torch

from kfold.model.modules.patch_geometry import PatchPairGeometryHead
from kfold.training.loss.patch_geometry import PatchPairGeometryLoss


def _fake_input():
    asym_id = torch.tensor([[1] * 8 + [2] * 8], dtype=torch.long)
    coords = torch.zeros(1, 16, 3)
    coords[0, :8, 0] = torch.arange(8, dtype=torch.float32)
    coords[0, 8:, 0] = 50.0 + torch.arange(8, dtype=torch.float32)
    token = SimpleNamespace(
        pad_mask=torch.ones(1, 16, dtype=torch.bool),
        repr_mask=torch.ones(1, 16, dtype=torch.bool),
        entity_id=asym_id.clone(),
        asym_id=asym_id,
        repr_index=torch.arange(16, dtype=torch.long).view(1, 16),
        repr_coords=coords,
        org_token_index=torch.arange(16, dtype=torch.long).view(1, 16),
    )
    atom = SimpleNamespace(
        atom_index=torch.tensor(
            [list(range(8)) + list(range(8))],
            dtype=torch.long,
        ),
    )
    chain = SimpleNamespace(
        pad_mask=torch.tensor([[True, True]], dtype=torch.bool),
        asym_id=torch.tensor([[1, 2]], dtype=torch.long),
    )
    return SimpleNamespace(
        token=token,
        atom=atom,
        chain=chain,
        is_batched=True,
    )


def _fake_single_chain_input():
    f_input = _fake_input()
    f_input.token.entity_id = torch.ones(1, 16, dtype=torch.long)
    f_input.token.asym_id = torch.ones(1, 16, dtype=torch.long)
    f_input.chain.pad_mask = torch.tensor([[True, False]], dtype=torch.bool)
    f_input.chain.asym_id = torch.tensor([[1, 0]], dtype=torch.long)
    return f_input


def _fake_repeated_entity_input():
    asym_id = torch.tensor([[1] * 4 + [2] * 4 + [3] * 4], dtype=torch.long)
    entity_id = torch.tensor([[1] * 4 + [2] * 8], dtype=torch.long)
    coords = torch.zeros(1, 12, 3)
    coords[0, 4:8, 0] = 5.0
    coords[0, 8:, 0] = 60.0
    token = SimpleNamespace(
        pad_mask=torch.ones(1, 12, dtype=torch.bool),
        repr_mask=torch.ones(1, 12, dtype=torch.bool),
        entity_id=entity_id,
        asym_id=asym_id,
        repr_index=torch.arange(12, dtype=torch.long).view(1, 12),
        repr_coords=coords,
        org_token_index=torch.arange(12, dtype=torch.long).view(1, 12),
    )
    atom = SimpleNamespace(
        atom_index=torch.tensor(
            [list(range(4)) * 3],
            dtype=torch.long,
        ),
    )
    return SimpleNamespace(token=token, atom=atom, is_batched=True)


def _fake_patch_level_symmetry_input():
    asym_id = torch.tensor([[1] * 4 + [2] * 4 + [3] * 4], dtype=torch.long)
    entity_id = torch.tensor([[1] * 4 + [2] * 8], dtype=torch.long)
    coords = torch.tensor(
        [
            [0.0, 0.1, 100.0, 100.1],
            [5.0, 5.1, 150.0, 150.1],
            [60.0, 60.1, 200.0, 200.1],
        ],
        dtype=torch.float32,
    ).reshape(1, 12, 1)
    coords = torch.nn.functional.pad(coords, (0, 2))
    token = SimpleNamespace(
        pad_mask=torch.ones(1, 12, dtype=torch.bool),
        repr_mask=torch.ones(1, 12, dtype=torch.bool),
        entity_id=entity_id,
        asym_id=asym_id,
        repr_index=torch.arange(12, dtype=torch.long).view(1, 12),
        repr_coords=coords,
        org_token_index=torch.arange(12, dtype=torch.long).view(1, 12),
    )
    atom = SimpleNamespace(
        atom_index=torch.tensor(
            [list(range(4)) * 3],
            dtype=torch.long,
        ),
    )
    return SimpleNamespace(token=token, atom=atom, is_batched=True)


def _active_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def test_patch_geometry_head_freezes_when_disabled():
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(enabled=False, channel_z=16),
    )

    assert not any(p.requires_grad for p in head.parameters())


def test_patch_geometry_head_and_loss_on_spatial_hard_negatives():
    cfg = PatchPairGeometryHead.Config(
        enabled=True,
        channel_z=16,
        patch_size=4,
        max_patches_per_chain=2,
        max_patch_pairs=8,
    )
    head = PatchPairGeometryHead(cfg)
    f_input = _fake_input()
    z = torch.randn(1, 16, 16, 16)

    out = head(f_input, z)
    loss, metrics = PatchPairGeometryLoss()(out)

    assert out["logits"].shape[0] > 0
    assert out["hard_negative"].sum() > 0
    assert torch.all(out["weight"] == 0.5)
    assert "valid_mask" not in out
    assert "timing" not in out
    assert torch.isfinite(loss)
    assert metrics["patch_geometry_valid_pairs"] > 0

    loss.backward()
    assert all(p.grad is not None for p in _active_params(head))


def test_patch_geometry_head_touches_active_params_without_patch_pairs():
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(
            enabled=True,
            channel_z=16,
            patch_size=4,
            max_patches_per_chain=2,
            max_patch_pairs=8,
        ),
    )
    f_input = _fake_single_chain_input()
    z = torch.randn(1, 16, 16, 16, requires_grad=True)

    out = head(f_input, z)
    loss, metrics = PatchPairGeometryLoss()(out)
    loss.backward()

    assert out["logits"].shape[0] == 0
    assert torch.isfinite(loss)
    assert metrics["patch_geometry_valid_pairs"] == 0
    assert all(p.grad is not None for p in _active_params(head))


def test_patch_geometry_head_masks_noncontacting_equivalent_chain_pair():
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(
            enabled=True,
            channel_z=16,
            patch_size=4,
            max_patches_per_chain=1,
            max_patch_pairs=8,
        ),
    )
    f_input = _fake_repeated_entity_input()
    z = torch.randn(1, 12, 12, 16)

    out = head(f_input, z)

    # A1-B1 is in contact, so the noncontacting A1-B2 pair is ambiguous under
    # exchange of the two B copies and is excluded. B1-B2 remains a valid
    # homomeric hard negative because no equivalent B-B chain pair is positive.
    assert out["logits"].shape == (2, head.num_bins)
    assert out["hard_negative"].sum() == 1


def test_patch_geometry_head_keeps_homomer_negative_without_equivalent_contact():
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(
            enabled=True,
            channel_z=16,
            patch_size=4,
            max_patches_per_chain=2,
            max_patch_pairs=8,
        ),
    )
    f_input = _fake_input()
    f_input.token.entity_id = torch.ones(1, 16, dtype=torch.long)
    z = torch.randn(1, 16, 16, 16)

    out = head(f_input, z)

    assert out["logits"].shape == (4, head.num_bins)
    assert out["hard_negative"].sum() == 4


def test_patch_geometry_head_masks_only_symmetry_equivalent_patch_pair():
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(
            enabled=True,
            channel_z=16,
            patch_size=2,
            max_patches_per_chain=2,
            max_patch_pairs=32,
        ),
    )
    f_input = _fake_patch_level_symmetry_input()
    z = torch.randn(1, 12, 12, 16)

    out = head(f_input, z)

    # Six patches form twelve inter-chain patch pairs. A1-p0 contacts B1-p0,
    # so only the symmetry-equivalent A1-p0/B2-p0 negative is removed. The
    # other three A1/B2 patch pairs remain valid hard negatives.
    assert out["logits"].shape == (11, head.num_bins)
    assert out["hard_negative"].sum() == 10
