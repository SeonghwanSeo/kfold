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
        asym_id=asym_id,
        repr_coords=coords,
        org_token_index=torch.arange(16, dtype=torch.long).view(1, 16),
    )
    chain = SimpleNamespace(
        pad_mask=torch.tensor([[True, True]], dtype=torch.bool),
        asym_id=torch.tensor([[1, 2]], dtype=torch.long),
    )
    return SimpleNamespace(
        token=token,
        chain=chain,
        is_batched=True,
    )


def _fake_single_chain_input():
    f_input = _fake_input()
    f_input.token.asym_id = torch.ones(1, 16, dtype=torch.long)
    f_input.chain.pad_mask = torch.tensor([[True, False]], dtype=torch.bool)
    f_input.chain.asym_id = torch.tensor([[1, 0]], dtype=torch.long)
    return f_input


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


def test_patch_geometry_post_pool_layers_run_once_across_chunks():
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(
            enabled=True,
            channel_z=16,
            patch_size=4,
            max_patches_per_chain=2,
            max_patch_pairs=8,
            pool_chunk_size=1,
        )
    )
    calls = 0

    def count_calls(*_):
        nonlocal calls
        calls += 1

    handle = head.transition.register_forward_hook(count_calls)
    try:
        out = head(_fake_input(), torch.randn(1, 16, 16, 16))
    finally:
        handle.remove()

    assert out["logits"].shape[0] > 1
    assert calls == 1


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
