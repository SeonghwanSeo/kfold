from types import SimpleNamespace

import torch

from kfold.model.modules.patch_geometry import PatchPairGeometryHead
from kfold.training.loss.patch_geometry import PatchPairGeometryLoss


def _fake_input() -> SimpleNamespace:
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
    return SimpleNamespace(token=token, atom=atom, chain=chain, is_batched=True)


def _fake_single_chain_input() -> SimpleNamespace:
    f_input = _fake_input()
    f_input.token.entity_id = torch.ones(1, 16, dtype=torch.long)
    f_input.token.asym_id = torch.ones(1, 16, dtype=torch.long)
    f_input.chain.pad_mask = torch.tensor([[True, False]], dtype=torch.bool)
    f_input.chain.asym_id = torch.tensor([[1, 0]], dtype=torch.long)
    return f_input


def _fake_repeated_entity_input() -> SimpleNamespace:
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
        atom_index=torch.tensor([list(range(4)) * 3], dtype=torch.long)
    )
    return SimpleNamespace(token=token, atom=atom, is_batched=True)


def _active_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def test_patch_geometry_head_freezes_when_disabled() -> None:
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(enabled=False, channel_z=16),
    )

    assert not any(p.requires_grad for p in head.parameters())
    assert head(_fake_input(), torch.randn(1, 16, 16, 16)) == {}


def test_patch_geometry_head_and_loss_use_uniform_pair_supervision() -> None:
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(
            enabled=True,
            channel_z=16,
            patch_size=4,
            max_patches_per_chain=2,
            max_patch_pairs=8,
        ),
    )
    out = head(_fake_input(), torch.randn(1, 16, 16, 16))
    loss, metrics = PatchPairGeometryLoss()(out)

    assert set(out) == {"logits", "target"}
    assert out["logits"].shape[0] == 4
    assert torch.isfinite(loss)
    assert metrics["patch_geometry_valid_pairs"] == 4

    loss.backward()
    assert all(p.grad is not None for p in _active_params(head))


def test_uniform_supervision_keeps_all_repeated_entity_pairs() -> None:
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(
            enabled=True,
            channel_z=16,
            patch_size=4,
            max_patches_per_chain=1,
            max_patch_pairs=8,
        ),
    )
    out = head(_fake_repeated_entity_input(), torch.randn(1, 12, 12, 16))

    # Three chains yield three inter-chain patch pairs. No copy-ambiguous pair
    # is removed because this supervision is intentionally symmetry-agnostic.
    assert out["logits"].shape == (3, head.num_bins)


def test_patch_geometry_head_touches_active_params_without_patch_pairs() -> None:
    head = PatchPairGeometryHead(
        PatchPairGeometryHead.Config(
            enabled=True,
            channel_z=16,
            patch_size=4,
            max_patches_per_chain=2,
            max_patch_pairs=8,
        ),
    )
    out = head(_fake_single_chain_input(), torch.randn(1, 16, 16, 16))
    loss, metrics = PatchPairGeometryLoss()(out)
    loss.backward()

    assert out["logits"].shape[0] == 0
    assert torch.isfinite(loss)
    assert metrics["patch_geometry_valid_pairs"] == 0
    assert all(p.grad is not None for p in _active_params(head))


def test_patch_geometry_loss_is_unweighted_mean_cross_entropy() -> None:
    logits = torch.tensor(
        [[2.0, -1.0, 0.5], [-0.5, 0.0, 1.5]],
        requires_grad=True,
    )
    target = torch.tensor([0, 2])
    loss, metrics = PatchPairGeometryLoss(
        min_dist=2.0,
        max_dist=5.0,
        num_bins=3,
        near_cutoff=3.0,
    )({"logits": logits, "target": target})

    expected = torch.nn.functional.cross_entropy(logits, target)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["patch_geometry_ce_loss"], expected.detach())

    loss.backward()
    assert logits.grad is not None
