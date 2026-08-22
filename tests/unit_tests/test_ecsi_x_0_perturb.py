from types import MethodType, SimpleNamespace

import pytest
import torch

from kfold.model.modules.structure.ecsi import KFoldECSI
from kfold.training.training_module import KFoldTrainingModule


def make_module(**overrides: object) -> KFoldECSI:
    values: dict[str, object] = {
        "train_x_0_perturb_time_min": 0.4,
        "train_x_0_perturb_time_max": 0.8,
        "train_x_0_perturb_prob": 1.0,
        "train_x_0_perturb_rotation_deg": 8.0,
        "train_x_0_perturb_translation_distance": 1.2,
    }
    values.update(overrides)
    return KFoldECSI(KFoldECSI.Config(**values), score_model=object())  # type: ignore[arg-type]


def make_endpoint_input() -> SimpleNamespace:
    return SimpleNamespace(
        chain=SimpleNamespace(
            asym_id=torch.tensor([[10, 20]]),
            pad_mask=torch.tensor([[True, True]]),
        ),
        token=SimpleNamespace(asym_id=torch.tensor([[10, 20]])),
        atom=SimpleNamespace(token_index=torch.tensor([[0, 0, 0, 1, 1, 1]])),
    )


def make_x_0() -> tuple[torch.Tensor, torch.Tensor]:
    x_0 = torch.tensor(
        [
            [
                [
                    [-2.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 0.0, 1.0],
                ],
                [
                    [-2.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 0.0, 1.0],
                ],
            ]
        ]
    )
    return x_0, torch.ones((1, 1, 6), dtype=torch.bool)


def test_disabled_selection_does_not_consume_rng() -> None:
    module = make_module(train_x_0_perturb_prob=0.0)
    t = torch.tensor([[0.5, 0.6]])
    chain_count = torch.tensor([2])
    torch.manual_seed(3)
    module._sample_x_0_perturb_masks(t=t, resolved_chain_count=chain_count)
    observed = torch.rand(4)
    torch.manual_seed(3)
    expected = torch.rand(4)
    torch.testing.assert_close(observed, expected)


def test_time_bounds_are_lower_closed_and_upper_open() -> None:
    module = make_module()
    t = torch.tensor([[0.3999, 0.4, 0.7999, 0.8]])
    masks = module._sample_x_0_perturb_masks(t=t, resolved_chain_count=torch.tensor([2]))
    assert masks["applied"].tolist() == [[False, True, True, False]]


def test_rotation_and_translation_have_fixed_magnitudes() -> None:
    module = make_module()
    torch.manual_seed(5)
    rotations = module._fixed_axis_angle_rotations(
        (128,), angle_degrees=8.0, dtype=torch.float64, device=torch.device("cpu")
    )
    traces = rotations.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    angles = torch.acos(((traces - 1.0) / 2.0).clamp(-1.0, 1.0))
    torch.testing.assert_close(
        angles,
        torch.full_like(angles, torch.deg2rad(torch.tensor(8.0)).item()),
        rtol=1e-10,
        atol=2e-9,
    )
    translations = module._fixed_direction_translations(
        (128,), distance=1.2, dtype=torch.float64, device=torch.device("cpu")
    )
    torch.testing.assert_close(
        translations.norm(dim=-1), torch.full((128,), 1.2, dtype=torch.float64)
    )


def test_chain_internal_distances_and_clean_frame_are_preserved() -> None:
    module = make_module()
    x_0, mask = make_x_0()
    torch.manual_seed(7)
    out = module._perturb_x_0(
        x_0=x_0,
        t=torch.tensor([[0.5, 0.9]]),
        f_input=make_endpoint_input(),
        x_0_mask=mask,
    )
    perturbed = out["x_0_bridge"]
    assert not torch.allclose(perturbed[:, 0], x_0[:, 0])
    assert torch.equal(perturbed[:, 1], x_0[:, 1])
    for chain_atoms in (slice(0, 3), slice(3, 6)):
        expected = torch.cdist(x_0[0, 0, chain_atoms], x_0[0, 0, chain_atoms])
        actual = torch.cdist(perturbed[0, 0, chain_atoms], perturbed[0, 0, chain_atoms])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        perturbed[0, 0].mean(dim=0), x_0[0, 0].mean(dim=0), atol=1e-5, rtol=0
    )


def test_monomer_and_zero_strength_are_exact_noops() -> None:
    x_0, mask = make_x_0()
    monomer_input = make_endpoint_input()
    monomer_input.chain.pad_mask = torch.tensor([[True, False]])
    monomer_input.atom.token_index = torch.tensor([[0, 0, 0, 0, 0, 0]])
    monomer_input.token.asym_id = torch.tensor([[10]])
    module = make_module()
    monomer = module._perturb_x_0(
        x_0=x_0,
        t=torch.tensor([[0.5, 0.6]]),
        f_input=monomer_input,
        x_0_mask=mask,
    )
    assert torch.equal(monomer["x_0_bridge"], x_0)
    assert not monomer["applied_mask"].any()

    zero_module = make_module(
        train_x_0_perturb_rotation_deg=0.0,
        train_x_0_perturb_translation_distance=0.0,
    )
    zero = zero_module._perturb_x_0(
        x_0=x_0,
        t=torch.tensor([[0.5, 0.6]]),
        f_input=make_endpoint_input(),
        x_0_mask=mask,
    )
    assert torch.equal(zero["x_0_bridge"], x_0)
    assert not zero["applied_mask"].any()


def test_bridge_delta_is_alpha_scaled_endpoint_delta() -> None:
    module = make_module(gamma_max=0.0)
    label_coords = torch.randn((1, 6, 3))
    seen: dict[str, torch.Tensor] = {}
    f_input = make_endpoint_input()
    f_input.batch_size = 1
    f_input.device = torch.device("cpu")
    f_input.atom.label_coords = label_coords
    f_input.atom.prior_coords = torch.randn((1, 6, 1, 3))
    f_input.atom.resolved_mask = torch.ones((1, 6), dtype=torch.bool)
    f_input.atom.pad_mask = torch.ones((1, 6), dtype=torch.bool)
    module.random_augmentation = lambda x, mask: x
    module.sample_noise_level = lambda shape, device: torch.tensor([[0.5, 0.7]])

    def add_offset(
        self: KFoldECSI,
        *,
        x_0: torch.Tensor,
        t: torch.Tensor,
        f_input: SimpleNamespace,
        x_0_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        seen["x_0"] = x_0.clone()
        sample_mask = torch.ones_like(t, dtype=torch.bool)
        return {
            "x_0_bridge": x_0 + 2.0,
            "time_eligible_mask": sample_mask,
            "eligible_mask": sample_mask,
            "requested_mask": sample_mask,
            "applied_mask": sample_mask,
            "x_0_rmsd": torch.full_like(t, 2.0 * 3**0.5),
            "resolved_chain_count": torch.full_like(t, 2, dtype=torch.long),
        }

    module._perturb_x_0 = MethodType(add_offset, module)  # type: ignore[method-assign]
    out = module.sample_train_input(f_input, diffusion_batch_size=2)
    clean = module._interpolate_bridge(out["x_0"], out["x_T"], out["noise"], out["t"])
    expected_delta = (1.0 - out["t"])[..., None, None] * 2.0
    torch.testing.assert_close(out["x_t"] - clean, expected_delta.expand_as(out["x_t"]))
    torch.testing.assert_close(
        out["x_0_perturb_x_t_rmsd"],
        (1.0 - out["t"]) * (2.0 * 3**0.5),
    )
    assert torch.equal(out["x_0"], seen["x_0"])


def test_logistic_schedule_requests_about_eight_percent() -> None:
    module = make_module(train_x_0_perturb_prob=0.5)
    torch.manual_seed(19)
    t = module.sample_noise_level((1, 200000), torch.device("cpu"))
    masks = module._sample_x_0_perturb_masks(t=t, resolved_chain_count=torch.tensor([2]))
    assert abs(masks["requested"].float().mean().item() - 0.0805) < 0.003


def test_telemetry_reports_actual_and_time_binned_dose() -> None:
    metrics: dict[str, torch.Tensor] = {}
    out = {
        "t": torch.tensor([[0.45, 0.55, 0.75, 0.85]]),
        "x_0_perturb_time_eligible_mask": torch.tensor([[True, True, True, False]]),
        "x_0_perturb_eligible_mask": torch.tensor([[True, True, True, False]]),
        "x_0_perturb_requested_mask": torch.tensor([[True, False, True, False]]),
        "x_0_perturb_applied_mask": torch.tensor([[True, False, True, False]]),
        "x_0_perturb_x_0_rmsd": torch.tensor([[1.0, 0.0, 3.0, 0.0]]),
        "x_0_perturb_x_t_rmsd": torch.tensor([[0.55, 0.0, 0.75, 0.0]]),
        "x_0_perturb_resolved_chain_count": torch.tensor([[2, 2, 2, 2]]),
    }
    KFoldTrainingModule._add_x_0_perturb_metrics(
        diffusion_metrics=metrics,
        diffusion_out=out,
    )
    torch.testing.assert_close(metrics["x_0_perturb_applied_fraction"], torch.tensor(0.5))
    torch.testing.assert_close(
        metrics["x_0_perturb_applied_given_eligible"], torch.tensor(2.0 / 3.0)
    )
    torch.testing.assert_close(metrics["x_0_perturb_x_t_rmsd_mean"], torch.tensor(0.65))
    torch.testing.assert_close(
        metrics["x_0_perturb_t04_05_applied_fraction"], torch.tensor(1.0)
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_x_0_perturb_prob": -0.1}, "must lie in"),
        ({"train_x_0_perturb_prob": 1.1}, "must lie in"),
        ({"train_x_0_perturb_rotation_deg": -1.0}, "non-negative"),
        ({"train_x_0_perturb_translation_distance": -1.0}, "non-negative"),
        ({"train_x_0_perturb_time_min": 0.9}, "times must lie"),
    ],
)
def test_invalid_x_0_perturbation_config_is_rejected(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_module(**overrides)
