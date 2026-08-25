from types import MethodType, SimpleNamespace

import pytest
import torch

import kfold.model.modules.structure.ecsi as ecsi_module
from kfold.model.modules.structure.ecsi import KFoldECSI
from kfold.training.training_module import KFoldTrainingModule


def make_module(**overrides: object) -> KFoldECSI:
    values: dict[str, object] = {
        "train_x_0_perturb_time_min": 0.7,
        "train_x_0_perturb_prob": 1.0,
        "train_x_0_perturb_translation_std": 4.0,
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


def make_endpoints() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    x_T = x_0 + torch.tensor([5.0, -3.0, 2.0])
    return x_0, x_T, torch.ones((1, 1, 6), dtype=torch.bool)


def test_disabled_selection_does_not_consume_rng() -> None:
    module = make_module(train_x_0_perturb_prob=0.0)
    t = torch.tensor([[0.8, 0.9]])
    chain_count = torch.tensor([2])
    torch.manual_seed(3)
    module._sample_x_0_perturb_masks(t=t, resolved_chain_count=chain_count)
    observed = torch.rand(4)
    torch.manual_seed(3)
    expected = torch.rand(4)
    torch.testing.assert_close(observed, expected)


def test_time_threshold_is_strictly_above_point_seven() -> None:
    module = make_module()
    t = torch.tensor([[0.6999, 0.7, 0.7001, 0.9999]])
    masks = module._sample_x_0_perturb_masks(t=t, resolved_chain_count=torch.tensor([2]))
    assert masks["applied"].tolist() == [[False, False, True, True]]


def test_chain_internal_distances_and_legacy_centered_frame_are_preserved() -> None:
    module = make_module()
    x_0, x_T, mask = make_endpoints()
    clean_x_0 = x_0.clone()
    clean_x_T = x_T.clone()
    torch.manual_seed(7)
    out = module._perturb_x_0(
        x_0=x_0,
        x_T=x_T,
        t=torch.tensor([[0.8, 0.5]]),
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
        perturbed[0, 0].mean(dim=0), torch.zeros(3), atol=1e-5, rtol=0
    )
    assert torch.equal(x_0, clean_x_0)
    assert torch.equal(x_T, clean_x_T)


def test_legacy_random_so3_and_axiswise_gaussian_translation_are_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = make_module()
    x_0, x_T, mask = make_endpoints()
    captured: dict[str, torch.Tensor | tuple[int, ...]] = {}

    def identity_rotations(
        shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        captured["rotation_shape"] = shape
        return torch.eye(3, dtype=dtype, device=device).expand(*shape, 3, 3)

    def unit_gaussian(values: torch.Tensor) -> torch.Tensor:
        translations = torch.ones_like(values)
        captured["translations"] = translations
        return translations

    monkeypatch.setattr(ecsi_module, "random_rotations_torch", identity_rotations)
    monkeypatch.setattr(torch, "randn_like", unit_gaussian)
    module._perturb_x_0(
        x_0=x_0,
        x_T=x_T,
        t=torch.tensor([[0.8, 0.9]]),
        f_input=make_endpoint_input(),
        x_0_mask=mask,
    )

    assert captured["rotation_shape"] == (1, 2, 2)
    torch.testing.assert_close(
        captured["translations"],  # type: ignore[arg-type]
        torch.full((1, 2, 2, 3), 4.0),
    )


def test_monomer_is_an_exact_noop() -> None:
    x_0, x_T, mask = make_endpoints()
    monomer_input = make_endpoint_input()
    monomer_input.chain.pad_mask = torch.tensor([[True, False]])
    monomer_input.atom.token_index = torch.tensor([[0, 0, 0, 0, 0, 0]])
    monomer_input.token.asym_id = torch.tensor([[10]])
    module = make_module()
    monomer = module._perturb_x_0(
        x_0=x_0,
        x_T=x_T,
        t=torch.tensor([[0.8, 0.9]]),
        f_input=monomer_input,
        x_0_mask=mask,
    )
    assert torch.equal(monomer["x_0_bridge"], x_0)
    assert not monomer["applied_mask"].any()


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
        x_T: torch.Tensor,
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


def test_logistic_schedule_requests_about_two_percent() -> None:
    module = make_module(train_x_0_perturb_prob=0.2)
    torch.manual_seed(19)
    t = module.sample_noise_level((1, 200000), torch.device("cpu"))
    masks = module._sample_x_0_perturb_masks(t=t, resolved_chain_count=torch.tensor([2]))
    assert abs(masks["requested"].float().mean().item() - 0.0183) < 0.002


def test_telemetry_reports_actual_and_time_binned_dose() -> None:
    metrics: dict[str, torch.Tensor] = {}
    out = {
        "t": torch.tensor([[0.65, 0.75, 0.85, 0.95]]),
        "x_0_perturb_time_eligible_mask": torch.tensor([[False, True, True, True]]),
        "x_0_perturb_eligible_mask": torch.tensor([[False, True, True, True]]),
        "x_0_perturb_requested_mask": torch.tensor([[False, True, False, True]]),
        "x_0_perturb_applied_mask": torch.tensor([[False, True, False, True]]),
        "x_0_perturb_x_0_rmsd": torch.tensor([[0.0, 1.0, 0.0, 3.0]]),
        "x_0_perturb_x_t_rmsd": torch.tensor([[0.0, 0.55, 0.0, 0.75]]),
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
        metrics["x_0_perturb_t07_08_applied_fraction"], torch.tensor(1.0)
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_x_0_perturb_prob": -0.1}, "must lie in"),
        ({"train_x_0_perturb_prob": 1.1}, "must lie in"),
        ({"train_x_0_perturb_translation_std": -1.0}, "non-negative"),
        ({"train_x_0_perturb_time_min": 1.0}, "time support"),
    ],
)
def test_invalid_x_0_perturbation_config_is_rejected(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_module(**overrides)
