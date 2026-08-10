"""Targeted correctness tests for the minimal ECSI churn sampler."""

import math

import pytest
import torch

from kfold.model.modules.structure.ecsi import KFoldECSI

TIME_MAX = 0.9999
CHURN_END_TIME = 0.5


def make_sampler(**overrides: object) -> KFoldECSI:
    """Build a sampler without requiring a score model for sampler-only tests."""
    config_values: dict[str, object] = {
        "time_max": TIME_MAX,
        "churn_end_time": CHURN_END_TIME,
        "churn_max_time": None,
    }
    config_values.update(overrides)
    config = KFoldECSI.Config(**config_values)
    return KFoldECSI(config, score_model=object())  # type: ignore[arg-type]


def make_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a float64 CPU state with one masked atom per sample."""
    generator = torch.Generator().manual_seed(17)
    shape = (1, 2, 5, 3)
    x_t = torch.randn(shape, generator=generator, dtype=torch.float64)
    x_T = torch.randn(shape, generator=generator, dtype=torch.float64)
    mask = torch.ones(shape[:-1], dtype=torch.bool)
    mask[..., -1] = False
    return x_t, x_T, mask


def test_global_sampler_defaults_are_the_fixed_sde_hybrid_bundle() -> None:
    config = KFoldECSI.Config()

    assert config.sampler_mode == "sde"
    assert config.sampler_ode_type == "ecsi"
    assert config.sampler_switch_gamma == 3.6
    assert config.sampler_after_switch_mode == "ode"
    assert config.sampler_after_switch_ode_type == "si"
    assert config.churn_factor == 0.1
    assert config.gamma_power == 1.0
    assert config.churn_end_time == CHURN_END_TIME
    assert config.churn_max_time is None
    assert not hasattr(config, "churn_space")
    assert not hasattr(config, "svgd_step")


def test_sigma_matched_churn_hits_requested_noise_inflation() -> None:
    sampler = make_sampler(gamma_power=2.0)
    x_t, x_T, mask = make_state()
    chi = 0.1
    t = 0.7

    _, t_hat = sampler._apply_sigma_matched_churn(x_t, x_T, mask, t, chi=chi)

    expected_sigma = (1.0 + chi) * float(sampler.coeff.sigma_eff(t))
    actual_sigma = float(sampler.coeff.sigma_eff(t_hat))
    assert math.isclose(actual_sigma, expected_sigma, rel_tol=1e-10)


def test_sigma_matched_churn_uses_the_default_bridge_closed_form() -> None:
    sampler = make_sampler()
    x_t, x_T, mask = make_state()
    chi = 0.1
    t = 0.7

    _, t_hat = sampler._apply_sigma_matched_churn(x_t, x_T, mask, t, chi=chi)

    expected_odds = (1.0 + chi) ** 2 * t / (1.0 - t)
    expected_time = expected_odds / (1.0 + expected_odds)
    assert t_hat == pytest.approx(expected_time)


def test_gamma_power_must_be_positive() -> None:
    with pytest.raises(ValueError, match="gamma_power must be positive"):
        make_sampler(gamma_power=0.0)


def test_sigma_matched_churn_uses_the_bridge_conditional() -> None:
    sampler = make_sampler()
    x_t, x_T, mask = make_state()
    chi = 0.1
    t = 0.7

    torch.manual_seed(23)
    expected_noise = torch.randn_like(x_t).masked_fill_(~mask[..., None], 0.0)
    torch.manual_seed(23)
    observed, t_hat = sampler._apply_sigma_matched_churn(x_t, x_T, mask, t, chi=chi)

    coeff = sampler.coeff
    alpha_ratio = float(coeff.alpha(t_hat)) / float(coeff.alpha(t))
    variance = (
        float(coeff.gamma(t_hat)) ** 2 - alpha_ratio**2 * float(coeff.gamma(t)) ** 2
    )
    expected = (
        alpha_ratio * x_t
        + (float(coeff.beta(t_hat)) - alpha_ratio * float(coeff.beta(t))) * x_T
        + math.sqrt(variance) * expected_noise
    )
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_churn_is_a_no_op_for_zero_chi_and_outside_its_window() -> None:
    sampler = make_sampler()
    x_t, x_T, mask = make_state()

    x_zero, t_zero = sampler._apply_sigma_matched_churn(x_t, x_T, mask, 0.7, chi=0.0)
    assert t_zero == 0.7
    torch.testing.assert_close(x_zero, x_t, rtol=0.0, atol=0.0)

    x_end, t_end = sampler._apply_forward_pinned_churn(
        x_t,
        x_T,
        mask,
        CHURN_END_TIME,
    )
    assert t_end == CHURN_END_TIME
    torch.testing.assert_close(x_end, x_t, rtol=0.0, atol=0.0)


def test_initial_sampling_time_is_excluded_from_churn_without_consuming_rng() -> None:
    sampler = make_sampler(time_max=0.8)
    x_t, x_T, mask = make_state()
    initial_time = sampler.get_sampling_schedule(num_steps=5)[0]

    assert initial_time == pytest.approx(sampler.time_max)
    torch.manual_seed(29)
    expected_next_noise = torch.randn_like(x_t)
    torch.manual_seed(29)
    x_initial, churn_time = sampler._apply_forward_pinned_churn(
        x_t,
        x_T,
        mask,
        initial_time,
    )
    actual_next_noise = torch.randn_like(x_t)

    assert churn_time == initial_time
    torch.testing.assert_close(x_initial, x_t, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_next_noise,
        expected_next_noise,
        rtol=0.0,
        atol=0.0,
    )


def test_sigma_matched_churn_clamps_to_trained_time_support() -> None:
    sampler = make_sampler(time_max=0.8)
    x_t, x_T, mask = make_state()

    _, t_hat = sampler._apply_sigma_matched_churn(x_t, x_T, mask, 0.75, chi=0.5)
    assert t_hat == pytest.approx(0.8)

    x_high, t_high = sampler._apply_sigma_matched_churn(x_t, x_T, mask, 0.8, chi=0.5)
    assert t_high == 0.8
    torch.testing.assert_close(x_high, x_t, rtol=0.0, atol=0.0)


def test_sde_to_ode_rollback_keeps_the_same_churn_move() -> None:
    sde_sampler = make_sampler()
    ode_sampler = make_sampler(sampler_mode="ode")
    x_t, x_T, mask = make_state()

    assert sde_sampler._select_update_method(0.9) == ("sde", "ecsi")
    assert sde_sampler._select_update_method(0.01) == ("ode", "si")
    assert ode_sampler._select_update_method(0.9) == ("ode", "ecsi")

    torch.manual_seed(41)
    sde_out, sde_time = sde_sampler._apply_forward_pinned_churn(x_t, x_T, mask, 0.7)
    torch.manual_seed(41)
    ode_out, ode_time = ode_sampler._apply_forward_pinned_churn(x_t, x_T, mask, 0.7)

    assert ode_time == sde_time
    torch.testing.assert_close(ode_out, sde_out, rtol=0.0, atol=0.0)
