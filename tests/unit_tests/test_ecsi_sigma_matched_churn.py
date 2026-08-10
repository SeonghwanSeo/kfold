"""Targeted correctness tests for the minimal ECSI churn sampler."""

import math
from types import SimpleNamespace

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


def make_class_routing_input() -> SimpleNamespace:
    """Return a minimal batched input with a covalent ligand-chain component."""
    return SimpleNamespace(
        chain=SimpleNamespace(
            asym_id=torch.tensor([[10, 20, 30, 40, 50]]),
            pad_mask=torch.tensor([[True, True, True, True, True]]),
            is_protein=torch.tensor([[True, False, True, False, False]]),
            is_ligand=torch.tensor([[False, True, False, False, False]]),
            is_rna=torch.tensor([[False, False, False, True, False]]),
            is_dna=torch.tensor([[False, False, False, False, True]]),
            num_residues=torch.tensor([[100, 1, 8, 12, 12]]),
        ),
        token=SimpleNamespace(
            asym_id=torch.tensor([[10, 20, 30, 40, 50]]),
            pad_mask=torch.tensor([[True, True, True, True, True]]),
        ),
        atom=SimpleNamespace(
            token_index=torch.tensor([[0, 1, 2, 3, 4, 0]]),
            pad_mask=torch.tensor([[True, True, True, True, True, False]]),
        ),
        bond=SimpleNamespace(
            asym_id=torch.tensor([[[20, 10]]]),
            pad_mask=torch.tensor([[True]]),
        ),
    )


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
    assert config.sampler_sde_atom_classes == ("all",)
    assert not hasattr(config, "churn_space")
    assert not hasattr(config, "svgd_step")


def test_class_sde_mask_selects_classes_and_covalent_components() -> None:
    sampler = make_sampler(sampler_sde_atom_classes=("ligand", "peptide", "rna"))

    selected = sampler._get_sde_atom_mask(make_class_routing_input())

    assert selected is not None
    assert torch.equal(
        selected,
        torch.tensor([[True, True, True, True, False, False]]),
    )


def test_protein_class_selects_protein_and_covalent_components() -> None:
    sampler = make_sampler(sampler_sde_atom_classes=("protein",))

    selected = sampler._get_sde_atom_mask(make_class_routing_input())

    assert torch.equal(
        selected,
        torch.tensor([[True, True, True, False, False, False]]),
    )


def test_all_and_empty_class_selectors_are_explicit() -> None:
    f_input = make_class_routing_input()

    assert torch.equal(
        make_sampler()._get_sde_atom_mask(f_input),
        f_input.atom.pad_mask,
    )
    selected = make_sampler(sampler_sde_atom_classes=())._get_sde_atom_mask(f_input)
    assert not selected.any()


def test_class_sde_selector_rejects_unknown_or_duplicate_classes() -> None:
    with pytest.raises(ValueError, match="must be explicit"):
        make_sampler(sampler_sde_atom_classes=None)
    with pytest.raises(ValueError, match="unsupported classes"):
        make_sampler(sampler_sde_atom_classes=("carbohydrate",))
    with pytest.raises(ValueError, match="must not contain duplicates"):
        make_sampler(sampler_sde_atom_classes=("ligand", "ligand"))
    with pytest.raises(ValueError, match="must be used alone"):
        make_sampler(sampler_sde_atom_classes=("all", "ligand"))


def test_class_sde_routing_uses_the_global_hybrid_profile() -> None:
    sampler = make_sampler(sampler_sde_atom_classes=("ligand",))

    assert sampler._select_update_method(0.9) == ("sde", "ecsi")
    assert sampler._select_update_method(0.3) == ("sde", "ecsi")
    assert sampler._select_update_method(0.05) == ("ode", "si")


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


def test_class_selective_update_keeps_unselected_atoms_on_si_ode() -> None:
    sampler = make_sampler(sampler_sde_atom_classes=("ligand",))
    x_t, x_T, mask = make_state()
    x_0_hat = torch.full_like(x_t, 0.25)
    selected_atoms = torch.tensor([[False, True, False, False, False]])

    expected_ode = sampler._update_step(
        x_t,
        x_0_hat,
        x_T,
        mask,
        0.7,
        0.6,
        mode="ode",
        ode_type="si",
        step_scale=sampler.sampler_step_scale,
    )
    torch.manual_seed(47)
    observed = sampler._apply_class_selective_update(
        x_t,
        x_0_hat,
        x_T,
        mask,
        selected_atoms,
        True,
        0.7,
        0.6,
    )

    unselected = ~selected_atoms[0]
    torch.testing.assert_close(
        observed[:, :, unselected, :],
        expected_ode[:, :, unselected, :],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(observed[:, :, 1, :], expected_ode[:, :, 1, :])


@pytest.mark.parametrize("sampler_mode", ["sde", "ode"])
def test_all_class_selector_matches_the_global_update(sampler_mode: str) -> None:
    sampler = make_sampler(sampler_mode=sampler_mode)
    x_t, x_T, mask = make_state()
    x_0_hat = torch.full_like(x_t, 0.25)
    selected_atoms = torch.ones(mask.shape[0], mask.shape[-1], dtype=torch.bool)
    mode, ode_type = sampler._select_update_method(0.7)

    torch.manual_seed(53)
    expected = sampler._update_step(
        x_t,
        x_0_hat,
        x_T,
        mask,
        0.7,
        0.6,
        mode=mode,
        ode_type=ode_type,
        step_scale=sampler.sampler_step_scale,
    )
    torch.manual_seed(53)
    observed = sampler._apply_class_selective_update(
        x_t,
        x_0_hat,
        x_T,
        mask,
        selected_atoms,
        True,
        0.7,
        0.6,
    )

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_empty_class_selector_uses_si_ode_without_consuming_rng() -> None:
    sampler = make_sampler(sampler_sde_atom_classes=())
    x_t, x_T, mask = make_state()
    x_0_hat = torch.full_like(x_t, 0.25)
    selected_atoms = torch.zeros(mask.shape[0], mask.shape[-1], dtype=torch.bool)

    expected_ode = sampler._update_step(
        x_t,
        x_0_hat,
        x_T,
        mask,
        0.7,
        0.6,
        mode="ode",
        ode_type="si",
        step_scale=sampler.sampler_step_scale,
    )
    torch.manual_seed(59)
    expected_next_noise = torch.randn_like(x_t)
    torch.manual_seed(59)
    observed = sampler._apply_class_selective_update(
        x_t,
        x_0_hat,
        x_T,
        mask,
        selected_atoms,
        False,
        0.7,
        0.6,
    )
    actual_next_noise = torch.randn_like(x_t)

    torch.testing.assert_close(observed, expected_ode, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_next_noise,
        expected_next_noise,
        rtol=0.0,
        atol=0.0,
    )


def test_class_sde_low_time_fallback_does_not_consume_rng() -> None:
    sampler = make_sampler(sampler_sde_atom_classes=("ligand",))
    x_t, x_T, mask = make_state()
    x_0_hat = torch.full_like(x_t, 0.25)
    selected_atoms = torch.tensor([[False, True, False, False, False]])

    expected_ode = sampler._update_step(
        x_t,
        x_0_hat,
        x_T,
        mask,
        0.05,
        0.01,
        mode="ode",
        ode_type="si",
        step_scale=sampler.sampler_step_scale,
    )
    torch.manual_seed(61)
    expected_next_noise = torch.randn_like(x_t)
    torch.manual_seed(61)
    observed = sampler._apply_class_selective_update(
        x_t,
        x_0_hat,
        x_T,
        mask,
        selected_atoms,
        True,
        0.05,
        0.01,
    )
    actual_next_noise = torch.randn_like(x_t)

    torch.testing.assert_close(observed, expected_ode, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_next_noise,
        expected_next_noise,
        rtol=0.0,
        atol=0.0,
    )


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
