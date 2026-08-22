import pytest
import torch

from kfold.model.modules.structure.ecsi import ECSISOARConfig, KFoldECSI, SICoeffs
from kfold.training.training_module import _training_metric_log_name


def make_math_only_ecsi() -> KFoldECSI:
    ecsi = object.__new__(KFoldECSI)
    ecsi.coeff = SICoeffs(gamma_max=24.0, gamma_power=1.0, eta=1.0)
    ecsi.time_min = 1e-8
    ecsi.time_max = 0.9999
    ecsi.churn_end_time = 0.5
    ecsi.churn_max_time = ecsi.time_max
    ecsi.churn_factor = 0.1
    ecsi.churn_max_multiplier = 4.0
    ecsi.churn_step_fraction = 0.4
    ecsi.churn_step_power = 1.0
    ecsi.ode_step_power = 2.0
    ecsi.stepwarp_power = 0.5
    return ecsi


def test_soar_config_accepts_exact_markov_model_sampler_only() -> None:
    config = ECSISOARConfig(mode="model_sampler")
    assert config.num_auxiliary_samples(16) == 64
    with pytest.raises(ValueError, match="Unknown ECSI SOAR mode"):
        ECSISOARConfig(mode="model_si")
    with pytest.raises(ValueError, match="Unknown ECSI SOAR config fields"):
        ECSISOARConfig.from_mapping({"mode": "model_sampler", "extra": 1})
    with pytest.raises(ValueError, match="auxiliary transition"):
        ECSISOARConfig(auxiliary_transition="coupled_transport")


def test_soar_metrics_use_top_level_namespace() -> None:
    assert _training_metric_log_name("soar_t_call_mean") == "soar/t_call_mean"
    assert _training_metric_log_name("loss") == "train/loss"


def test_soar_times_follow_effective_0811_schedule() -> None:
    ecsi = make_math_only_ecsi()
    config = ECSISOARConfig(mode="model_sampler", auxiliary_samples_per_root=2)
    schedule = torch.tensor(
        ecsi.get_effective_sampling_schedule(config.rollout_schedule_num_steps),
        dtype=torch.float64,
    )
    indices = torch.tensor([0, 7, 25, 72, 98, 99])
    t0 = schedule[indices].view(1, -1)
    torch.manual_seed(11)
    t1, t2 = ecsi.sample_soar_auxiliary_times(t0=t0, config=config)
    torch.testing.assert_close(
        t1,
        schedule[indices + 1].clamp_min(ecsi.time_min).view(1, -1),
        rtol=0.0,
        atol=1e-15,
    )
    assert t2.shape == (1, len(indices), 2)
    assert torch.all(t2 >= t1.unsqueeze(-1))
    assert torch.all(t2 <= ecsi.time_max)


def test_shared_noise_pairs_model_and_oracle_ecsi_sde_updates() -> None:
    ecsi = make_math_only_ecsi()
    x_t = torch.randn(1, 1, 5, 3, dtype=torch.float64)
    x_T = torch.randn_like(x_t)
    model_endpoint = torch.randn_like(x_t)
    oracle_endpoint = torch.randn_like(x_t)
    mask = torch.ones(1, 1, 5, dtype=torch.bool)
    noise = torch.randn_like(x_t)
    model = ecsi._update_step(
        x_t,
        model_endpoint,
        x_T,
        mask,
        0.8,
        0.75,
        mode="sde",
        ode_type="ecsi",
        noise=noise,
    )
    oracle = ecsi._update_step(
        x_t,
        oracle_endpoint,
        x_T,
        mask,
        0.8,
        0.75,
        mode="sde",
        ode_type="ecsi",
        noise=noise,
    )
    zero_noise_model = ecsi._update_step(
        x_t,
        model_endpoint,
        x_T,
        mask,
        0.8,
        0.75,
        mode="sde",
        ode_type="ecsi",
        noise=torch.zeros_like(noise),
    )
    zero_noise_oracle = ecsi._update_step(
        x_t,
        oracle_endpoint,
        x_T,
        mask,
        0.8,
        0.75,
        mode="sde",
        ode_type="ecsi",
        noise=torch.zeros_like(noise),
    )
    torch.testing.assert_close(
        model - oracle,
        zero_noise_model - zero_noise_oracle,
        rtol=1e-12,
        atol=1e-12,
    )


def test_soar_rollout_bypasses_churn_exactly() -> None:
    ecsi = make_math_only_ecsi()
    config = ECSISOARConfig(mode="model_sampler", apply_rollout_churn=False)
    x_t = torch.randn(1, 2, 4, 3)
    x_T = torch.randn_like(x_t)
    mask = torch.ones(1, 2, 4, dtype=torch.bool)
    t = torch.tensor([[0.7, 0.9]])
    x_call, t_call = ecsi._prepare_soar_rollout_call(
        config=config, x_t=x_t, x_T=x_T, mask=mask, t=t
    )
    assert x_call is x_t
    assert t_call is t


def test_exact_forward_transition_matches_formula_and_seeded_noise() -> None:
    ecsi = make_math_only_ecsi()
    x_t1 = torch.randn(1, 2, 4, 3, dtype=torch.float64)
    x_T = torch.randn_like(x_t1)
    mask = torch.ones(1, 2, 4, dtype=torch.bool)
    t1 = torch.tensor([[0.4, 0.7]], dtype=torch.float64)
    t2 = torch.tensor([[[0.5, 0.8], [0.75, 0.9]]], dtype=torch.float64)
    noise = torch.randn(1, 2, 2, 4, 3, dtype=torch.float64)
    transition = ecsi._exact_ecsi_forward_transition(
        x_t1=x_t1,
        x_T=x_T,
        mask=mask,
        t1=t1,
        t2=t2,
        noise=noise,
    )
    retention = (1.0 - t2) / (1.0 - t1).unsqueeze(-1)
    expected_variance = ecsi.coeff.eta * (
        ecsi.coeff.gamma(t2).square()
        - retention.square() * ecsi.coeff.gamma(t1).unsqueeze(-1).square()
    )
    expected_mean = retention[..., None, None] * x_t1.unsqueeze(2) + (
        t2 - retention * t1.unsqueeze(-1)
    )[..., None, None] * x_T.unsqueeze(2)
    torch.testing.assert_close(transition["variance"], expected_variance)
    torch.testing.assert_close(
        transition["x_t2"],
        expected_mean + expected_variance.sqrt()[..., None, None] * noise,
    )


def test_mid_only_root_policy_is_uniform_over_intersecting_schedule_cells() -> None:
    ecsi = make_math_only_ecsi()
    config = ECSISOARConfig(
        mode="model_sampler",
        root_time_policy="mid_high_schedule_stratified",
        mid_time_lower=0.2,
        high_time_split=0.8,
        mid_time_probability=1.0,
    )
    base_t0 = torch.zeros(1, 20000)
    torch.manual_seed(23)
    roots = ecsi.construct_soar_root_times(base_t0=base_t0, config=config)
    assert torch.all((roots >= 0.2) & (roots < 0.8))

    schedule = torch.tensor(
        ecsi.get_effective_sampling_schedule(config.rollout_schedule_num_steps),
        dtype=roots.dtype,
    )
    cell_high = torch.minimum(schedule[:-1], torch.full_like(schedule[:-1], 0.8))
    cell_low = torch.maximum(schedule[1:], torch.full_like(schedule[1:], 0.2))
    valid_indices = torch.nonzero(cell_high > cell_low, as_tuple=False).flatten()
    sampled_indices = ((schedule[:-1] >= roots.unsqueeze(-1)).sum(dim=-1) - 1).clamp(
        min=0, max=config.rollout_schedule_num_steps - 1
    )
    counts = torch.stack([(sampled_indices == index).sum() for index in valid_indices])
    expected = roots.numel() / valid_indices.numel()
    assert torch.all((counts.float() - expected).abs() < 0.2 * expected)


def test_mid_only_root_policy_preserves_recent_rng_consumption() -> None:
    ecsi = make_math_only_ecsi()
    config = ECSISOARConfig(
        mode="model_sampler",
        root_time_policy="mid_high_schedule_stratified",
        mid_time_lower=0.2,
        high_time_split=0.8,
        mid_time_probability=1.0,
    )
    base_t0 = torch.zeros(1, 32)
    schedule = torch.tensor(
        ecsi.get_effective_sampling_schedule(config.rollout_schedule_num_steps),
        dtype=base_t0.dtype,
    )
    torch.manual_seed(31)
    observed = ecsi.construct_soar_root_times(base_t0=base_t0, config=config)
    observed_next_random = torch.rand_like(base_t0)

    torch.manual_seed(31)
    expected = ecsi._sample_soar_schedule_cell_band(
        shape=base_t0.shape,
        schedule=schedule,
        lower=0.2,
        upper=0.8,
    )
    ecsi._sample_soar_schedule_cell_band(
        shape=base_t0.shape,
        schedule=schedule,
        lower=0.8,
        upper=ecsi.time_max,
    )
    torch.rand_like(base_t0)
    expected_next_random = torch.rand_like(base_t0)
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(observed_next_random, expected_next_random)
