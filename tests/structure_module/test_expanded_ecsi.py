import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.model.modules.structure_module.kfold_ecsi import (
    KFoldECSI,
    SamplingConfig,
    TrainTimeSamplingConfig,
)


class DummyScoreModel(BaseScoreModel):
    def __init__(self) -> None:
        super().__init__(cfg=None, kernel_config=None)

    def forward(  # type: ignore[override]
        self,
        r_noisy: torch.Tensor,
        c_noise: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache: dict | None = None,
    ) -> torch.Tensor:
        del c_noise, f_input, s_inputs, s_trunk, z_trunk, model_cache
        return torch.zeros_like(r_noisy[..., :3])


def _build_modules(
    *,
    gamma_max: float = 4.0,
    gamma_scale_com: float = 1.0,
    gamma_scale_internal: float = 1.0,
    sampling_eta: float = 1.0,
    sampling_eta_com: float | None = None,
    sampling_eta_internal: float | None = None,
) -> KFoldECSI:
    score_model = DummyScoreModel()
    cfg = KFoldECSI.Config(
        gamma_max=gamma_max,
        gamma_scale_com=gamma_scale_com,
        gamma_scale_internal=gamma_scale_internal,
        sampling=SamplingConfig(
            steps=16,
            time_min=0.001,
            time_max=0.999,
            eta=sampling_eta,
            eta_com=sampling_eta_com,
            eta_internal=sampling_eta_internal,
            perturb_xt=False,
            use_pinned_churn=False,
        ),
        train_time_sampling=TrainTimeSamplingConfig(),
        sigma_data=16.0,
        sigma_data_end=16.0,
        cov_xy=128.0,
        coordinate_augmentation=False,
    )
    return KFoldECSI(cfg, score_model)


def _make_coords() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    apo = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    holo = torch.tensor(
        [
            [
                [
                    [2.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [2.0, 1.0, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.ones(1, 4, dtype=torch.bool)
    return apo, holo, mask


def test_decompose_recompose_is_exact() -> None:
    expanded = _build_modules()
    coords, _, mask = _make_coords()
    com, internal = expanded.decompose_coords(coords, mask)
    recomposed = expanded.recompose_coords(com, internal, mask)

    assert torch.allclose(recomposed, coords)
    assert torch.allclose(expanded.compute_com(internal, mask), torch.zeros_like(com))


def test_interpolate_matches_full_noise_when_component_maxima_match() -> None:
    expanded = _build_modules()
    apo, holo, mask = _make_coords()
    t_hat = torch.full((1, 1), 0.37)

    t_exp = t_hat[:, :, None, None]
    alpha_t = expanded.si_coeffs.alpha(t_exp)
    beta_t = expanded.si_coeffs.beta(t_exp)
    gamma_t = expanded.si_coeffs.gamma(t_exp)

    torch.manual_seed(7)
    expanded_xt = expanded.interpolate(apo, holo, t_hat, mask)
    torch.manual_seed(7)
    expected_xt = (
        alpha_t * holo
        + beta_t * apo
        + gamma_t * torch.randn_like(apo)
    ) * mask[:, None, :, None]

    assert torch.allclose(expanded_xt, expected_xt, atol=1e-6)


def test_interpolate_boundary_with_zero_noise_scales() -> None:
    expanded = _build_modules(gamma_scale_com=0.0, gamma_scale_internal=0.0)
    apo, holo, mask = _make_coords()

    x_at_zero = expanded.interpolate(apo, holo, torch.zeros((1, 1)), mask)
    x_at_one = expanded.interpolate(apo, holo, torch.ones((1, 1)), mask)

    assert torch.allclose(x_at_zero, holo, atol=1e-6)
    assert torch.allclose(x_at_one, apo, atol=1e-6)


def test_component_scales_follow_component_maxima() -> None:
    expanded = _build_modules(
        gamma_max=4.0,
        gamma_scale_com=2.0,
        gamma_scale_internal=1.0,
    )
    t_hat = torch.full((1, 1, 1, 1), 0.37)

    gamma_com, gamma_internal, gamma_dot_com, gamma_dot_internal = (
        expanded._component_scales(t_hat)
    )

    assert torch.allclose(gamma_com, expanded.si_coeffs.gamma_com(t_hat), atol=1e-6)
    assert torch.allclose(
        gamma_internal,
        expanded.si_coeffs.gamma_internal(t_hat),
        atol=1e-6,
    )
    assert torch.allclose(
        gamma_dot_com,
        expanded.si_coeffs.gamma_com_deriv(t_hat),
        atol=1e-6,
    )
    assert torch.allclose(
        gamma_dot_internal,
        expanded.si_coeffs.gamma_internal_deriv(t_hat),
        atol=1e-6,
    )


def test_train_time_uniform_beta_mixture_sampling_runs() -> None:
    expanded = _build_modules()
    expanded.train_time_sampling.uniform_mix_prob = 0.5

    t_hat = expanded.sample_noise_level(
        batch_size=4,
        num_diffusion_samples=8,
        device=torch.device("cpu"),
    )

    assert t_hat.shape == (4, 8)
    assert torch.all(t_hat >= expanded.sampling.time_min)
    assert torch.all(t_hat <= expanded.sampling.time_max)
