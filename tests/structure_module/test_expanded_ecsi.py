import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.model.modules.structure_module.expanded_ecsi import KFoldExpandedECSI
from kfold.model.modules.structure_module.kfold_ecsi import KFoldECSI


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
    gamma_scale_com: float = 1.0,
    gamma_scale_internal: float = 1.0,
    eta: float = 1.0,
    eta_com: float | None = None,
    eta_internal: float | None = None,
) -> tuple[KFoldECSI, KFoldExpandedECSI]:
    score_model = DummyScoreModel()
    base_cfg = KFoldECSI.Config(
        num_steps=16,
        sigma_min=0.001,
        sigma_max=0.999,
        gamma_max=4.0,
        sigma_data=16.0,
        sigma_data_end=16.0,
        cov_xy=128.0,
        eta=eta,
        coordinate_augmentation=False,
        perturb_xt=False,
        use_forward_pinned_churn=False,
    )
    expanded_cfg = KFoldExpandedECSI.Config(
        num_steps=16,
        sigma_min=0.001,
        sigma_max=0.999,
        gamma_max=4.0,
        sigma_data=16.0,
        sigma_data_end=16.0,
        cov_xy=128.0,
        eta=eta,
        eta_com=eta_com,
        eta_internal=eta_internal,
        gamma_scale_com=gamma_scale_com,
        gamma_scale_internal=gamma_scale_internal,
        coordinate_augmentation=False,
        perturb_xt=False,
        use_forward_pinned_churn=False,
    )
    return KFoldECSI(base_cfg, score_model), KFoldExpandedECSI(expanded_cfg, score_model)


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
    _, expanded = _build_modules()
    coords, _, mask = _make_coords()
    com, internal = expanded.decompose_coords(coords, mask)
    recomposed = expanded.recompose_coords(com, internal, mask)

    assert torch.allclose(recomposed, coords)
    assert torch.allclose(expanded.compute_com(internal, mask), torch.zeros_like(com))


def test_interpolate_matches_base_when_scales_are_one() -> None:
    base, expanded = _build_modules()
    apo, holo, mask = _make_coords()
    t_hat = torch.full((1, 1), 0.37)

    torch.manual_seed(7)
    base_xt = base.interpolate(apo, holo, t_hat, mask)
    torch.manual_seed(7)
    expanded_xt = expanded.interpolate(apo, holo, t_hat, mask)

    assert torch.allclose(expanded_xt, base_xt, atol=1e-6)


def test_interpolate_boundary_with_zero_noise_scales() -> None:
    _, expanded = _build_modules(gamma_scale_com=0.0, gamma_scale_internal=0.0)
    apo, holo, mask = _make_coords()

    x_at_zero = expanded.interpolate(apo, holo, torch.zeros((1, 1)), mask)
    x_at_one = expanded.interpolate(apo, holo, torch.ones((1, 1)), mask)

    assert torch.allclose(x_at_zero, holo, atol=1e-6)
    assert torch.allclose(x_at_one, apo, atol=1e-6)


def test_split_drift_matches_base_when_scales_and_eta_match() -> None:
    base, expanded = _build_modules()
    apo, holo, mask = _make_coords()
    t_hat = torch.full((1, 1, 1, 1), 0.37)
    x_t = 0.6 * holo + 0.4 * apo
    x0_hat = 0.7 * holo + 0.3 * apo

    alpha_t = base.alpha(t_hat)
    beta_t = base.beta(t_hat)
    gamma_t = base.gamma(t_hat)
    alpha_dot = base.alpha_deriv(t_hat)
    beta_dot = base.beta_deriv(t_hat)
    gamma_dot = base.gamma_deriv(t_hat)
    z_hat = (x_t - alpha_t * x0_hat - beta_t * apo) / (gamma_t + 1e-8)
    eps_t = base.eta * (
        gamma_t * gamma_dot - (alpha_dot / (alpha_t + 1e-8)) * gamma_t**2
    )
    base_drift = (
        alpha_dot * x0_hat
        + beta_dot * apo
        + (gamma_dot + eps_t / (gamma_t + 1e-8)) * z_hat
    )

    drift_com, drift_internal, eps_com, eps_internal = (
        expanded._compute_drift_components(
            x_t=x_t,
            x0_hat=x0_hat,
            x_T=apo,
            t_exp=t_hat,
            mask=mask.unsqueeze(1),
        )
    )

    assert torch.allclose(drift_com + drift_internal, base_drift, atol=1e-5)
    assert torch.allclose(eps_com, eps_internal, atol=1e-6)
