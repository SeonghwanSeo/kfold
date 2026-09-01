"""Public API for fused incoming and outgoing triangle multiplicative update."""

import torch

from .._common.ln_absorption import absorb_ln
from .kernels import input_phase, output_phase


def precompute(
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
):
    """Fold LayerNorm affine parameters into the four projection weights."""
    proj_dtype = p_in_weight.dtype
    p_in_c, sWp_in, Bp_in = absorb_ln(
        p_in_weight, norm_in_weight, norm_in_bias, weight_dtype=proj_dtype
    )
    g_in_c, sWg_in, Bg_in = absorb_ln(
        g_in_weight, norm_in_weight, norm_in_bias, weight_dtype=proj_dtype
    )
    p_out_c, sWp_out, Bp_out = absorb_ln(
        p_out_weight, norm_out_weight, norm_out_bias, weight_dtype=proj_dtype
    )
    g_out_c, sWg_out, Bg_out = absorb_ln(
        g_out_weight, norm_in_weight, norm_in_bias, weight_dtype=proj_dtype
    )
    return {
        "p_in_combined": p_in_c,
        "g_in_combined": g_in_c,
        "p_out_combined": p_out_c,
        "g_out_combined": g_out_c,
        "sum_W_p_in": sWp_in,
        "sum_W_g_in": sWg_in,
        "sum_W_p_out": sWp_out,
        "sum_W_g_out": sWg_out,
        "B_p_in_const": Bp_in,
        "B_g_in_const": Bg_in,
        "B_p_out_const": Bp_out,
        "B_g_out_const": Bg_out,
    }


def forward(x, pre, *, direction, mask=None, eps=1e-5):
    """Apply the selected update direction using precomputed weights."""
    ab_t = input_phase(
        x,
        mask,
        pre["p_in_combined"],
        pre["g_in_combined"],
        pre["sum_W_p_in"],
        pre["sum_W_g_in"],
        pre["B_p_in_const"],
        pre["B_g_in_const"],
        eps,
    )  # (2D, B, L, L)

    a, b_ab = torch.chunk(ab_t, 2, dim=0)  # each (D, B, L, L)
    if direction == "outgoing":
        y = torch.einsum("dbik,dbjk->dbij", a, b_ab)
    elif direction == "incoming":
        y = torch.einsum("dbki,dbkj->dbij", a, b_ab)
    else:
        raise ValueError(
            f"unknown direction: {direction!r} (expected 'outgoing'/'incoming')"
        )

    return output_phase(
        y,
        x,
        pre["p_out_combined"],
        pre["g_out_combined"],
        pre["sum_W_p_out"],
        pre["sum_W_g_out"],
        pre["B_p_out_const"],
        pre["B_g_out_const"],
        eps,
    )


def triangle_multiplicative_update(
    x,
    *,
    direction,
    mask=None,
    eps=1e-5,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
):
    """Run triangle multiplicative update with projection precomputation included."""
    pre = precompute(
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        g_out_weight,
    )
    return forward(x, pre, direction=direction, mask=mask, eps=eps)
