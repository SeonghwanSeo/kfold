# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public API for fused incoming and outgoing triangle multiplicative update."""

import torch

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
    """Pack projections and retain FP32 LayerNorm affine parameters.

    Normalizing before the dot products avoids quantized folded weights and
    cancellation when inputs have a large mean, while keeping the kernels fused.
    """
    return {
        "p_in_weight": p_in_weight.contiguous(),
        "g_in_weight": g_in_weight.contiguous(),
        "p_out_weight": p_out_weight.contiguous(),
        "g_out_weight": g_out_weight.contiguous(),
        "norm_in_weight": norm_in_weight.float().contiguous(),
        "norm_in_bias": norm_in_bias.float().contiguous(),
        "norm_out_weight": norm_out_weight.float().contiguous(),
        "norm_out_bias": norm_out_bias.float().contiguous(),
    }


def forward(x, pre, *, direction, mask=None, eps=1e-5):
    """Apply the selected update direction using precomputed weights."""
    ab_t = input_phase(
        x,
        mask,
        pre["p_in_weight"],
        pre["g_in_weight"],
        pre["norm_in_weight"],
        pre["norm_in_bias"],
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
        pre["p_out_weight"],
        pre["g_out_weight"],
        pre["norm_out_weight"],
        pre["norm_out_bias"],
        pre["norm_in_weight"],
        pre["norm_in_bias"],
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
