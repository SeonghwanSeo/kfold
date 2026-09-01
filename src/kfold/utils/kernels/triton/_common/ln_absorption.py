"""LayerNorm-affine absorption — the shared lever behind all three kernels.

A LayerNorm immediately followed by a linear projection can be collapsed so the
LN affine is folded into the projection weight.  This removes the separate LN
kernel/buffer and lets the fused kernel produce (mean, var, projection) in one
pass over the feature dim.

Derivation (per token, over feature dim of size C):
    LN(x)[i] = (x[i] - mean) · rstd · w_ln[i] + b_ln[i]
    (LN(x) @ Wᵀ)[o] = rstd · ( Σ_i x[i]·Wc[o,i] − mean·ΣW[o] ) + Bc[o]
        Wc[o,i] = w_ln[i] · W[o,i]
        ΣW[o]   = Σ_i Wc[o,i]
        Bc[o]   = Σ_i b_ln[i] · W[o,i]

So a model-init precompute of (Wc, ΣW, Bc) lets the kernel apply the LN affine
with only the per-token (mean, rstd) it already computes via Welford.

Two conventions are provided because the three ops carry weights differently:
  - `absorb_ln`        : weight is (out, in)  — torch.nn.Linear `.weight` style.
  - `absorb_ln_matmul` : weight is (in, out)  — bare `x @ W` matmul style.
Both return ΣW and Bc in fp32 (they feed fp32 accumulators); Wc is cast to
`weight_dtype` (default = weight.dtype) so it can stay bf16/fp16 for tensor cores.
"""

import torch


def absorb_ln(
    weight: torch.Tensor,
    w_ln: torch.Tensor,
    b_ln: torch.Tensor,
    *,
    weight_dtype: torch.dtype | None = None,
):
    """Fold LN affine into an (out, in) nn.Linear-convention weight.

    Returns: Wc (out, in)[weight_dtype], sum_W (out,)[fp32], B_const (out,)[fp32].
    """
    Wf = weight.float()
    Wc = Wf * w_ln.float().unsqueeze(0)  # (out, in)
    sum_W = Wc.sum(dim=1).contiguous()  # (out,)
    B_const = (Wf @ b_ln.float()).contiguous()  # (out,)
    wd = weight_dtype if weight_dtype is not None else weight.dtype
    return Wc.to(wd).contiguous(), sum_W, B_const


def absorb_ln_matmul(
    weight: torch.Tensor,
    w_ln: torch.Tensor,
    b_ln: torch.Tensor,
    *,
    weight_dtype: torch.dtype | None = None,
):
    """Fold LN affine into an (in, out) matmul-convention weight (y = LN(x) @ W).

    Returns: Wc (in, out)[weight_dtype], sum_W (out,)[fp32], B_const (out,)[fp32].
    """
    Wf = weight.float()
    Wc = Wf * w_ln.float().unsqueeze(-1)  # (in, out)
    sum_W = Wc.sum(dim=0).contiguous()  # (out,)
    B_const = (b_ln.float() @ Wf).contiguous()  # (out,)
    wd = weight_dtype if weight_dtype is not None else weight.dtype
    return Wc.to(wd).contiguous(), sum_W, B_const
