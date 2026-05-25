import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class MoEFFN(nn.Module):
    """
    Sparse MoE FFN with optional shared experts (DeepSeek-MoE / Qwen-MoE style).

    Layout (default for this project):
        num_routed_experts = 8   (router selects top_k of these per token)
        top_k = 2                (routed experts active per token)
        => total experts        = 8 + 1 = 9
           active experts/token = 2 routed + 1 shared = 3

    Switch Transformer (Fedus et al., JMLR 2022) load balancing aux loss:
        For N routed experts and a batch of T tokens:
            f_i = fraction of tokens routed to expert i (non-diff)
            P_i = mean of router softmax prob for expert i over the batch (diff)
            L_aux = N * Σ_i (f_i * P_i)        (computed per layer)
        Gradient flows only through P_i. Final scaling by alpha is applied
        once at the LightningModule level after summing across all layers.

    `last_aux_loss` is populated on every forward and consumed by the
    enclosing TransformerStack.
    """

    def __init__(
        self,
        d_model: int,
        expansion_ratio: float,
        num_routed_experts: int = 4,
        top_k: int = 1,
        capacity_factor: float = 2.0,
    ):
        super().__init__()
        assert num_routed_experts >= 1
        self.d_model = d_model
        self.num_routed = num_routed_experts
        self.top_k = top_k
        assert 1 <= top_k <= self.num_routed, (
            f"top_k ({top_k}) must be in [1, num_routed={self.num_routed}]"
        )
        self.capacity_factor: float = capacity_factor

        # Pre-FFN LayerNorm (matches dense ffn structure)
        self.norm = nn.LayerNorm(d_model)

        # Router (no bias, as in Switch Transformer / Mixtral)
        # routing='modality': router is unused but kept as a no-op nn.Linear
        # so the module's state_dict layout stays stable across routing modes.
        self.router = nn.Linear(d_model, self.num_routed, bias=False)

        # Stacked routed expert weights — single big parameters for batched
        # bmm in the vectorized forward path. ffn_type='swiglu', bias=False:
        #   per-expert: Linear(d, 2h) -> SwiGLU -> Linear(h, d)
        # Stacked:
        #   w13: (E, 2h, d)   [first Linear]
        #   w2:  (E, d, h)    [second Linear]
        self.h_dim: int = swiglu_correction_fn(expansion_ratio, d_model)
        self.w13_stacked = nn.Parameter(
            torch.empty(self.num_routed, 2 * self.h_dim, d_model)
        )
        self.w2_stacked = nn.Parameter(torch.empty(self.num_routed, d_model, self.h_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)

        h = self.norm(x_flat)  # (T, D)

        # Routing (run on ALL positions — no correctness issue, Mixtral/Qwen2 style)
        router_logits = self.router(h)  # (T, N)
        router_probs = F.softmax(router_logits, dim=-1)  # (T, N), differentiable

        # Top-k selection (k routed experts per token)
        top_probs, top_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (T, k)
        # Renormalize so the routed-expert weights sum to 1 (Mixtral-style)
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Determine dtype for the routed expert dispatch buffer.
        # Under bf16-mixed autocast, the LayerNorm output `h` stays in fp32
        # while expert Linear outputs are bf16. We must allocate buffers in
        # the autocast dtype so index_add_ (strict on dtype) succeeds.
        if torch.is_autocast_enabled():
            out_dtype = torch.get_autocast_gpu_dtype()
        else:
            out_dtype = h.dtype

        # Dispatch tokens to their selected routed experts.
        out = self._dispatch_vectorized(h, top_idx, top_probs, out_dtype)

        return out.view(orig_shape)

    def _dispatch_vectorized(
        self,
        h: torch.Tensor,
        top_idx: torch.Tensor,
        top_probs: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Vectorized routed-expert dispatch using padded capacity grouped bmm:
        """
        T = h.shape[0]
        D = self.d_model
        E = self.num_routed
        K = self.top_k
        device = h.device

        # ---- Flatten (T, k) routing into (T*k,) per-slot tensors -------
        flat_top_idx = top_idx.reshape(-1)  # (T*k,)
        flat_top_probs = top_probs.reshape(-1)  # (T*k,)
        # Original token index for each (token, slot) pair.
        flat_token_src = (
            torch.arange(T, device=device).unsqueeze(-1).expand(T, K).reshape(-1)
        )  # (T*k,)

        # ---- Sort slots by expert id so each expert's slots are contiguous ----
        order = flat_top_idx.argsort()
        sorted_expert = flat_top_idx[order]  # (T*k,)
        sorted_token_src = flat_token_src[order]  # (T*k,)
        sorted_weights = flat_top_probs[order]  # (T*k,)

        # ---- Per-expert offsets via bincount + cumsum ----
        counts = torch.bincount(sorted_expert, minlength=E)  # (E,)
        offsets = counts.cumsum(0) - counts  # (E,) start of each expert's chunk

        # Position of each sorted slot within its expert's bucket.
        positions_in_bucket = (
            torch.arange(T * K, device=device) - offsets[sorted_expert]
        )  # (T*k,)

        # ---- Compute capacity (rounded up to multiple of 8) ----
        # Using a Python int derived from a symbolic T: under dynamic=True,
        # this stays symbolic in the compile graph (no recompile per shape).
        capacity = math.ceil(T * K / E * self.capacity_factor / 8) * 8
        capacity = max(capacity, 8)  # never zero

        # ---- Identify overflow + clamp positions to safe range ----
        overflow_mask = positions_in_bucket >= capacity
        positions_clamped = positions_in_bucket.clamp(max=capacity - 1)
        # Effective dispatch weight: zero for overflowed slots → drops them.
        valid_mask = (~overflow_mask).to(h.dtype)  # (T*k,)
        eff_weights = sorted_weights * valid_mask

        # ---- Build dispatch buffer (E, capacity, D) via index_add_ ----
        flat_buf_idx = sorted_expert * capacity + positions_clamped  # (T*k,)
        src_h = h.index_select(0, sorted_token_src)  # (T*k, D)
        # Zero out overflow contributions BEFORE the index_add to avoid
        # corrupting the legitimate (capacity-1) cell on collision.
        src_h_masked = src_h * valid_mask.unsqueeze(-1).to(h.dtype)
        buf_flat = torch.zeros(E * capacity, D, dtype=h.dtype, device=device)
        buf_flat.index_add_(0, flat_buf_idx, src_h_masked)
        buf = buf_flat.view(E, capacity, D)  # (E, capacity, D)

        # ---- Batched expert forward (single big bmm × 2) ----
        # buf: (E, capacity, D)
        # w13_stacked: (E, 2h, D)  →  transpose → (E, D, 2h)
        # bmm:       (E, capacity, D) @ (E, D, 2h)  →  (E, capacity, 2h)
        gate_up = torch.bmm(buf, self.w13_stacked.transpose(-1, -2))  # (E, capacity, 2h)
        g, u = gate_up.chunk(2, dim=-1)
        activated = F.silu(g) * u  # (E, capacity, h)
        # w2_stacked: (E, D, h) → transpose → (E, h, D)
        out_buf = torch.bmm(
            activated, self.w2_stacked.transpose(-1, -2)
        )  # (E, capacity, D)

        # ---- Unpermute: gather expert outputs back into per-slot order ----
        out_buf_flat = out_buf.reshape(E * capacity, D)  # (E*capacity, D)
        gathered = out_buf_flat.index_select(0, flat_buf_idx)  # (T*k, D)
        # Apply per-slot weight (top_prob * valid_mask) — overflow → zero.
        weighted = gathered.to(out_dtype) * eff_weights.unsqueeze(-1).to(out_dtype)

        # ---- Scatter weighted outputs back to (T, D) ----
        out = torch.zeros(T, D, dtype=out_dtype, device=device)
        out.index_add_(0, sorted_token_src, weighted)
        return out
