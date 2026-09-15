import math

import torch
import torch.nn.functional as F

from .nn import LayerNorm, Linear


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class MoEFFN(torch.nn.Module):
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
        self.norm = LayerNorm(d_model)

        # Router (no bias, as in Switch Transformer / Mixtral)
        # routing='modality': router is unused but kept as a no-op nn.Linear
        # so the module's state_dict layout stays stable across routing modes.
        self.router = Linear(d_model, self.num_routed, bias=False)

        # Stacked routed expert weights — single big parameters for batched
        # bmm in the vectorized forward path. ffn_type='swiglu', bias=False:
        #   per-expert: Linear(d, 2h) -> SwiGLU -> Linear(h, d)
        # Stacked:
        #   w13: (E, 2h, d)   [first Linear]
        #   w2:  (E, d, h)    [second Linear]
        self.h_dim: int = swiglu_correction_fn(expansion_ratio, d_model)
        self.w13_stacked = torch.nn.Parameter(
            torch.empty(self.num_routed, 2 * self.h_dim, d_model)
        )
        self.w2_stacked = torch.nn.Parameter(
            torch.empty(self.num_routed, d_model, self.h_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        mask_flat = mask.reshape(-1)

        h = self.norm(x_flat)  # (T, D)

        router_logits = self.router(h)  # (T, N)
        router_probs = F.softmax(router_logits, dim=-1)  # (T, N), differentiable

        # Top-k selection (k routed experts per token)
        top_probs, top_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (T, k)
        # Renormalize so the routed-expert weights sum to 1 (Mixtral-style)
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Dispatch tokens to their selected routed experts.
        out = self._dispatch_vectorized(
            h,
            top_idx,
            top_probs,
            mask_flat,
            x_flat.dtype,
        )

        return out.view(orig_shape)

    def _dispatch_vectorized(
        self,
        h: torch.Tensor,
        top_idx: torch.Tensor,
        top_probs: torch.Tensor,
        mask: torch.Tensor,
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
        flat_slot_valid = mask.unsqueeze(-1).expand(T, K).reshape(-1)  # (T*k,)
        # Original token index for each (token, slot) pair.
        flat_token_src = (
            torch.arange(T, device=device).unsqueeze(-1).expand(T, K).reshape(-1)
        )  # (T*k,)

        # ---- Sort valid slots by expert id; place masked slots last -----------
        # Include the slot index to make keys unique and preserve token order
        # within each expert without relying on sort stability.
        flat_slot_idx = torch.arange(T * K, device=device)
        sort_key = torch.where(
            flat_slot_valid,
            flat_top_idx * (T * K) + flat_slot_idx,
            E * (T * K) + flat_slot_idx,
        )
        order = sort_key.argsort()
        sorted_expert = flat_top_idx[order]  # (T*k,)
        sorted_token_src = flat_token_src[order]  # (T*k,)
        sorted_weights = flat_top_probs[order]  # (T*k,)
        sorted_slot_valid = flat_slot_valid[order]  # (T*k,)

        # ---- Per-expert offsets from valid slots only ------------------------
        counts = (
            F.one_hot(flat_top_idx, num_classes=E) * flat_slot_valid.unsqueeze(-1)
        ).sum(dim=0)  # (E,)
        offsets = counts.cumsum(0) - counts  # (E,) start of each expert's chunk

        # Position of each sorted slot within its expert's bucket.
        positions_in_bucket = (
            torch.arange(T * K, device=device) - offsets[sorted_expert]
        )  # (T*k,)

        # ---- Compute static buffer and data-dependent effective capacity ------
        # `buffer_capacity` determines tensor/BMM shapes and depends only on the
        # padded input shape. `effective_capacity` depends on the number of real
        # tokens, but is used only as a scalar mask threshold.
        buffer_capacity = math.ceil(T * K / E * self.capacity_factor / 8) * 8
        buffer_capacity = max(buffer_capacity, 8)
        valid_token_count = mask.sum()
        effective_capacity = (
            torch.ceil(
                valid_token_count.to(torch.float32) * (K * self.capacity_factor / (E * 8))
            ).clamp_min(1)
            * 8
        )

        # ---- Identify overflow + clamp positions to safe range ----
        dispatch_mask = sorted_slot_valid & (positions_in_bucket < effective_capacity)
        positions_clamped = positions_in_bucket.clamp(max=buffer_capacity - 1)
        dispatch_weight = dispatch_mask.to(h.dtype)  # (T*k,)
        eff_weights = sorted_weights * dispatch_weight

        # ---- Build dispatch buffer (E, capacity, D) via index_add_ ----
        flat_buf_idx = sorted_expert * buffer_capacity + positions_clamped  # (T*k,)
        src_h = h.index_select(0, sorted_token_src)  # (T*k, D)
        # Zero out overflow contributions BEFORE the index_add to avoid
        # corrupting the legitimate (capacity-1) cell on collision.
        src_h_masked = src_h * dispatch_weight.unsqueeze(-1)
        buf_flat = torch.zeros(E * buffer_capacity, D, dtype=h.dtype, device=device)
        buf_flat.index_add_(0, flat_buf_idx, src_h_masked)
        buf = buf_flat.view(E, buffer_capacity, D)  # (E, capacity, D)

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
        out_buf_flat = out_buf.reshape(E * buffer_capacity, D)  # (E*capacity, D)
        gathered = out_buf_flat.index_select(0, flat_buf_idx)  # (T*k, D)
        # Apply per-slot weight (top_prob * valid_mask) — overflow → zero.
        weighted = gathered.to(out_dtype) * eff_weights.unsqueeze(-1).to(out_dtype)

        # ---- Scatter weighted outputs back to (T, D) ----
        out = torch.zeros(T, D, dtype=out_dtype, device=device)
        out.index_add_(0, sorted_token_src, weighted)
        return out
