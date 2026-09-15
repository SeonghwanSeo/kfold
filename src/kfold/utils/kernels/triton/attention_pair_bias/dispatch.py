import math

import torch
import torch.nn.functional as F

try:
    from cuequivariance_ops_torch.attention_pair_bias_torch import (
        attention_pair_bias_mask as cueq_attention_pair_bias_mask,
    )
except ImportError:
    cueq_attention_pair_bias_mask = None

from .._common.layouts import interleave_qkv
from .kernels import apb_diffusion_forward as triton_apb_forward
from .ops import gated_output_projection_blc

_TRITON_GLOBAL_FUSED_MAX_LENGTH = 64
_TRITON_GLOBAL_MAX_LENGTH = 608
_TRITON_LOCAL_TRUE_EPILOGUE_MIN_WINDOWS = 80
_TRITON_CONFIDENCE_EAGER_MAX_LENGTH = 1504
_TRITON_CONFIDENCE_FUSED_MAX_LENGTH = 1600


def _triton_apb_dispatch(
    call_site: str,
    query_length: int,
    *,
    local_windows: int | None = None,
) -> tuple[str, bool]:
    """Return ``(attention mode, true epilogue)`` for a KFold APB call site."""
    if call_site == "diffusion_global":
        if query_length <= _TRITON_GLOBAL_FUSED_MAX_LENGTH:
            return "fused", False
        if query_length <= _TRITON_GLOBAL_MAX_LENGTH:
            return "split", False
        return "cueq_packed_strided", False
    if call_site == "diffusion_local":
        if local_windows is None:
            raise ValueError("diffusion local APB dispatch requires local_windows")
        return "fused", local_windows >= _TRITON_LOCAL_TRUE_EPILOGUE_MIN_WINDOWS
    if call_site == "confidence":
        if query_length <= _TRITON_CONFIDENCE_EAGER_MAX_LENGTH:
            return "fused", False
        if query_length <= _TRITON_CONFIDENCE_FUSED_MAX_LENGTH:
            return "fused", True
        return "split", False
    raise ValueError(f"Unknown Triton APB call site: {call_site!r}")


def _triton_parameter_key(module, dtype: torch.dtype) -> tuple:
    parameters = (
        module.linear_q.weight,
        module.linear_q.bias,
        module.linear_k.weight,
        module.linear_k.bias,
        module.linear_v.weight,
        module.linear_v.bias,
        module.linear_g.weight,
        module.linear_out.weight,
    )
    return (
        module.linear_q.weight.device.type,
        module.linear_q.weight.device.index,
        dtype,
        module.num_heads,
        module.head_dim,
        *(
            value
            for parameter in parameters
            if parameter is not None
            for value in (
                parameter.data_ptr(),
                None if torch.is_inference(parameter) else parameter._version,
            )
        ),
    )


def _triton_packed_weights(module, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Pack APB weights once and invalidate the cache after parameter changes."""
    if interleave_qkv is None:
        raise ImportError("triton is required for the Triton APB backend")
    cache_key = _triton_parameter_key(module, dtype)
    if getattr(module, "_triton_apb_weight_cache_key", None) == cache_key:
        return module._triton_apb_weight_cache

    device = module.linear_q.weight.device

    def matmul_weight(linear) -> torch.Tensor:
        return linear.weight.t().to(device=device, dtype=dtype).contiguous()

    wq = matmul_weight(module.linear_q)
    wk = matmul_weight(module.linear_k)
    wv = matmul_weight(module.linear_v)
    w_qkv = interleave_qkv(wq, wk, wv, module.num_heads, module.head_dim)
    packed = w_qkv.view(
        module.channel_a,
        module.num_heads,
        3,
        module.head_dim,
    )
    bq = module.linear_q.bias
    if bq is None:
        bq = torch.zeros(module.channel_a, device=device, dtype=dtype)
    else:
        bq = bq.to(device=device, dtype=dtype).contiguous()
    packed_linear_weight = (
        torch.cat(
            (
                module.linear_q.weight,
                module.linear_k.weight,
                module.linear_v.weight,
            ),
            dim=0,
        )
        .to(device=device, dtype=dtype)
        .contiguous()
    )

    def linear_bias(linear) -> torch.Tensor:
        if linear.bias is None:
            return torch.zeros(module.channel_a, device=device, dtype=dtype)
        return linear.bias.to(device=device, dtype=dtype)

    packed_linear_bias = torch.cat(
        (
            linear_bias(module.linear_q),
            linear_bias(module.linear_k),
            linear_bias(module.linear_v),
        )
    ).contiguous()
    cache = {
        "w_qkv": w_qkv,
        "w_qkv_linear": packed_linear_weight,
        "b_qkv_linear": packed_linear_bias,
        "w_q": packed[:, :, 0, :]
        .reshape(module.channel_a, module.channel_a)
        .contiguous(),
        "w_kv": packed[:, :, 1:, :]
        .reshape(module.channel_a, 2 * module.channel_a)
        .contiguous(),
        "bq": bq,
        "wg": matmul_weight(module.linear_g),
        "wo": matmul_weight(module.linear_out),
    }
    module._triton_apb_weight_cache = cache
    module._triton_apb_weight_cache_key = cache_key
    return cache


def _aligned_batch_shape(shape: torch.Size, ndim: int) -> tuple[int, ...]:
    if len(shape) > ndim:
        raise ValueError(f"batch rank {len(shape)} exceeds query batch rank {ndim}")
    return (1,) * (ndim - len(shape)) + tuple(shape)


def _triton_flatten_batches(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[int, ...],
]:
    """Flatten KFold batches with compact bias-sharing entries kept adjacent."""
    if x_q.shape[:-2] != x_k.shape[:-2]:
        raise ValueError(f"query/key batch dims differ: {x_q.shape} vs {x_k.shape}")
    query_batch = tuple(x_q.shape[:-2])
    batch_ndim = len(query_batch)
    lq, lk = x_q.shape[-2], x_k.shape[-2]
    if pair_bias.shape[-2:] != (lq, lk):
        raise ValueError(
            f"pair bias has wrong attention shape: {pair_bias.shape}, expected {lq}x{lk}"
        )

    pair_batch = _aligned_batch_shape(pair_bias.shape[:-3], batch_ndim)
    try:
        broadcast_batch = torch.broadcast_shapes(query_batch, pair_batch)
    except RuntimeError as exc:
        raise ValueError(
            f"pair bias batch dims are not broadcastable: {query_batch} vs {pair_batch}"
        ) from exc
    if tuple(broadcast_batch) != query_batch:
        raise ValueError("pair bias would expand the query batch dimensions")

    context_axes = [
        axis
        for axis, (pair_size, query_size) in enumerate(
            zip(pair_batch, query_batch, strict=True)
        )
        if pair_size == query_size
    ]
    multiplicity_axes = [
        axis
        for axis, (pair_size, query_size) in enumerate(
            zip(pair_batch, query_batch, strict=True)
        )
        if pair_size == 1 and pair_size != query_size
    ]
    ordered_axes = tuple(context_axes + multiplicity_axes)
    if len(ordered_axes) != batch_ndim:
        raise ValueError(
            f"unsupported pair-bias broadcast: query={query_batch}, pair={pair_batch}"
        )

    def reorder(tensor: torch.Tensor, trailing_dims: int) -> torch.Tensor:
        return tensor.permute(
            *ordered_axes,
            *range(batch_ndim, batch_ndim + trailing_dims),
        )

    self_attention = (
        x_q.data_ptr() == x_k.data_ptr()
        and x_q.shape == x_k.shape
        and x_q.stride() == x_k.stride()
    )
    flat_q = reorder(x_q, 2).reshape(-1, lq, x_q.shape[-1]).contiguous()
    if self_attention:
        flat_k = flat_q
    else:
        flat_k = reorder(x_k, 2).reshape(-1, lk, x_k.shape[-1]).contiguous()

    heads = pair_bias.shape[-3]
    pair_bias = pair_bias.reshape(*pair_batch, heads, lq, lk)
    context_count = math.prod(query_batch[axis] for axis in context_axes)
    flat_bias = reorder(pair_bias, 3).reshape(context_count, heads, lq, lk)
    flat_bias = flat_bias.contiguous()

    mask_batch = _aligned_batch_shape(mask.shape[:-1], batch_ndim)
    try:
        mask_broadcast_batch = torch.broadcast_shapes(query_batch, mask_batch)
    except RuntimeError as exc:
        raise ValueError(
            f"mask batch dims are not broadcastable: {query_batch} vs {mask_batch}"
        ) from exc
    if tuple(mask_broadcast_batch) != query_batch or mask.shape[-1] != lk:
        raise ValueError(f"mask would expand or has wrong key length: {mask.shape}")

    compact_mask = all(mask_batch[axis] == 1 for axis in multiplicity_axes)
    if compact_mask:
        mask_target_batch = tuple(
            query_batch[axis] if axis in context_axes else 1 for axis in range(batch_ndim)
        )
    else:
        mask_target_batch = query_batch
    flat_mask = mask.reshape(*mask_batch, lk).expand(*mask_target_batch, lk)
    flat_mask = reorder(flat_mask, 1).reshape(-1, lk).contiguous()

    ordered_batch = tuple(query_batch[axis] for axis in ordered_axes)
    return (
        flat_q,
        flat_k,
        flat_bias,
        flat_mask,
        ordered_axes,
        ordered_batch,
    )


def _triton_restore_batches(
    output: torch.Tensor,
    original_shape: torch.Size,
    ordered_axes: tuple[int, ...],
    ordered_batch: tuple[int, ...],
) -> torch.Tensor:
    batch_ndim = len(ordered_axes)
    inverse_order = tuple(ordered_axes.index(axis) for axis in range(batch_ndim))
    output = output.reshape(*ordered_batch, *original_shape[-2:])
    return output.permute(
        *inverse_order,
        batch_ndim,
        batch_ndim + 1,
    ).reshape(original_shape)


def _packed_strided_qkv(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_bias: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project QKV in one GEMM and return strided BHLD views."""
    qkv = F.linear(x, packed_weight, packed_bias)
    qkv = qkv.view(*x.shape[:-1], 3, num_heads, head_dim)
    return tuple(qkv[..., index, :, :].movedim(-2, -3) for index in range(3))


def _cueq_packed_strided_attention_pair_bias(
    module,
    x: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Keep cuEq bias/SDPA while removing three QKV GEMMs and layout copies."""
    if cueq_attention_pair_bias_mask is None:
        raise ImportError("cuequivariance_ops_torch is required for global APB")
    if pair_bias.shape[0] < 1 or x.shape[0] % pair_bias.shape[0] != 0:
        raise ValueError(
            "global APB packed path requires an integer bias multiplicity: "
            f"x={x.shape}, pair_bias={pair_bias.shape}"
        )
    if mask.shape[0] != pair_bias.shape[0]:
        raise ValueError(
            "global APB packed path requires a compact mask per bias batch: "
            f"mask={mask.shape}, pair_bias={pair_bias.shape}"
        )

    q, k, v = _packed_strided_qkv(
        x,
        weights["w_qkv_linear"],
        weights["b_qkv_linear"],
        num_heads=module.num_heads,
        head_dim=module.head_dim,
    )
    dense_bias, _ = cueq_attention_pair_bias_mask(
        pair_bias,
        mask,
        None,
        None,
        None,
        None,
        num_heads=module.num_heads,
        multiplicity=x.shape[0] // pair_bias.shape[0],
        inf=module.inf,
        return_z_proj=False,
        is_cached_z_proj=True,
    )
    with torch.nn.attention.sdpa_kernel(
        backends=[
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        ],
        set_priority=True,
    ):
        attention_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=dense_bias,
            scale=module.head_dim**-0.5,
        )
    attention_out = attention_out.movedim(-3, -2).contiguous().reshape_as(x)
    return _attention_epilogue(module, x, attention_out)


@torch.compiler.disable
def triton_attention_pair_bias(
    module,
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    *,
    call_site: str,
) -> torch.Tensor:
    """Run the measured Triton APB policy and restore KFold's batch layout."""
    if torch.is_grad_enabled():
        raise RuntimeError("The Triton APB backend is inference-only.")
    if not x_q.is_cuda:
        raise RuntimeError("Triton APB requires CUDA tensors")

    local_windows = pair_bias.shape[-4] if call_site == "diffusion_local" else None
    mode, true_epilogue = _triton_apb_dispatch(
        call_site,
        x_q.shape[-2],
        local_windows=local_windows,
    )
    if mode == "cueq_packed_strided" and cueq_attention_pair_bias_mask is None:
        mode = "split"
    if torch.is_autocast_enabled(x_q.device.type):
        compute_dtype = torch.get_autocast_dtype(x_q.device.type)
    else:
        compute_dtype = x_q.dtype
    original_shape = x_q.shape
    x_q = x_q.to(compute_dtype)
    x_k = x_k.to(compute_dtype)
    pair_bias = pair_bias.to(compute_dtype)
    (
        flat_q,
        flat_k,
        flat_bias,
        flat_mask,
        ordered_axes,
        ordered_batch,
    ) = _triton_flatten_batches(x_q, x_k, pair_bias, mask)
    weights = _triton_packed_weights(module, compute_dtype)
    if mode == "cueq_packed_strided":
        if flat_q.data_ptr() != flat_k.data_ptr():
            raise ValueError("global APB packed path requires self attention")
        output = _cueq_packed_strided_attention_pair_bias(
            module,
            flat_q,
            flat_bias,
            flat_mask,
            weights,
        )
        return _triton_restore_batches(
            output,
            original_shape,
            ordered_axes,
            ordered_batch,
        )

    attention_out = torch.empty_like(flat_q)
    triton_apb_forward(
        flat_q,
        flat_k,
        weights["w_qkv"],
        weights["bq"],
        flat_bias,
        flat_mask,
        attention_out,
        scale=module.head_dim**-0.5,
        inf=module.inf,
        mode=mode,
        fp32_dot_precision="ieee",
        split_w_q=weights["w_q"] if flat_q.data_ptr() != flat_k.data_ptr() else None,
        split_w_kv=weights["w_kv"] if flat_q.data_ptr() != flat_k.data_ptr() else None,
    )

    use_true_epilogue = (
        true_epilogue and module.linear_g.bias is None and module.linear_out.bias is None
    )
    if use_true_epilogue:
        output = gated_output_projection_blc(
            flat_q,
            attention_out,
            weights["wg"],
            weights["wo"],
            fp32_dot_precision="ieee",
        )
    else:
        output = _attention_epilogue(module, flat_q, attention_out)
    return _triton_restore_batches(
        output,
        original_shape,
        ordered_axes,
        ordered_batch,
    )


def _attention_epilogue(module, s, attention_out):
    """Apply the unfused APB gate and output projection under caller autocast."""
    gate = torch.sigmoid(F.linear(s, module.linear_g.weight, module.linear_g.bias))
    return F.linear(
        gate * attention_out.to(gate.dtype),
        module.linear_out.weight,
        module.linear_out.bias,
    )
