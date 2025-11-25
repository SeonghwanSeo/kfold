import torch

from .utils import add, div


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
    use_high_precision: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    """Compute the attention operation.
    Parameters
    ----------
    query : torch.Tensor
        The query tensor of shape (..., Q, C)
    key : torch.Tensor
        The key tensor of shape (..., K, C)
    value : torch.Tensor
        The value tensor of shape (..., K, C)
    bias : Optional[torch.Tensor]
        The attention bias of shape (..., Q, K), default None
    scale : Optional[float]
        The scaling factor for the query-key dot product, default None
    use_high_precision : bool
        Whether to use high precision (float32) for attention computation, default False
    inplace : bool
        Whether to perform operations in-place, default False

    Returns
    -------
    out : torch.Tensor
        The output tensor of shape (..., Q, C)
    """

    dtype = query.dtype if not use_high_precision else torch.float32

    if scale is not None:
        query = div(query, scale, inplace=inplace)

    with torch.autocast("cuda", dtype=dtype):
        # Compute attention weights
        attn = torch.einsum("...qc,...kc->...qk", query, key)

        # Add attention bias
        if bias is not None:
            attn = add(attn, bias, inplace=inplace)

        # Softmax normalization
        with torch.autocast("cuda", dtype=torch.float32):
            attn = attn.softmax(dim=-1)

    # Compute output
    out = torch.einsum("...qk,...kc->...qc", attn.to(value.dtype), value)

    return out
