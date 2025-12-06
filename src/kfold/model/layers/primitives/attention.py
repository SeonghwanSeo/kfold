import torch

from .utils import add, div


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
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
    inplace : bool
        Whether to perform operations in-place, default False

    Returns
    -------
    out : torch.Tensor
        The output tensor of shape (..., Q, C)
    """

    with torch.autocast("cuda", dtype=torch.float32):
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        bias = bias.to(torch.float32) if bias is not None else None

        if scale is not None:
            query = div(query, scale, inplace=inplace)

        # Compute attention weights
        attn = torch.einsum("...qc,...kc->...qk", query, key)

        # Add attention bias
        if bias is not None:
            attn = add(attn, bias, inplace=inplace)

        # Softmax
        attn = attn.softmax(dim=-1)

    # Compute output
    out = torch.einsum("...qk,...kc->...qc", attn.to(value.dtype), value)

    return out
