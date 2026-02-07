import torch
import torch.nn as nn

from kfold.model.layers.primitives import LayerNorm, Linear


class PairwiseProdDiff(nn.Module):
    """Convert single embeddings to pairwise embeddings.
    Inspired by ESMFold's implementation.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        assert c_out % 2 == 0, "c_out must be even."
        c_hidden = c_out // 2
        self.linear_in = Linear(c_in, c_hidden * 2, init="default")
        self.linear_out = Linear(c_hidden * 2, c_out, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Compute pairwise embeddings from single representations using
        element-wise differences and products.

        Parameters
        ----------
        s : torch.Tensor
            The single representation (*, L, c_in).

        Returns
        -------
        torch.Tensor
            The output tensor (*, L, L, c_out).
        """
        s_i, s_j = torch.chunk(
            self.linear_in(s), 2, dim=-1
        )  # (*, L, c_hid), (*, L, c_hid)

        s_i = s_i.unsqueeze(-2)  # (*, L, 1, c_hidden)
        s_j = s_j.unsqueeze(-3)  # (*, 1, L, c_hidden)

        # Combine Diff (Asymmetry) and Prod (Correlation)
        # NOTE: summation is derived from production operation with linear bias
        # (W1(s_i) + b1) * (W2(s_j) + b2)
        #   = W1(s_i)W2(s_j) + b1*W2(s_j) + b2*W1(s_i) + b1*b2
        z = torch.cat([s_i - s_j, s_i * s_j], dim=-1)  # (*, L, L, c_hidden * 2)

        z = self.linear_out(z)  # (*, L, L, c_out)
        return z


class PLMModule(nn.Module):
    def __init__(
        self,
        channel_plm: int = 256,
        channel_z: int = 128,
        use_separate_projections: bool = True,
    ) -> None:
        super().__init__()
        self.channel_plm: int = channel_plm
        self.channel_z: int = channel_z
        self.use_separate_projections: bool = use_separate_projections

        self.layernorm = LayerNorm(channel_plm, create_offset=False)
        if self.use_separate_projections:
            self.pairwise_proj_intra = PairwiseProdDiff(channel_plm, channel_z)
            self.pairwise_proj_inter = PairwiseProdDiff(channel_plm, channel_z)
        else:
            self.pairwise_proj = PairwiseProdDiff(channel_plm, channel_z)

    def forward(
        self,
        z: torch.Tensor,
        s_plm: torch.Tensor,
        asym_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pair representations
        s_plm : torch.Tensor
            The sequence embeddings
        asym_id : torch.Tensor
            The asymmetry IDs of shape (B, L)
        mask : torch.Tensor
            The token mask of shape (B, L)

        Returns
        -------
        torch.Tensor
            The updated pair representations
        """
        s_plm = self.layernorm(s_plm)
        if self.use_separate_projections:
            intra_mask = asym_id[..., None] == asym_id[..., None, :]
            intra_mask = intra_mask & (mask[..., None] & mask[..., None, :])
            z = z + self.pairwise_proj_intra(s_plm) * intra_mask[..., None]
            z = z + self.pairwise_proj_inter(s_plm) * (~intra_mask)[..., None]
        else:
            pair_mask = mask[..., None] & mask[..., None, :]
            z = z + self.pairwise_proj(s_plm) * pair_mask[..., None]
        return z
