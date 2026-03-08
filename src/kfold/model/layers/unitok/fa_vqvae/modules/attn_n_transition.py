"""
Code adopted from La-Proteina (https://github.com/NVIDIA-Digital-Bio/la-proteina).
"""

import torch

from .pair_bias_attn import MultiHeadPairBiasedAttention


class MultiheadAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers
    are their adaptive versions which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or
            sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        dim_token,
        dim_pair,
        nheads,
        residual_mha,
        residual_transition,
        parallel_mha_transition,
        use_qkln,
        dropout=0.0,
    ):
        super().__init__()
        self.parallel = parallel_mha_transition

        # If parallel do not allow both layers to have a residual connection
        # since it leads to adding x twice
        if self.parallel and residual_mha and residual_transition:
            residual_transition = False

        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mhba = MultiHeadPairBiasedAttention(
            dim_token=dim_token,
            dim_pair=dim_pair,
            nheads=nheads,
            use_qkln=use_qkln,
        )

        self.norm = torch.nn.LayerNorm(dim_token)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_token, dim_token * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_token * 2, dim_token),
        )

    def _apply_mha(self, x, pair_rep, mask):
        x_attn = self.mhba(x, pair_rep, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn

    def _apply_transition(self, x):
        x_tr = x + self.mlp(self.norm(x))
        return x_tr

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            mask: binary mask, shape [b, n]
            pair_rep: Pair representation (if provided, if no bias will be ignored),
                shape [b, n, n, dim_pair] or None

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        if self.parallel:
            x = self._apply_mha(x, pair_rep, mask) + self._apply_transition(x)
        else:
            x = self._apply_mha(x, pair_rep, mask)
            x = self._apply_transition(x)
        return x
