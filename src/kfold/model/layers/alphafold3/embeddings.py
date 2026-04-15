import math

import torch
import torch.nn.functional as F
from torch import nn

from kfold.data.types.model_input import FoldingInput


class RelativePositionEncoding(nn.Module):
    """Relative position encoder.
    NOTE: Differ to AlphaFold3 official algorithm, its official algorithm does
    not pass linear projection layer here.
    """

    def __init__(self, r_max: int = 32, s_max: int = 2):
        """Initialize the relative position encoder.

        Parameters
        ----------
        channel_z : int
            The pair representation dimension.
        r_max : int, optional
            The maximum index distance, by default 32.
        s_max : int, optional
            The maximum chain distance, by default 2.

        """
        super().__init__()
        self.r_max: int = r_max
        self.s_max: int = s_max
        self.dimension: int = 4 * (r_max + 1) + 2 * (s_max + 1) + 1

    def forward(
        self, f_input: FoldingInput, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """See Section 3.1.2 Algorithm 3: Relative position encoding in the AF3 paper.
        NOTE: Differ to AlphaFold3 official algorithm, its official algorithm does
        not pass linear projection layer here.
        """
        with torch.no_grad():
            return self.get_relative_position_encoding(f_input, dtype)

    def get_relative_position_encoding(
        self, f_input: FoldingInput, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        # All shape: [B, Lt]
        asym_id = f_input.token.asym_id
        entity_id = f_input.token.entity_id
        sym_id = f_input.token.sym_id
        residue_index = f_input.token.residue_index
        token_index = f_input.token.token_index

        # Line 1
        b_same_chain = torch.eq(asym_id[:, :, None], asym_id[:, None, :])
        # Line 2
        b_same_residue = torch.eq(residue_index[:, :, None], residue_index[:, None, :])
        # Line 3
        b_same_entity = torch.eq(entity_id[:, :, None], entity_id[:, None, :])

        # Line 4
        d_residue = torch.clip(
            residue_index[:, :, None] - residue_index[:, None, :] + self.r_max,
            min=0,
            max=2 * self.r_max,
        )
        d_residue = torch.where(
            b_same_chain,
            d_residue,
            2 * self.r_max + 1,
        )
        # Line 5
        a_rel_pos = F.one_hot(d_residue, 2 * self.r_max + 2).to(dtype)

        # Line 6
        d_token = torch.clip(
            token_index[:, :, None] - token_index[:, None, :] + self.r_max,
            min=0,
            max=2 * self.r_max,
        )
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            2 * self.r_max + 1,
        )
        # Line 7
        a_rel_token = F.one_hot(d_token, 2 * self.r_max + 2).to(dtype)

        # Line 8
        d_chain = torch.clip(
            sym_id[:, :, None] - sym_id[:, None, :] + self.s_max,
            min=0,
            max=2 * self.s_max,
        )
        # NOTE: (seonghwanseo) In the original paper and Boltz implementation,
        # it is written as b_same_chain.
        # However, it is implemented as b_same_entity according to AF3 official
        # implementation.
        d_chain = torch.where(
            b_same_entity,
            d_chain,
            2 * self.s_max + 1,
        )
        # Line 9
        a_rel_chain = F.one_hot(d_chain, 2 * self.s_max + 2).to(dtype)

        # Line 10 (concat)
        rel_position_encoding = torch.cat(
            [
                a_rel_pos,
                a_rel_token,
                b_same_entity.to(dtype).unsqueeze(-1),
                a_rel_chain,
            ],
            dim=-1,
        )
        return rel_position_encoding  # [B, L, L, D]


class FourierEmbedding(nn.Module):
    """Fourier embedding layer.
    Section 3.7 Algorithm 22 Fourier Embedding
    """

    def __init__(self, channel: int):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        channel : int
            The fourier embedding dimension.
        seed : int, optional
            The random seed, by default 42

        """
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(42)

        # Line 1: Randomly generate weight/bias once before training
        w = torch.randn(size=(1, channel), generator=generator)
        b = torch.randn(size=(1, channel), generator=generator)
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("b", b, persistent=False)

    def forward(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        See Section 3.7 Algorithm 22 of AlphaFold3 paper.

        Parameters
        ----------
        t_hat : torch.Tensor
            The input noise level. Shape (B, N,)

        Returns
        -------
        torch.Tensor
            The Fourier embeddings. Shape (B, N, channel)
        """
        # Line 2
        return torch.cos((2 * math.pi) * t_hat[..., None] * self.w + self.b)
