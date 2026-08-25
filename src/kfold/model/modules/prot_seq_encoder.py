from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from atlaslm.pretrained import load_model

from kfold.constants.sequence import PAD_TOKEN_INDEX
from kfold.data.types.model_input import FoldingInput
from kfold.utils.config import configurable


@configurable
class ProteinSequenceEncoder(torch.nn.Module):
    @dataclass(kw_only=True)
    class Config:
        """Configuration for the AtlasLM sequence encoder.

        Attributes
        ----------
        cache_dir: str | None
            Directory used to cache weights downloaded from Hugging Face.
        vocab_size: int
            Size of the input token vocabulary.
        d_model: int
            Dimension of model hidden states and embeddings.
        n_heads: int
            Number of attention heads in the transformer.
        n_layers: int
            Number of transformer layers.
        """

        cache_dir: str | None = None
        vocab_size: int = 64
        d_model: int = 2304
        n_heads: int = 36
        n_layers: int = 48

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg: ProteinSequenceEncoder.Config = cfg
        self.lm = load_model(
            model_name="atlaslm-3b-base", cache_dir=cfg.cache_dir, dtype=torch.bfloat16
        )
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @property
    def n_layers(self) -> int:
        return self.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def n_heads(self) -> int:
        return self.cfg.n_heads

    def forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, N, D) containing sequence representations,
            where N is the number of layers and D is the model dimension.
        """
        with (
            torch.autocast(f_input.device.type, dtype=torch.bfloat16),
            torch.no_grad(),
        ):
            return self._forward(f_input)

    def _forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, Nlayer+1, D) containing sequence representations,
            where D is the model dimension.
        """
        # NOTE: padding tokens have seq_id=-1, which will be masked out in
        # attention computation. (entity_id is 1-indexed for valid tokens)
        input_ids = f_input.sequence.seq_token_id
        seq_id = f_input.sequence.asym_id
        pos_id = f_input.sequence.pos_id

        # === Mask out invalid sequence tokens === #
        seq_mask = f_input.sequence.pad_mask & f_input.sequence.is_protein
        seq_id = seq_id.masked_fill(~seq_mask, -1)
        input_ids = input_ids.masked_fill(~seq_mask, PAD_TOKEN_INDEX)

        # === Forward pass === #
        x = self.lm.embed(input_ids)
        x_list = [x]
        scale_factor = math.sqrt(self.n_layers / 36)
        for b in self.lm.transformer.blocks:
            r = b.attn(x, seq_id, pos_id)[0]
            x = x + r / scale_factor
            r = b.ffn(x)
            x = x + r / scale_factor
            x_list.append(x)
        x = torch.stack(x_list, dim=-2)  # [B, Nseq, Nlayer+1, D]

        # sequence -> token index mapping
        seq_token_idx = f_input.token.seq_token_index.clamp(min=0)  # [B, Ntokens]
        seq_token_idx = seq_token_idx[..., None, None].expand(
            -1, -1, self.n_layers + 1, self.d_model
        )
        x_out = x.gather(1, seq_token_idx)

        # Mask out invalid tokens
        token_mask = f_input.token.pad_mask & f_input.token.is_protein
        x_out.masked_fill_(~token_mask[..., None, None], 0.0)
        return x_out
