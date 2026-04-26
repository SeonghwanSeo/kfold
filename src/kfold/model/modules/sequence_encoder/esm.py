from __future__ import annotations

import torch
import torch.nn as nn

from kfold.constants.sequence import MASK_TOKEN_INDEX
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.esm.esmc import RegressionHead, TransformerStack
from kfold.utils.registry import SEQUENCE_ENCODER

from .base import BaseSequenceEncoder


@SEQUENCE_ENCODER.register()
class ESMO(BaseSequenceEncoder):
    class Config(BaseSequenceEncoder.Config):
        """Configuration for ESM-O sequence encoder.

        Attributes
        ----------
        path: str
            Path to pretrained weights.
        vocab_size: int
            Size of the input token vocabulary.
        d_model: int
            Dimension of model hidden states and embeddings.
        n_heads: int
            Number of attention heads in the transformer.
        n_layers: int
            Number of transformer layers.
        """

        path: str  # Path to pretrained weights.
        vocab_size: int = 64
        d_model: int = 1152
        n_heads: int = 18
        n_layers: int = 36

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg: ESMO.Config = cfg

        # Create model components
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.transformer = TransformerStack(cfg.d_model, cfg.n_heads, cfg.n_layers)
        self.sequence_head = RegressionHead(cfg.d_model, cfg.vocab_size)

        # Load pretrained weights
        state_dict = torch.load(cfg.path, map_location="cpu")
        self.load_state_dict(state_dict)
        del state_dict

        # Remove sequence head since we only need sequence representations.
        del self.sequence_head

        # Convert to bfloat16
        self.embed = self.embed.to(torch.bfloat16)
        self.transformer = self.transformer.to(torch.bfloat16)

        # Set to eval mode
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        # NOTE (Seonghwan): Inspired by AF3's MSA sampling, we can mask out some
        # tokens to introduce stochasticity during inference. This can be used to
        # generate multiple diverse predictions for the same input by adjusting
        # the evolutionary signal.
        self.mask_token_id: int = MASK_TOKEN_INDEX

    @property
    def n_layers(self) -> int:
        return self.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def n_heads(self) -> int:
        return self.cfg.n_heads

    def forward(self, f_input: FoldingInput) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations,
            where Ntoken is the number of tokens and D is the model dimension.
        attention: torch.Tensor | None
            Tensor of shape (B, Ntoken, Ntoken, N*H) containing attention weights,
            where N is number of layers and H is number of heads.
        """
        # NOTE: ESMC uses bfloat16 for inference.
        with (
            torch.autocast(f_input.device.type, dtype=torch.bfloat16),
            torch.no_grad(),
        ):
            return self.forward_attn(f_input)

    def prepare_emb_mask(self, f_input: FoldingInput) -> torch.Tensor:
        """Prepare output mask"""
        return f_input.token.pad_mask & f_input.token.is_protein

    def prepare_out_attn_mask(
        self, f_input: FoldingInput, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """Prepare output attention mask"""
        # attention mask: [B, Ntoken, Ntoken]
        attn_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
        # mask out attention between different chains
        asym_id = f_input.token.asym_id
        attn_mask &= asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)
        return attn_mask

    def forward_attn(self, f_input: FoldingInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations.
        attention: torch.Tensor
            Tensor of shape (B, Ntoken, Ntoken, N*H) containing attention weights,
            where N is number of layers and H is number of heads.
        """
        dtype, device = torch.bfloat16, f_input.device
        N, H = self.n_layers, self.n_heads

        # NOTE: padding tokens have seq_id=-1, which will be masked out in
        # attention computation. (entity_id is 1-indexed for valid tokens)
        input_ids = f_input.sequence.seq_token_id
        seq_id = f_input.sequence.entity_id
        pos_id = f_input.sequence.pos_id
        mlm_mask = f_input.sequence.mlm_mask

        # sequence -> token index mapping
        seq_token_idx = f_input.token.seq_token_index
        B, L = seq_token_idx.shape

        # === MLM masking === #
        input_ids = input_ids.masked_fill(mlm_mask, self.mask_token_id)

        # === Forward pass === #
        x = self.embed(input_ids)

        b_idcs = torch.arange(B, device=device)
        attn_out = torch.empty((B, L, L, N, H), dtype=dtype, device=device)
        for i, block in enumerate(self.transformer.blocks):
            x, attn_weights = block(x, seq_id, pos_id)

            # [B, n_heads, seq_len, seq_len] -> [B, n_heads, n_tokens, n_tokens]
            _attn = attn_weights[
                b_idcs[:, None, None],  # [B, 1, 1]
                :,
                seq_token_idx[:, :, None],  # [B, 1, n_tokens]
                seq_token_idx[:, None, :],  # [B, n_tokens, 1]
            ]
            attn_out[:, :, :, i, :] = _attn
            del attn_weights

        x = x[b_idcs[:, None], seq_token_idx]  # [B, n_tokens, d_model]
        x_out = self.transformer.norm(x).to(dtype)
        del x

        # Flatten attention output to shape [B, Ntoken, Ntoken, N*H]
        attn_out = attn_out.view(B, L, L, N * H)

        # Mask out invalid tokens
        token_mask = self.prepare_emb_mask(f_input)
        attn_mask = self.prepare_out_attn_mask(f_input, token_mask)
        x_out.masked_fill_(~token_mask[..., None], 0.0)
        attn_out.masked_fill_(~attn_mask[..., None], 0.0)

        return x_out, attn_out
