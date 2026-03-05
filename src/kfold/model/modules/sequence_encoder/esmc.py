from __future__ import annotations

import torch
import torch.nn as nn

from kfold.constants.sequence import MASK_TOKEN_INDEX
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.esm.esmc import RegressionHead, TransformerStack
from kfold.utils.registry import SEQUENCE_ENCODER

from .base import BaseSequenceEncoder


@SEQUENCE_ENCODER.register()
class ESMC(BaseSequenceEncoder):
    class Config(BaseSequenceEncoder.Config):
        """Configuration for ESMC sequence encoder.

        Attributes
        ----------
        path: str
            Path to pretrained weights.
        d_model: int
            Dimension of token embeddings and transformer hidden states.
        n_heads: int
            Number of attention heads in the transformer.
        n_layers: int
            Number of transformer layers.
        return_attn: bool
            Whether to return attention weights from the transformer.

        """

        path: str  # Path to pretrained weights.
        d_model: int = 1152
        n_heads: int = 18
        n_layers: int = 36
        return_attn: bool = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg: ESMC.Config = cfg
        self.return_attn: bool = cfg.return_attn

        # Create model components
        self.embed = nn.Embedding(64, cfg.d_model)
        self.transformer = TransformerStack(
            cfg.d_model, cfg.n_heads, cfg.n_layers, return_attn=cfg.return_attn
        )
        self.sequence_head = RegressionHead(cfg.d_model, 64)

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

        # Freeze parameters since we are only doing inference.
        for param in self.parameters():
            param.requires_grad = False

        # NOTE: Inspired by AF3's MSA sampling, we can mask out some tokens to introduce
        # stochasticity during inference. This can be used to generate multiple diverse
        # predictions for the same input by adjusting the evolutionary signal.
        self.mask_token_id: int = MASK_TOKEN_INDEX

    @property
    def d_attn(self) -> int:
        cfg = self.cfg
        if not cfg.return_attn:
            return 0
        else:
            return self.cfg.n_heads * self.cfg.n_layers

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
            torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16),
            torch.no_grad(),
        ):
            if self.return_attn:
                return self.forward_attn(f_input)
            else:
                return self.forward_no_attn(f_input), None

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

    def forward_no_attn(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations.
        """
        input_ids = f_input.sequence.seq_token_id
        seq_id = f_input.sequence.entity_id
        pos_id = f_input.sequence.pos_id
        mlm_mask = f_input.sequence.mlm_mask

        # MLM masking
        input_ids = input_ids.masked_fill(mlm_mask, self.mask_token_id)

        x = self.embed(input_ids)
        for b in self.transformer.blocks:
            x, _ = b(x, seq_id, pos_id)

        # sequence -> token index mapping
        batch_index = torch.arange(x.shape[0], device=x.device)[:, None]
        seq_token_idx = f_input.token.seq_token_index
        x = x[batch_index, seq_token_idx]

        x = self.transformer.norm(x).to(torch.bfloat16)

        mask = self.prepare_emb_mask(f_input)
        x.masked_fill_(~mask[:, :, None], 0.0)
        return x

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
        # NOTE: padding tokens have seq_id=-1, which will be masked out in
        # attention computation. (entity_id is 1-indexed for valid tokens)
        input_ids = f_input.sequence.seq_token_id
        seq_id = f_input.sequence.entity_id
        pos_id = f_input.sequence.pos_id
        mlm_mask = f_input.sequence.mlm_mask

        # sequence -> token index mapping
        seq_token_i = f_input.token.seq_token_index
        B, L = seq_token_i.shape

        # MLM masking
        input_ids = input_ids.masked_fill(mlm_mask, self.mask_token_id)

        # Initialize output
        x_out = torch.empty(
            (B, seq_token_i.shape[1], self.cfg.d_model),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )
        attn_out = torch.empty(
            (B, seq_token_i.shape[1], seq_token_i.shape[1], self.d_attn),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )

        x = self.embed(input_ids)
        # NOTE: While for loop is inefficient in PyTorch, it is fine since our batch size
        # is very small (often 1)

        for i in range(B):
            _x = x[i]  # [seq_len, d_model]
            _seq_id = seq_id[i]  # [seq_len]
            _pos_id = pos_id[i]  # [seq_len]
            _seq_token_i = seq_token_i[i]  # [n_tokens]

            _attn_list: list[torch.Tensor] = []
            for j, block in enumerate(self.transformer.blocks):
                _x, _attn_i = block(_x, _seq_id, _pos_id)

                # Insert attention weights for this layer.
                # [n_heads, seq_len, seq_len] -> [n_heads, n_tokens, n_tokens]
                h_st, h_end = j * self.cfg.n_heads, (j + 1) * self.cfg.n_heads
                _attn_i = _attn_i[:, _seq_token_i[:, None], _seq_token_i[None, :]]
                _attn_i = _attn_i.permute(1, 2, 0)  # [n_tokens, n_tokens, n_heads]
                attn_out[i, :, :, h_st:h_end] = _attn_i

            x_out[i] = _x[_seq_token_i]

        # normalize
        x_out = self.transformer.norm(x_out).to(torch.bfloat16)

        # mask out invalid tokens
        token_mask = self.prepare_emb_mask(f_input)
        attn_mask = self.prepare_out_attn_mask(f_input, token_mask)

        x_out.masked_fill_(~token_mask[..., None], 0.0)
        attn_out.masked_fill_(~attn_mask[..., None], 0.0)
        return x_out, attn_out
