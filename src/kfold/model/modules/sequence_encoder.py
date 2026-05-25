from __future__ import annotations

import torch

from kfold.constants.sequence import MASK_TOKEN_INDEX
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.seq_enc.transformer_stack import TransformerStack
from kfold.utils.registry import SEQUENCE_ENCODER, BaseConfig


@SEQUENCE_ENCODER.register()
class SequenceEncoder(torch.nn.Module):
    class Config(BaseConfig):
        """Configuration for ESM-C sequence encoder.

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
        super().__init__()
        self.cfg: SequenceEncoder.Config = cfg

        # Create model components
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.transformer = TransformerStack(cfg.d_model, cfg.n_heads, cfg.n_layers)

        # Load pretrained weights
        state_dict = torch.load(cfg.path, map_location="cpu")
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("sequence_head")
        }
        self.load_state_dict(state_dict)
        del state_dict

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

    def forward(self, f_input: FoldingInput) -> tuple[torch.Tensor, torch.Tensor]:
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
        attention: torch.Tensor
            Tensor of shape (B, Ntoken, Ntoken, N, H) containing attention weights,
            where N is number of layers and H is number of heads.
        """
        # NOTE: ESMC uses bfloat16 for inference.
        with (
            torch.autocast(f_input.device.type, dtype=torch.bfloat16),
            torch.no_grad(),
        ):
            return self._forward(f_input)

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

    def _forward(self, f_input: FoldingInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations,
            where D is the model dimension.
        attention: torch.Tensor
            Tensor of shape (B, Ntoken, Ntoken, N, H) containing attention weights,
            where N is number of layers and H is number of heads.
        """
        dtype, device = torch.bfloat16, f_input.device
        N, H, D = self.n_layers, self.n_heads, self.d_model

        # NOTE: padding tokens have seq_id=-1, which will be masked out in
        # attention computation. (entity_id is 1-indexed for valid tokens)
        input_ids = f_input.sequence.seq_token_id
        seq_id = f_input.sequence.asym_id
        pos_id = f_input.sequence.pos_id
        mlm_mask = f_input.sequence.mlm_mask

        # sequence -> token index mapping
        seq_token_idx = f_input.token.seq_token_index.clamp(min=0)  # [B, n_tokens]
        B, Ntoken = seq_token_idx.shape
        b_idx = torch.arange(B, device=device)[:, None, None]
        row_idx = seq_token_idx[:, :, None]
        col_idx = seq_token_idx[:, None, :]

        # === MLM masking === #
        input_ids = input_ids.masked_fill(mlm_mask, self.mask_token_id)

        # === Forward pass === #
        x = self.embed(input_ids)
        attn_out = torch.empty((B, Ntoken, Ntoken, N, H), dtype=dtype, device=device)

        for i, block in enumerate(self.transformer.blocks):
            x, attn_weights = block(x, seq_id, pos_id)
            # [B, n_heads, seq_len, seq_len] -> [B, n_tokens, n_tokens, n_heads]
            _attn = attn_weights.permute(0, 2, 3, 1)  # [B, seq_len, seq_len, n_heads]
            _attn = _attn[b_idx, row_idx, col_idx]
            attn_out[:, :, :, i, :] = _attn.to(dtype)
            del attn_weights
        x_out = x.gather(1, seq_token_idx[..., None].expand(-1, -1, D))

        # Mask out invalid tokens
        token_mask = self.prepare_emb_mask(f_input)
        attn_mask = self.prepare_out_attn_mask(f_input, token_mask)
        x_out.masked_fill_(~token_mask[..., None], 0.0)
        attn_out.masked_fill_(~attn_mask[..., None, None], 0.0)

        return x_out, attn_out
