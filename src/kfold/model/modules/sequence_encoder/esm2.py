import torch

from kfold.constants.sequence import MASK_TOKEN_INDEX, PAD_TOKEN_INDEX, UNK_TOKEN_INDEX
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.esm.esm2 import RobertaLMHead, TransformerLayer
from kfold.utils.registry import SEQUENCE_ENCODER

from .base import BaseSequenceEncoder


@SEQUENCE_ENCODER.register()
class ESM2(BaseSequenceEncoder):
    class Config(BaseSequenceEncoder.Config):
        """Configuration for ESM-2 sequence encoder.

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
        vocab_size: int = 33
        d_model: int = 2560
        n_heads: int = 40
        n_layers: int = 36

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg: ESM2.Config = cfg

        self.vocab_size: int = cfg.vocab_size
        self.unk_idx: int = UNK_TOKEN_INDEX
        self.mask_idx: int = MASK_TOKEN_INDEX
        self.pad_idx: int = PAD_TOKEN_INDEX

        self.embed_tokens = torch.nn.Embedding(
            cfg.vocab_size, cfg.d_model, padding_idx=self.pad_idx
        )
        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    expansion_ratio=4,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.emb_layer_norm_after = torch.nn.LayerNorm(cfg.d_model)
        self.lm_head = RobertaLMHead(
            cfg.d_model, cfg.vocab_size, self.embed_tokens.weight
        )

        # Load pretrained weights
        state_dict = torch.load(cfg.path, "cpu", weights_only=False)["model"]
        state_dict = {
            k.removeprefix("encoder.").removeprefix("sentence_encoder."): v
            for k, v in state_dict.items()
        }
        state_dict = {k: v for k, v in state_dict.items() if "inv_freq" not in k}
        self.load_state_dict(state_dict)
        del state_dict

        # Remove sequence head since we only need sequence representations.
        del self.lm_head

        # Set to eval mode
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
        with torch.no_grad():
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
            Tensor of shape (B, Ntoken, N, D) containing sequence representations,
            where N is the number of layers and D is the model dimension.
        attention: torch.Tensor
            Tensor of shape (B, Ntoken, Ntoken, N, H) containing attention weights,
            where N is number of layers and H is number of heads.
        """
        dtype, device = torch.bfloat16, f_input.device
        N, H, D = self.n_layers, self.n_heads, self.d_model

        # NOTE: padding tokens have seq_id=-1, which will be masked out in
        # attention computation. (entity_id is 1-indexed for valid tokens)
        input_ids = f_input.sequence.seq_token_id
        seq_id = f_input.sequence.entity_id
        pos_id = f_input.sequence.pos_id
        mlm_mask = f_input.sequence.mlm_mask

        # sequence -> token index mapping
        seq_token_idx = f_input.token.seq_token_index.clamp(min=0)
        B, Ntoken = seq_token_idx.shape
        b_idx = torch.arange(B, device=device)[:, None, None]
        row_idx = seq_token_idx[:, :, None]
        col_idx = seq_token_idx[:, None, :]

        # === MLM masking ===
        input_ids = input_ids.masked_fill(mlm_mask, self.mask_idx)
        input_ids[~f_input.sequence.is_protein] = self.unk_idx

        # === Forward pass ===
        x = self.embed_tokens(input_ids)
        is_masked = input_ids == self.mask_idx
        is_padding = input_ids == self.pad_idx
        x.masked_fill_(is_masked.unsqueeze(-1), 0.0)

        # Scaling
        mask_ratio_train = 0.15 * 0.8
        mask_ratio_observed = (is_masked.sum(-1) / (~is_padding).sum(-1)).to(x.dtype)
        x *= (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        # === Forward pass === #
        x_out = torch.empty((B, Ntoken, N, D), dtype=dtype, device=device)
        attn_out = torch.empty((B, Ntoken, Ntoken, N, H), dtype=dtype, device=device)
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, seq_id, pos_id)

            # [B, seq_len, d_model] -> [B, n_tokens, d_model]
            _x = x.gather(1, seq_token_idx[..., None].expand(-1, -1, D))
            x_out[:, :, i, :] = _x.to(dtype)

            # [B, n_heads, seq_len, seq_len] -> [B, n_tokens, n_tokens, n_heads]
            _attn = attn_weights.permute(0, 2, 3, 1)  # [B, seq_len, seq_len, n_heads]
            _attn = _attn[b_idx, row_idx, col_idx]
            attn_out[:, :, :, i, :] = _attn.to(dtype)
            del attn_weights

        # Mask out invalid tokens
        token_mask = self.prepare_emb_mask(f_input)
        attn_mask = self.prepare_out_attn_mask(f_input, token_mask)
        x_out.masked_fill_(~token_mask[..., None, None], 0.0)
        attn_out.masked_fill_(~attn_mask[..., None, None], 0.0)

        return x_out, attn_out
