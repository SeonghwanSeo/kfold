from __future__ import annotations

import torch

from kfold.constants.sequence import MASK_TOKEN_INDEX, PAD_TOKEN_INDEX
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
        chain_type: str
            Type of sequence chain to encode. Must be one of "protein", "dna", or "rna".
        vocab_size: int
            Size of the input token vocabulary.
        d_model: int
            Dimension of model hidden states and embeddings.
        n_heads: int
            Number of attention heads in the transformer.
        n_layers: int
            Number of transformer layers.
        use_moe: bool
            Whether to use Mixture of Experts (MoE) FFN layers instead of dense FFN.
        """

        path: str  # Path to pretrained weights.
        vocab_size: int = 64
        chain_type: str = "protein"
        d_model: int = 1152
        n_heads: int = 18
        n_layers: int = 36
        use_moe: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg: SequenceEncoder.Config = cfg
        self.chain_type = cfg.chain_type

        # Create model components
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.transformer = TransformerStack(
            cfg.d_model, cfg.n_heads, cfg.n_layers, use_moe=cfg.use_moe
        )

        # Convert to bfloat16
        self.embed = self.embed.to(torch.bfloat16)
        self.transformer = self.transformer.to(torch.bfloat16)

        # Load pretrained weights
        state_dict = torch.load(cfg.path, map_location="cpu")
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("sequence_head")
        }
        self.load_state_dict(state_dict)
        del state_dict

        # Set to eval mode
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        # NOTE (Seonghwan): Inspired by AF3's MSA sampling, we can mask out some
        # tokens to introduce stochasticity during inference. This can be used to
        # generate multiple diverse predictions for the same input by adjusting
        # the evolutionary signal.
        self.mask_token_id: int = MASK_TOKEN_INDEX
        self.pad_token_id: int = PAD_TOKEN_INDEX

    def train(self, mode: bool = True):
        """Ensure the module remains in eval mode regardless of parent state."""
        super().train(False)
        return self

    @property
    def n_layers(self) -> int:
        return self.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def n_heads(self) -> int:
        return self.cfg.n_heads

    @property
    def n_attns(self) -> int:
        return self.n_layers * self.n_heads

    def forward(
        self,
        f_input: FoldingInput,
        mask_ratio: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features
        mask_ratio: float
            The ratio of tokens to mask for MLM during inference. Default is 0.15.

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
            return self._forward(f_input, mask_ratio)

    @torch.compiler.disable
    def get_seq_mask(self, f_input: FoldingInput) -> torch.Tensor:
        """Prepare output mask"""
        if self.chain_type == "protein":
            return f_input.sequence.pad_mask & f_input.sequence.is_protein
        elif self.chain_type == "dna":
            return f_input.sequence.pad_mask & f_input.sequence.is_dna
        elif self.chain_type == "rna":
            return f_input.sequence.pad_mask & f_input.sequence.is_rna
        else:
            raise ValueError(f"Unsupported chain type: {self.chain_type}")

    @torch.compiler.disable
    def get_token_mask(self, f_input: FoldingInput) -> torch.Tensor:
        """Prepare output mask"""
        if self.chain_type == "protein":
            return f_input.token.pad_mask & f_input.token.is_protein
        elif self.chain_type == "dna":
            return f_input.token.pad_mask & f_input.token.is_dna
        elif self.chain_type == "rna":
            return f_input.token.pad_mask & f_input.token.is_rna
        else:
            raise ValueError(f"Unsupported chain type: {self.chain_type}")

    def _forward(
        self,
        f_input: FoldingInput,
        mask_ratio: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        # === Mask out invalid sequence tokens === #
        seq_mask = self.get_seq_mask(f_input)
        seq_id = seq_id.masked_fill(~seq_mask, -1)
        input_ids = input_ids.masked_fill(~seq_mask, self.pad_token_id)

        # === MLM masking === #
        mlm_mask = torch.rand(input_ids.shape, device=input_ids.device) < mask_ratio
        mlm_mask &= seq_mask  # Only mask valid tokens
        input_ids = input_ids.masked_fill(mlm_mask, self.mask_token_id)

        # sequence -> token index mapping
        seq_token_idx = f_input.token.seq_token_index.clamp(min=0)  # [B, n_tokens]
        B, Ntoken = seq_token_idx.shape
        b_idx = torch.arange(B, device=device)[:, None, None]
        row_idx = seq_token_idx[:, :, None]
        col_idx = seq_token_idx[:, None, :]

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
        token_mask = self.get_token_mask(f_input)
        x_out.masked_fill_(~token_mask[..., None], 0.0)

        pair_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
        asym_id = f_input.token.asym_id
        pair_mask &= asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)
        attn_out.masked_fill_(~pair_mask[..., None, None], 0.0)

        return x_out, attn_out
