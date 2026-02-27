from __future__ import annotations

import torch
import torch.nn as nn

from kfold.model.layers.esm.esmc import RegressionHead, TransformerStack
from kfold.utils.registry import SEQUENCE_ENCODER, BaseConfig

from .base import BaseSequenceEncoder

ESM_PARAMS = {
    "esmc_600m": {
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
    },
    "esmc_3b": {
        "d_model": 2048,
        "n_heads": 32,
        "n_layers": 60,
    },
}

# fmt: off
AMINO_ACIDS = [
    'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
    'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
    'X', 'B', 'U', 'Z', 'O', '.', '-', '|',
]
VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    *AMINO_ACIDS,
    "<mask>",
]
# fmt: on


class Alphabet:
    def __init__(self):
        self.tokens: list[str] = list(VOCAB)
        self.tok_to_idx: dict[str, int] = {tok: i for i, tok in enumerate(self.tokens)}
        self.unk_idx: int = self.tok_to_idx["<unk>"]
        self.bos_idx: int = self.tok_to_idx["<cls>"]
        self.eos_idx: int = self.tok_to_idx["<eos>"]
        self.pad_idx: int = self.tok_to_idx["<pad>"]
        self.mask_idx: int = self.tok_to_idx["<mask>"]
        self.aa_idxs: list[int] = [
            self.tok_to_idx[tok] for tok in AMINO_ACIDS if tok in self.tok_to_idx
        ]

    def __len__(self):
        return len(self.tokens)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        tok_to_idx_get = self.tok_to_idx.get
        unk = self.unk_idx
        encoded = [tok_to_idx_get(tok, unk) for tok in sequence]
        if add_special_tokens:
            encoded = [self.bos_idx] + encoded + [self.eos_idx]
        return encoded

    def encode_batch(
        self, sequences: list[str], add_special_tokens: bool = True
    ) -> list[list[int]]:
        return [self.encode(seq, add_special_tokens) for seq in sequences]

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)


class ESMCConfig(BaseConfig):
    path: str  # Path to pretrained weights.
    model_name: str = "esmc_600m"
    d_model: int = 1152
    n_heads: int = 18
    n_layers: int = 36

    @classmethod
    def from_model_name(cls, path: str, model_name: str, **kwargs) -> ESMCConfig:
        if model_name not in ESM_PARAMS:
            raise ValueError(f"Unknown model_name: {model_name}")
        params = ESM_PARAMS[model_name]
        return cls(path=path, model_name=model_name, **params, **kwargs)


@SEQUENCE_ENCODER.register(config_cls=ESMCConfig)
class ESMC(BaseSequenceEncoder):
    def __init__(self, cfg: ESMCConfig):
        super().__init__(cfg)
        # Create model components
        self.alphabet = Alphabet()
        self.embed = nn.Embedding(64, cfg.d_model)
        self.transformer = TransformerStack(cfg.d_model, cfg.n_heads, cfg.n_layers)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor of shape (B, L) containing sequence tokens.
        attn_mask: torch.Tensor
            Attention mask of shape (B, L), where True indicates valid tokens.
        pos_id: torch.Tensor
            Position ids of shape (B, L) for rotary positional embeddings.

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, L, D) containing sequence representations.
        attention: torch.Tensor | None
            Tensor of shape (B, N, H, L, L) containing attention weights,
            where N is number of layers and H is number of heads.
        """
        # NOTE: ESMC uses bfloat16 for inference.
        with (
            torch.no_grad(),
            torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16),
        ):
            x = self.embed(input_ids)
            x, attn_list = self.transformer(x, attn_mask, pos_id)
        attn = torch.stack(attn_list, dim=1)  # [B, N, H, L, L]
        return x, attn
