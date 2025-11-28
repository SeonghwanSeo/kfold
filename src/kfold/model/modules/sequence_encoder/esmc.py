"""ESMC model, copyright: evolutionary-scale."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn

from kfold.utils.registry import SEQUENCE_ENCODER, BaseConfig

from .base import BaseSequenceEncoder

try:
    from flash_attn.bert_padding import pad_input, unpad_input  # type:ignore

    is_flash_attn_available = True
except ImportError:
    pad_input = None
    unpad_input = None
    is_flash_attn_available = False

ESM_PARAMS = {
    "esmc_300m": {
        "d_model": 960,
        "n_heads": 15,
        "n_layers": 30,
    },
    "esmc_600m": {
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
    },
}


class ESMCConfig(BaseConfig):
    model_name: str = "esmc_300m"  # or "esmc_600m"
    d_model: int = 960
    n_heads: int = 15
    n_layers: int = 30
    use_flash_attn: bool = False
    load_pretrained: bool = True

    @classmethod
    def from_model_name(cls, model_name: str, **kwargs) -> ESMCConfig:
        if model_name not in ESM_PARAMS:
            raise ValueError(f"Unknown model_name: {model_name}")
        params = ESM_PARAMS[model_name]
        return cls(model_name=model_name, **params, **kwargs)


@SEQUENCE_ENCODER.register(config_cls=ESMCConfig)
class ESMC(BaseSequenceEncoder):
    def __init__(self, cfg: ESMCConfig):
        super().__init__(cfg)

        # Lazy import to avoid unnecessary dependency if not used.
        from esm.layers.transformer_stack import TransformerStack
        from esm.pretrained import load_local_model

        self.embed = nn.Embedding(64, cfg.d_model)
        self._use_flash_attn = is_flash_attn_available and cfg.use_flash_attn
        self.transformer = TransformerStack(
            cfg.d_model,
            cfg.n_heads,
            None,
            cfg.n_layers,
            n_layers_geom=0,
        )
        self.eval()

        if cfg.load_pretrained:
            # Load pretrained weights
            model = load_local_model(cfg.model_name, device=torch.device("cpu"))
            del model.sequence_head  # remove the head to avoid size mismatch
            self.load_state_dict(model.state_dict(), strict=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_id: torch.Tensor,
        chain_id: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        sequence_tokens : torch.Tensor
            Tensor of shape (B, L) containing sequence tokens.
        sequence_id : torch.Tensor
            Tensor of shape (B, L) containing sequence idx.
        chain_ids : torch.Tensor
            Tensor of shape (B, L) containing chain ids.
        return_attention : bool, optional
            Whether to return attention weights. Default is False.

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, L, D) containing sequence representations.
        attention: torch.Tensor | None
            Tensor of shape (B, N, H, L, L) containing attention weights,
            where N is number of layers and H is number of heads.
        """
        if return_attention:
            raise NotImplementedError(
                "Attention weights are not implemented in this module."
            )

        x = self.embed(sequence_tokens)

        # If sequence_id looks like a mask.
        B, L = x.shape[:2]
        if self._use_flash_attn:
            assert sequence_id.dtype == torch.bool, (
                "sequence_id must be a boolean mask if Flash Attention is used"
            )
            assert sequence_id.shape == (B, L)
            assert unpad_input is not None
            x, indices, *_ = unpad_input(  # type: ignore
                x, sequence_id
            )
        else:
            indices = None

        x, _, _ = self.transformer(x, sequence_id=sequence_id)

        if self._use_flash_attn:
            assert indices is not None
            assert pad_input is not None
            x = pad_input(x, indices, B, L)  # Back to [B, L, D]

        return x, None

    @staticmethod
    @lru_cache
    def _get_token_to_id() -> dict[str, int]:
        # fmt: off
        SEQUENCE_VOCAB = [
            "<cls>", "<pad>", "<eos>", "<unk>",
            "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
            "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
            "O", ".", "-", "|",
            "<mask>",
        ]
        # fmt: on
        return {v: i for i, v in enumerate(SEQUENCE_VOCAB)}

    @classmethod
    def _encode_sequence(cls, seq: str) -> list[int]:
        token_to_id = cls._get_token_to_id()
        unk_token = token_to_id["X"]  # Unknown amino acid
        ids = [token_to_id.get(residue, unk_token) for residue in seq]
        return ids

    def encode(
        self,
        sequences: list[str],
        add_special_tokens: bool = True,
        include_special_tokens: bool = False,
    ) -> list[torch.Tensor]:
        """Encode a batch of sequences.

        Parameters
        ----------
        sequences : list[str]
            The sequences to encode.
        add_special_tokens : bool, optional
            Whether to add special tokens (<cls> and <eos>) to the sequences.
        include_special_tokens : bool, optional
            Whether to include special tokens in the output. Default is False.

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, L, D) containing sequence representations.
        """
        token_to_id = self._get_token_to_id()
        cls_token = token_to_id["<cls>"]
        eos_token = token_to_id["<eos>"]
        pad_token = token_to_id["<pad>"]
        unk_token = token_to_id["X"]  # Unknown amino acid

        batch_ids = []
        for seq in sequences:
            ids = [token_to_id.get(residue, unk_token) for residue in seq]
            if add_special_tokens:
                ids = [cls_token] + ids + [eos_token]
            batch_ids.append(ids)

        max_len = max(len(ids) for ids in batch_ids)
        batch_tensor = torch.full((len(batch_ids), max_len), pad_token, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            batch_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        batch_tensor = batch_tensor.to(self.device)
        attention_mask = batch_tensor != pad_token
        sequence_embedding, _ = self.forward(batch_tensor, attention_mask, attention_mask)

        outs: list[torch.Tensor] = []
        for i, ids in enumerate(batch_ids):
            length = len(ids)
            if include_special_tokens:
                outs.append(sequence_embedding[i, :length, :])
            else:
                start = 1 if add_special_tokens else 0
                end = length - 1 if add_special_tokens else length
                outs.append(sequence_embedding[i, start:end, :])
        return outs
