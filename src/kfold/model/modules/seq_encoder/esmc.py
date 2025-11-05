"""ESMC model, copyright: evolutionary-scale."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn

from kfold.utils.registry import SEQUNECE_ENCODER, BaseConfig

from .base import BaseSequenceEncoder

try:
    from flash_attn.bert_padding import pad_input, unpad_input  # type:ignore

    is_flash_attn_available = True
except ImportError:
    pad_input = None
    unpad_input = None
    is_flash_attn_available = False


@dataclass
class ESMCConfig(BaseConfig):
    model_name: str = "ESMC_300M"
    d_model: int = 960
    n_heads: int = 15
    n_layers: int = 30
    use_flash_attn: bool = False
    load_pretrained: bool = True


@SEQUNECE_ENCODER.register(config_cls=ESMCConfig)
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

        if cfg.load_pretrained:
            # Load pretrained weights
            model = load_local_model(cfg.model_name, device=torch.device("cpu"))
            del model.sequence_head  # remove the head to avoid size mismatch
            self.load_state_dict(model.state_dict(), strict=True)

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
        chain_id : torch.Tensor
            Tensor of shape (B, L) containing chain idx.
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

        # TODO: remap token_ids from our vocab to ESMC vocab.
        warnings.warn(
            "(seonghwanseo) I did not remap token ids to ESMC vocab."
            " Make sure your token ids are compatible with ESMC vocab.",
            UserWarning,
            stacklevel=2,
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
