from __future__ import annotations

from dataclasses import dataclass

import torch

from kfold.constants.sequence import MASK_TOKEN_INDEX, PAD_TOKEN_INDEX
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.seq_enc.transformer_stack import TransformerStack
from kfold.utils.config import configurable


@configurable
class SequenceEncoder(torch.nn.Module):
    @dataclass(kw_only=True)
    class Config:
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
        # NOTE: ESMC uses bfloat16 for inference.
        with (
            torch.autocast(f_input.device.type, dtype=torch.bfloat16),
            torch.no_grad(),
        ):
            return self._forward(f_input)

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
    ) -> torch.Tensor:
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
        seq_mask = self.get_seq_mask(f_input)
        seq_id = seq_id.masked_fill(~seq_mask, -1)
        input_ids = input_ids.masked_fill(~seq_mask, self.pad_token_id)

        # === Forward pass === #
        x = self.embed(input_ids)
        x_list = [x]
        for block in self.transformer.blocks:
            x = block(x, seq_id, pos_id)
            x_list.append(x)
        x = torch.stack(x_list, dim=-2)  # [B, Nseq, Nlayer+1, D]

        # sequence -> token index mapping
        seq_token_idx = f_input.token.seq_token_index.clamp(min=0)  # [B, Ntokens]
        seq_token_idx = seq_token_idx[..., None, None].expand(
            -1, -1, self.n_layers + 1, self.d_model
        )
        x_out = x.gather(1, seq_token_idx)

        # Mask out invalid tokens
        token_mask = self.get_token_mask(f_input)
        x_out.masked_fill_(~token_mask[..., None, None], 0.0)
        return x_out
