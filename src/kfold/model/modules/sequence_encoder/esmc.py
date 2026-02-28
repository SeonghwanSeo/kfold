from __future__ import annotations

import torch
import torch.nn as nn

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.esm.esmc import RegressionHead, TransformerStack
from kfold.utils.registry import SEQUENCE_ENCODER

from .base import BaseSequenceEncoder

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
        return_attn: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg: ESMC.Config = cfg
        self.return_attn: bool = cfg.return_attn

        # Create model components
        self.alphabet = Alphabet()
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
        input_ids = f_input.sequence.input_id
        seq_id = f_input.sequence.entity_id
        pos_id = f_input.sequence.pos_id
        # sequence -> token index mapping
        seq_token_idx = f_input.token.seq_token_index

        with torch.no_grad():
            x = self.embed(input_ids)

            x_list: list[torch.Tensor] = []
            for i in range(f_input.batch_size):
                _x = x[i]  # [seq_len, d_model]
                _seq_id = seq_id[i]  # [seq_len]
                _pos_id = pos_id[i]  # [seq_len]
                _seq_token_idx = seq_token_idx[i]  # [n_tokens]
                for block in self.transformer.blocks:
                    _x, _ = block(_x, _seq_id, _pos_id)  # [seq_len, d_model]
                x_list.append(_x[_seq_token_idx, :])

            # Stack and unpad
            x = torch.stack(x_list, dim=0)

            # normalize
            x = self.transformer.norm(x)

        # mask out invalid tokens
        pad_mask = f_input.token.pad_mask
        # mask out non-protein tokens
        token_mask = pad_mask & f_input.token.is_protein
        x = x * token_mask[..., None]
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
        input_ids = f_input.sequence.input_id
        seq_id = f_input.sequence.entity_id
        pos_id = f_input.sequence.pos_id

        # sequence -> token index mapping
        seq_token_i = f_input.token.seq_token_index
        B, L = seq_token_i.shape

        # Initialize output
        x_list: list[torch.Tensor] = []
        attn_list: list[torch.Tensor] = []

        x = self.embed(input_ids)
        for i in range(B):
            _x = x[i]  # [seq_len, d_model]
            _seq_id = seq_id[i]  # [seq_len]
            _pos_id = pos_id[i]  # [seq_len]
            _seq_token_i = seq_token_i[i]  # [n_tokens]

            _attn_list: list[torch.Tensor] = []
            for block in self.transformer.blocks:
                _x, _attn_i = block(_x, _seq_id, _pos_id)
                # [n_heads, seq_len, seq_len] -> [n_heads, n_tokens, n_tokens]
                _attn_i = _attn_i.to(torch.bfloat16)
                _attn_i = _attn_i[:, _seq_token_i, :][:, :, _seq_token_i]
                _attn_list.append(_attn_i.permute(1, 2, 0))

            x_list.append(_x[_seq_token_i, :])
            attn_list.append(torch.cat(_attn_list, dim=-1))

        x = torch.stack(x_list, dim=0)
        attn = torch.stack(attn_list, dim=0)

        # normalize
        x = self.transformer.norm(x).to(torch.bfloat16)

        # mask out invalid tokens
        token_mask = f_input.token.pad_mask
        # mask out non-protein tokens
        token_mask = token_mask & f_input.token.is_protein
        attn_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)

        x = x * token_mask[..., None]
        attn = attn * attn_mask[..., None]
        return x, attn
