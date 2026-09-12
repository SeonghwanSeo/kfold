from pathlib import Path

import torch
import torch.nn as nn

from .nn import Embedding, Linear
from .transformer_layers import RotaryEmbedding, TransformerLayer

# From main non-JEPA training path: non-foldseek uses 4 special tokens.
NUM_SPECIAL_TOKENS = 4
NUM_SEQ_TOKENS = 33 + NUM_SPECIAL_TOKENS  # 33 ESM3 vocab + 4 special
NUM_BB_TOKENS = 512 + NUM_SPECIAL_TOKENS  # 512 VQVAE vocab + 4 special
NUM_FA_TOKENS = 512 + NUM_SPECIAL_TOKENS  # 512 VQVAE vocab + 4 special


class ProteinNetEncoder(nn.Module):
    """Three-modality (seq + bb + fa-optional) encoder"""

    def __init__(
        self,
        embed_dim: int = 2560,
        encoder_depth: int = 33,
        encoder_heads: int = 40,
        seq_vocab_size: int = 37,
        bb_vocab_size: int = 516,
        fa_vocab_size: int = 516,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
        self.seq_vocab_size = seq_vocab_size
        self.bb_vocab_size = bb_vocab_size
        self.fa_vocab_size = fa_vocab_size

        self.seq_embedding = Embedding(seq_vocab_size, embed_dim)
        self.bb_embedding = Embedding(bb_vocab_size, embed_dim)
        self.fa_embedding = Embedding(fa_vocab_size, embed_dim)
        self.fuse_layer = nn.Sequential(
            Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim, eps=1e-12),
        )

        self.chain_embedding = Embedding(100, embed_dim)
        self.rotary = RotaryEmbedding(embed_dim // encoder_heads)

        self.encoder = nn.ModuleList(
            [
                TransformerLayer(embed_dim, 4 * embed_dim, encoder_heads)
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm_after = nn.LayerNorm(embed_dim, eps=1e-12)

    def forward(
        self,
        seq_token_id: torch.Tensor,
        bb_token_id: torch.Tensor,
        fa_token_id: torch.Tensor,
        seq_id: torch.Tensor | None = None,
        pos_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode sequence, backbone, and full-atom tokens.

        Parameters
        ----------
        seq_token_id, bb_token_id, fa_token_id : torch.Tensor
            Token IDs of shape (B, L) for the three input modalities.
        seq_id : torch.Tensor | None
            Chain IDs of shape (B, L) for attention masking. If omitted,
            all positions belong to the same chain.
        pos_id : torch.Tensor | None
            Position IDs of shape (B, L). If omitted, use arange(L).

        Returns
        -------
        torch.Tensor
            Encoded features of shape (B, L, embed_dim).
        """
        x_seq = self.seq_embedding(seq_token_id)
        x_bb = self.bb_embedding(bb_token_id)
        x_fa = self.fa_embedding(fa_token_id)
        x = self.fuse_layer(torch.cat([x_seq, x_bb, x_fa], dim=-1)).to(x_seq.dtype)
        chain_ids = torch.zeros((1, 1), dtype=torch.long, device=seq_token_id.device)
        chain_emb = self.chain_embedding(chain_ids)  # [1, 1, D]
        x = x + chain_emb  # [B, L, D]

        if seq_id is None:
            seq_id = torch.ones_like(seq_token_id)
        if pos_id is None:
            pos_id = (
                torch.arange(seq_token_id.shape[1], device=seq_token_id.device)
                .unsqueeze(0)
                .expand_as(seq_token_id)
            )

        attn_mask = (seq_id[:, None, :] == seq_id[:, :, None]).unsqueeze(1)
        rotary = self.rotary(pos_id)
        for layer in self.encoder:
            x = layer(x, attn_mask, rotary)

        return self.encoder_norm_after(x).to(x.dtype)

    def load_pretrained_weights(self, path: str | Path) -> None:
        """Load release weights in BF16, combining QKV projections once on CPU."""
        state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        inv_freq = state_dict["encoder.0.self_attn.rot_emb.inv_freq"]

        # Convert the checkpoint's separate projections to the fused layout.
        for index in range(self.encoder_depth):
            prefix = f"encoder.{index}.self_attn."
            for field in ("weight", "bias"):
                state_dict[f"{prefix}qkv_proj.{field}"] = torch.cat(
                    [
                        state_dict.pop(f"{prefix}{projection}_proj.{field}")
                        for projection in ("q", "k", "v")
                    ]
                )
            del state_dict[f"{prefix}rot_emb.inv_freq"]

        state_dict = {name: tensor.bfloat16() for name, tensor in state_dict.items()}
        state_dict["rotary.inv_freq"] = inv_freq.float()
        self.load_state_dict(state_dict, strict=True, assign=True)
        self.rotary.init_cache()
        self.requires_grad_(False).eval()
