import torch
import torch.nn as nn

from .nn import Embedding, Linear
from .transformer_layers import ESM1LayerNorm, TransformerLayer

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
            ESM1LayerNorm(embed_dim),
        )

        self.chain_embedding = Embedding(100, embed_dim)

        self.encoder = nn.ModuleList(
            [
                TransformerLayer(embed_dim, 4 * embed_dim, encoder_heads)
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm_after = ESM1LayerNorm(embed_dim)

    def forward(
        self,
        seq_token_id: torch.Tensor,
        bb_token_id: torch.Tensor,
        fa_token_id: torch.Tensor,
        seq_id: torch.Tensor | None = None,
        pos_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the backbone encoder.
        Args:
            seq_token_id: Tensor of shape (B, L) containing sequence token IDs.
            bb_token_id: Tensor of shape (B, L) containing backbone token IDs.
            fa_token_id: Tensor of shape (B, L) containing full-atom token IDs.
            seq_id: Optional tensor of shape (B, L) containing sequence IDs for
                attention masking. If None, no attention masking is applied.
            pos_id: Optional tensor of shape (B, L) containing position IDs for
                rotary embeddings. If None, arange(L) is used.
        """
        x_seq = self.seq_embedding(seq_token_id)
        x_bb = self.bb_embedding(bb_token_id)
        x_fa = self.fa_embedding(fa_token_id)
        x = self.fuse_layer(torch.cat([x_seq, x_bb, x_fa], dim=-1))
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

        # --- Encoder loop ---------------------------------------------------
        for layer in self.encoder:
            x = layer(x, seq_id, pos_id)

        x = self.encoder_norm_after(x)  # [B, L, embed_dim]
        return x  # [B, L, embed_dim]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "ProteinNetEncoder":
        """Load model from pretrained checkpoint."""
        model = cls().to(device=device, dtype=dtype).eval()
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        return model
