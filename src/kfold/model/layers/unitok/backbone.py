import torch
import torch.nn as nn

from .transformer_layers import ESM1LayerNorm, TransformerLayer

# From main non-JEPA training path: non-foldseek uses 4 special tokens.
NUM_SPECIAL_TOKENS = 4
NUM_SEQ_TOKENS = 33 + NUM_SPECIAL_TOKENS  # 33 ESM3 vocab + 4 special
NUM_BB_TOKENS = 512 + NUM_SPECIAL_TOKENS  # 512 VQVAE vocab + 4 special
NUM_FA_TOKENS = 256 + NUM_SPECIAL_TOKENS  # 256 VQVAE vocab + 4 special


class UniTokBackbone(nn.Module):
    """Backbone network producing fused seq/bb/fa embeddings and decoder outputs."""

    def __init__(
        self,
        embed_dim: int = 1536,
        encoder_depth: int = 30,
        encoder_heads: int = 24,
        decoder_dim: int | None = 1536,
        decoder_depth: int | None = 3,
        decoder_heads: int | None = 24,
        seq_vocab_size: int = NUM_SEQ_TOKENS,
        bb_vocab_size: int = NUM_BB_TOKENS,
        fa_vocab_size: int = NUM_FA_TOKENS,
    ):
        super().__init__()
        self.embed_dim: int = embed_dim
        self.seq_vocab_size: int = seq_vocab_size
        self.bb_vocab_size: int = bb_vocab_size
        self.fa_vocab_size: int = fa_vocab_size

        self.num_special_tokens: int = NUM_SPECIAL_TOKENS
        self.seq_embedding = nn.Embedding(seq_vocab_size, embed_dim)
        self.bb_embedding = nn.Embedding(bb_vocab_size, embed_dim)
        self.fa_embedding = nn.Embedding(fa_vocab_size, embed_dim)
        self.fuse_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            ESM1LayerNorm(embed_dim),
        )

        self.token_type_embedding = nn.Embedding(16, embed_dim)
        # --- Encoder ---
        self.encoder = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=embed_dim,
                    n_heads=encoder_heads,
                    expansion_ratio=4.0,
                    use_rotary_embeddings=True,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm_after = ESM1LayerNorm(embed_dim)

        # --- Decoder ---
        # NOTE: we do not initialize the decoder for inference
        if False:
            self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
            self.decoder = nn.ModuleList(
                [
                    TransformerLayer(
                        d_model=decoder_dim,
                        n_heads=decoder_heads,
                        expansion_ratio=4.0,
                        use_rotary_embeddings=True,
                    )
                    for _ in range(decoder_depth)
                ]
            )
            self.decoder_norm_after = ESM1LayerNorm(decoder_dim)
            # --- Prediction Heads ---
            self.lm_head_seq = nn.Linear(decoder_dim, seq_vocab_size)
            self.lm_head_bb = nn.Linear(decoder_dim, bb_vocab_size)
            self.lm_head_fa = nn.Linear(decoder_dim, fa_vocab_size)

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
        B, L = seq_token_id.shape

        n_special = self.num_special_tokens
        seq_token_id = seq_token_id + n_special
        bb_token_id = bb_token_id + n_special
        fa_token_id = fa_token_id + n_special

        x_seq = self.seq_embedding(seq_token_id)  # [B, L, dim]
        x_bb = self.bb_embedding(bb_token_id)  # [B, L, dim]
        x_fa = self.fa_embedding(fa_token_id)  # [B, L, dim]
        x = self.fuse_layer(torch.cat([x_seq, x_bb, x_fa], dim=-1))  # [B, L, dim]

        token_identifier = torch.zeros((B, L), device=x.device, dtype=torch.long)
        x += self.token_type_embedding(token_identifier)

        for layer in self.encoder:
            x = layer(x, seq_id=seq_id, pos_id=pos_id)

        return self.encoder_norm_after(x)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device | str,
    ) -> "UniTokBackbone":
        """Load model from pretrained checkpoint."""
        state_dict = torch.load(pretrained_path, map_location="cpu")

        # Remove potential "state_dict" wrapper and "model." prefix from keys.
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

        # Remove decoder weights since we are only doing inference with the encoder.
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith(("decoder", "lm_head"))
        }
        model = cls(
            embed_dim=1536,
            encoder_depth=30,
            encoder_heads=24,
            decoder_dim=1536,
            decoder_depth=3,
            decoder_heads=24,
            seq_vocab_size=NUM_SEQ_TOKENS,
            bb_vocab_size=NUM_BB_TOKENS,
            fa_vocab_size=NUM_FA_TOKENS,
        )
        model.load_state_dict(state_dict, strict=True)
        return model.eval().to(device)
