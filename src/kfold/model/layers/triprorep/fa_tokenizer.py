"""Intra-residue VQ-VAE Tokenizer for Protein Structures"""

import contextlib
import dataclasses
from pathlib import Path

import torch

from .fa_vqvae.encoder import AtomisticImageEncoder
from .fa_vqvae.quantizer import Quantizer


@dataclasses.dataclass
class QuantizerConfig:
    codebook_size: int = 512
    use_linear_project: bool = False


@dataclasses.dataclass
class EncoderConfig:
    d_model: int = 256
    d_out: int = 256
    d_pair: int = 128
    n_heads: int = 8
    n_layers: int = 6
    update_pair_repr_every_n: int = 2


@dataclasses.dataclass
class VQVAEConfig:
    quantizer: QuantizerConfig = dataclasses.field(default_factory=QuantizerConfig)
    encoder: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)


class FullAtomTokenizer(torch.nn.Module):
    def __init__(self, config: VQVAEConfig | None = None):
        super().__init__()
        config = config or VQVAEConfig()
        self.config: VQVAEConfig = config
        self.encoder = AtomisticImageEncoder(
            c_s=config.encoder.d_model,
            c_z=config.encoder.d_pair,
            c_out=config.encoder.d_out,
            n_heads=config.encoder.n_heads,
            n_layers=config.encoder.n_layers,
            update_pair_repr_every_n=config.encoder.update_pair_repr_every_n,
        )
        self.quantizer = Quantizer(
            embed_size=config.encoder.d_out,
            codebook_size=config.quantizer.codebook_size,
            use_linear_project=config.quantizer.use_linear_project,
        )

    def forward(
        self,
        aatypes: torch.Tensor,
        coords: torch.Tensor,
        res_idx: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            coords: [B, L, 37, 3]
                Dummy atoms should be set to NaN.
            aatypes: [B, L]
                Amino acid type indices.
            res_idx: [B, L]
                Residue indices (1-based).
            attn_mask: [B, L] (optional)
                Optional attention mask.

        Returns:
            tokens: [B, L]
                Structure token indices.
        """
        assert aatypes.ndim == 2, "Expected aatypes to have shape (B, L)"
        B, L = coords.shape[:2]
        if not coords.shape == (B, L, 37, 3):
            raise ValueError(
                f"Expected coords to have shape {(B, L, 37, 3)}, got {coords.shape}"
            )
        if not res_idx.shape == (B, L):
            raise ValueError(
                f"Expected res_idx to have shape {(B, L)}, got {res_idx.shape}"
            )
        z = self.encoder(aatypes, coords, res_idx, attn_mask)
        return self.quantizer.embedding2indices(z)

    @torch.inference_mode()
    def tokenize(
        self,
        aatypes: torch.Tensor,
        coords: torch.Tensor,
        res_idx: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tokenize protein full-atom structure

        Parameters
        ----------
        aatypes: torch.Tensor
            Amino acid type indices of shape (L,).
            These should be indices corresponding to the ESM sequence vocabulary.
        coords: torch.Tensor
            Full-atom coordinates of shape (L, 37, 3), where L is the number of
            residues and 37 is the number of atoms per residue.
            Dummy atoms should be set to NaN.
        res_idx: torch.Tensor | None
            Optional residue indices of shape (L,).
        attn_mask: torch.Tensor | None
            Optional attention mask of shape (L,), where 1 indicates valid residues
            and 0 indicates padding.

        Returns
        -------
        struct_ids: torch.Tensor
            Structure token IDs for each residue, of shape (L,).
        """
        assert aatypes.ndim == 1, "Expected aatypes to have shape (L,)"
        return self.tokenize_batch(
            aatypes.unsqueeze(0),
            coords.unsqueeze(0),
            res_idx.unsqueeze(0) if res_idx is not None else None,
            attn_mask.unsqueeze(0) if attn_mask is not None else None,
        ).squeeze(0)

    @torch.inference_mode()
    def tokenize_batch(
        self,
        aatypes: torch.Tensor,
        coords: torch.Tensor,
        res_idx: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tokenize protein full-atom structure

        Parameters
        ----------
        aatypes: torch.Tensor
            Amino acid type indices of shape (B, L,).
            These should be indices corresponding to the ESM sequence vocabulary.
        coords: torch.Tensor
            Full-atom coordinates of shape (B, L, 37, 3), where L is the number of
            residues and 37 is the number of atoms per residue.
            Dummy atoms should be set to NaN.

        Returns
        -------
        struct_ids: torch.Tensor
            Structure token IDs for each residue, of shape (B, L,).
        """
        assert aatypes.ndim == 2, "Expected aatypes to have shape (B, L)"
        B, L = aatypes.shape
        device_type = aatypes.device.type

        if res_idx is None:
            res_idx = torch.arange(1, L + 1, device=coords.device, dtype=torch.long)
        res_idx = res_idx.unsqueeze(0).expand(B, -1)  # (B, L)

        with (
            torch.autocast(device_type, dtype=torch.bfloat16)
            if device_type == "cuda"
            else contextlib.nullcontext()
        ):
            struct_ids = self(aatypes, coords, res_idx, attn_mask)
        return struct_ids

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        config: VQVAEConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> "FullAtomTokenizer":
        """Load FullAtomTokenizer from a pretrained checkpoint."""

        # If config is not provided, use default config
        config = config or VQVAEConfig()

        # Initialize tokenizer
        tok = cls(config)

        # Load model weights
        model_states = torch.load(
            pretrained_path, map_location=device, weights_only=False
        )
        if "state_dict" in model_states:
            model_states = model_states["state_dict"]
        if "module" in model_states:
            model_states = model_states["module"]

        model_states = {k.removeprefix("model."): v for k, v in model_states.items()}
        # Only load encoder and quantizer weights
        model_states = {
            k: v
            for k, v in model_states.items()
            if k.startswith(("encoder", "quantizer"))
        }
        model_states = {
            k: v for k, v in model_states.items() if not k.startswith("quantizer._ema")
        }
        tok.load_state_dict(model_states, strict=True)
        return tok.eval().to(device)
