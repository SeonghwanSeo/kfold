"""Backbone VQ-VAE Tokenizer for Protein Structures"""

import contextlib
import dataclasses
from pathlib import Path

import torch

from .bb_vqvae.encoder import BackboneEncoder
from .bb_vqvae.quantizer import Quantizer


@dataclasses.dataclass
class QuantizerConfig:
    codebook_size: int = 512
    use_linear_project: bool = True


@dataclasses.dataclass
class EncoderConfig:
    d_model: int = 1024
    n_heads: int = 1
    v_heads: int = 128
    n_layers: int = 2
    d_out: int = 1024


@dataclasses.dataclass
class VQVAEConfig:
    quantizer: QuantizerConfig = dataclasses.field(default_factory=QuantizerConfig)
    encoder: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)


class BackboneTokenizer(torch.nn.Module):
    def __init__(self, config: VQVAEConfig | None = None):
        super().__init__()
        config = config or VQVAEConfig()
        self.config: VQVAEConfig = config
        self.encoder = BackboneEncoder(
            d_model=config.encoder.d_model,
            n_heads=config.encoder.n_heads,
            v_heads=config.encoder.v_heads,
            n_layers=config.encoder.n_layers,
            d_out=config.encoder.d_out,
        )
        self.quantizer = Quantizer(
            embed_size=config.encoder.d_out,
            codebook_size=config.quantizer.codebook_size,
            use_linear_project=config.quantizer.use_linear_project,
        )

    def forward(self, coords: torch.Tensor, res_idx: torch.Tensor) -> torch.Tensor:
        B, L = coords.shape[:2]
        if not coords.shape == (B, L, 3, 3):
            raise ValueError(
                f"Expected coords to have shape {(B, L, 3, 3)}, got {coords.shape}"
            )
        if not res_idx.shape == (B, L):
            raise ValueError(
                f"Expected res_idx to have shape {(B, L)}, got {res_idx.shape}"
            )
        z = self.encoder(coords, res_idx)
        return self.quantizer.embedding2indices(z)

    @torch.inference_mode()
    def tokenize(
        self,
        coords: torch.Tensor,
        res_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tokenize protein backbone structure

        Parameters
        ----------
        coords: torch.Tensor
            Backbone atom coordinates of shape (L, Natom, 3), where L is the number of
            residues and Natom is the number of atoms per residue.
            First 3 atoms should be N, CA, C in that order.
        res_idx: torch.Tensor, optional
            Residue indices of shape (L,). If not provided, will be set to range(1, L+1).

        Returns
        -------
        struct_ids: torch.Tensor
            Structure token IDs for each residue, of shape (L,).
        """
        assert coords.ndim == 3, "Expected coords to have shape (L, Natom, 3)"
        return self.tokenize_batch(
            coords.unsqueeze(0),
            res_idx.unsqueeze(0) if res_idx is not None else None,
        ).squeeze(0)

    @torch.inference_mode()
    def tokenize_batch(
        self,
        coords: torch.Tensor,
        res_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tokenize protein backbone structure

        Parameters
        ----------
        coords: torch.Tensor
            Backbone atom coordinates of shape (B, L, Natom, 3), where L is the number of
            residues and Natom is the number of atoms per residue.
            First 3 atoms should be N, CA, C in that order.
        res_idx: torch.Tensor, optional
            Residue indices of shape (B, L,). If not provided, will be set to
            arange(1, L+1).

        Returns
        -------
        struct_ids: torch.Tensor
            Structure token IDs for each residue, of shape (B, L,).
        """
        assert coords.ndim == 4, "Expected coords to have shape (B, L, Natom, 3)"
        B, L = coords.shape[:2]
        device = coords.device

        bb_coords = coords[..., :3, :]  # Use only N, CA, C atoms
        if res_idx is None:
            res_idx = torch.arange(1, L + 1, device=coords.device, dtype=torch.long)
        res_idx = res_idx.unsqueeze(0).expand(B, -1)  # (B, L)

        with (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        ):
            struct_ids = self(bb_coords, res_idx)
        return struct_ids

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        config: VQVAEConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> "BackboneTokenizer":
        """Load AminoAseed tokenizer from checkpoint."""

        # If config is not provided, use default config
        config = config or VQVAEConfig()

        # Initialize tokenizer
        tok = cls(config)

        # Load model weights
        model_states = torch.load(pretrained_path, map_location=device)
        if "module" in model_states:
            model_states = model_states["module"]
        model_states = {k.removeprefix("model."): v for k, v in model_states.items()}
        # Only load encoder and quantizer weights
        model_states = {
            k: v
            for k, v in model_states.items()
            if k.startswith(("encoder", "quantizer"))
        }
        tok.load_state_dict(model_states, strict=True)
        return tok.eval().to(device)
