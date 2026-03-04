import dataclasses

import torch

from .encoder import VanillaStructureTokenEncoder
from .quantizer import Quantizer


@dataclasses.dataclass
class QuantizerConfig:
    codebook_size: int = 512
    codebook_embed_size: int = 1024
    use_linear_project: bool = False


@dataclasses.dataclass
class EncoderConfig:
    d_model: int = 1024
    n_heads: int = 1
    v_heads: int = 128
    n_layers: int = 2
    d_out: int = 1024


@dataclasses.dataclass
class DecoderConfig:
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 8


@dataclasses.dataclass
class VQVAEConfig:
    quantizer: QuantizerConfig = dataclasses.field(default_factory=QuantizerConfig)
    encoder: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)
    decoder: DecoderConfig = dataclasses.field(default_factory=DecoderConfig)


class VQVAE_EncoderOnly(torch.nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config: VQVAEConfig = config
        self.quantizer = Quantizer(
            codebook_size=config.quantizer.codebook_size,
            codebook_embed_size=config.quantizer.codebook_embed_size,
            use_linear_project=config.quantizer.use_linear_project,
        )
        self.encoder = VanillaStructureTokenEncoder(
            d_model=config.encoder.d_model,
            n_heads=config.encoder.n_heads,
            v_heads=config.encoder.v_heads,
            n_layers=config.encoder.n_layers,
            d_out=config.encoder.d_out,
            n_codes=config.quantizer.codebook_size,
        )
        assert self.quantizer.codebook_embed_size == self.encoder.d_out

    def forward(
        self,
        coords: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coords: [B, L, 3, 3]
            residue_index: [B, L]

        Returns:
            struct_tokens: [B, L]
        """
        z = self.encoder.encode(coords, residue_index)
        return self.quantizer.embedding2indices(z)
