import dataclasses

import torch

from .encoder import AtomisticImageEncoder
from .quantizer import Quantizer


@dataclasses.dataclass
class QuantizerConfig:
    codebook_size: int = 256
    codebook_embed_size: int = 384
    use_linear_project: bool = True


@dataclasses.dataclass
class EncoderConfig:
    d_single: int = 384
    d_pair: int = 128
    d_out: int = 384
    n_heads: int = 6
    n_layers: int = 8
    update_pair_repr_every_n: int = 2


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
        self.encoder = AtomisticImageEncoder(
            d_single=config.encoder.d_single,
            d_pair=config.encoder.d_pair,
            d_out=config.encoder.d_out,
            n_heads=config.encoder.n_heads,
            n_layers=config.encoder.n_layers,
            update_pair_repr_every_n=config.encoder.update_pair_repr_every_n,
        )
        self.quantizer = Quantizer(
            codebook_size=config.quantizer.codebook_size,
            codebook_embed_size=config.quantizer.codebook_embed_size,
            use_linear_project=config.quantizer.use_linear_project,
        )
        assert self.quantizer.codebook_embed_size == self.encoder.d_out

    def forward(
        self,
        aatypes: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            aatypes: [B, L]
            coords: [B, L, 37, 3]

        Returns:
            struct_tokens: [B, L]
        """
        z = self.encoder.encode(coords, aatypes)
        return self.quantizer.embedding2indices(z)
