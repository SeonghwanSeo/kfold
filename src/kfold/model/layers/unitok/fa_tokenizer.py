from pathlib import Path

import torch

from .fa_vqvae import VQVAE_EncoderOnly, VQVAEConfig
from .utils.esm_utils.constant import SEQUENCE_VOCAB
from .utils.openfold_utils.residue_constants import restypes as OPENFOLD_VOCAB


def centering(coords: torch.Tensor) -> torch.Tensor:
    """Center the coordinates by subtracting the mean position of the CA atoms.

    Parameters
    ----------
    coords: torch.Tensor
        Backbone atom coordinates of shape (*, L, Natom, 3), where L is the number of
        residues and Natom is the number of atoms per residue.
        First 3 atoms should be N, CA, C in that order.

    Returns
    -------
    centered_coords: torch.Tensor
        Centered backbone atom coordinates of shape (*, L, Natom, 3).
    """
    with torch.autocast(device_type=coords.device.type, enabled=False):
        mask = coords.isfinite().all(dim=-1)
        coords = coords.masked_fill(~mask[..., None], 0.0)

        ca_coords = coords[..., 1, :]  # CA atom is the second atom (index 1)
        centroid = ca_coords.sum(dim=-2) / mask[..., 1].sum(dim=-1, keepdim=True).clamp(
            min=1
        )  # Mean CA position, avoid division by zero

        coords -= centroid[..., None, None, :]
    coords[~mask] = 0.0
    return coords


def create_esm_to_of() -> torch.Tensor:
    # Create a mapping from esm vocab to openfold vocab
    default_value = 0  # Default to 'A' for any unknown amino acids
    esm_to_openfold = torch.full(
        (len(SEQUENCE_VOCAB),), fill_value=default_value, dtype=torch.long
    )
    for idx, aa in enumerate(SEQUENCE_VOCAB):
        if aa in OPENFOLD_VOCAB:
            openfold_idx = OPENFOLD_VOCAB.index(aa)
            esm_to_openfold[idx] = openfold_idx
    return esm_to_openfold


class FullAtomTokenizer(torch.nn.Module):
    def __init__(self, config: VQVAEConfig | None = None):
        super().__init__()
        config = config or VQVAEConfig()
        self.model = VQVAE_EncoderOnly(config).eval()
        # Freeze model parameters
        for p in self.parameters():
            p.requires_grad = False

        esm_to_of = create_esm_to_of()
        self.register_buffer("esm_to_of", esm_to_of, persistent=False)

    def forward(
        self,
        aatypes: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        return self.tokenize(aatypes, coords)

    @torch.inference_mode()
    def tokenize(
        self,
        aatypes: torch.Tensor,
        coords: torch.Tensor,
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

        Returns
        -------
        struct_ids: torch.Tensor
            Structure token IDs for each residue, of shape (L,).
        """
        assert aatypes.ndim == 1, "Expected aatypes to have shape (L,)"
        return self.tokenize_batch(
            aatypes.unsqueeze(0),
            coords.unsqueeze(0),
        ).squeeze(0)

    @torch.inference_mode()
    def tokenize_batch(
        self,
        aatypes: torch.Tensor,
        coords: torch.Tensor,
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

        Returns
        -------
        struct_ids: torch.Tensor
            Structure token IDs for each residue, of shape (B, L,).
        """
        assert aatypes.ndim == 2, "Expected aatypes to have shape (B, L)"
        B, L = aatypes.shape
        device = aatypes.device

        # Map ESM amino acid types to OpenFold amino acid types
        aatypes = self.esm_to_of[aatypes]

        # Center coordinates by CA atom
        coords = centering(coords)

        # Run VQVAE encoder to get quantized indices
        with (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast(device_type=device.type, enabled=False)
        ):
            struct_ids = self.model(aatypes, coords)
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
        tok.model.load_state_dict(model_states, strict=True)
        return tok.eval().to(device)
