"""Backbone VQ-VAE Tokenizer for Protein Structures"""

from pathlib import Path

import torch

from .bb_vqvae import VQVAE_EncoderOnly, VQVAEConfig


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
        ca_coords = coords[..., 1, :]  # CA atom is the second atom (index 1)
        ca_mask = ca_coords.isfinite().all(dim=-1)
        ca_coords = ca_coords.masked_fill(~ca_mask[..., None], 0.0)
        n_ca = ca_mask.sum(dim=-1, keepdim=True)
        centroid = ca_coords.sum(dim=-2) / n_ca.clamp(min=1)
        coords = coords - centroid[..., None, None, :]
    return coords


class BackboneTokenizer(torch.nn.Module):
    def __init__(self, config: VQVAEConfig | None = None):
        super().__init__()
        config = config or VQVAEConfig()
        self.model = VQVAE_EncoderOnly(config).eval()
        # Freeze model parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        coords: torch.Tensor,
        residue_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.tokenize(coords, residue_index)

    @torch.inference_mode()
    def tokenize(
        self,
        coords: torch.Tensor,
        residue_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tokenize protein backbone structure

        Parameters
        ----------
        coords: torch.Tensor
            Backbone atom coordinates of shape (L, Natom, 3), where L is the number of
            residues and Natom is the number of atoms per residue.
            First 3 atoms should be N, CA, C in that order.
        residue_index: torch.Tensor, optional
            Residue indices of shape (L,). If not provided, will be set to range(1, L+1).

        Returns
        -------
        struct_ids: torch.Tensor
            Structure token IDs for each residue, of shape (L,).
        """
        assert coords.ndim == 3, "Expected coords to have shape (L, Natom, 3)"
        return self.tokenize_batch(
            coords.unsqueeze(0),
            residue_index.unsqueeze(0) if residue_index is not None else None,
        ).squeeze(0)

    @torch.inference_mode()
    def tokenize_batch(
        self,
        coords: torch.Tensor,
        residue_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tokenize protein backbone structure

        Parameters
        ----------
        coords: torch.Tensor
            Backbone atom coordinates of shape (B, L, Natom, 3), where L is the number of
            residues and Natom is the number of atoms per residue.
            First 3 atoms should be N, CA, C in that order.
        residue_index: torch.Tensor, optional
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
        if residue_index is None:
            residue_index = torch.arange(1, L + 1, device=coords.device, dtype=torch.long)
        residue_index = residue_index.unsqueeze(0).expand(B, -1)  # (B, L)

        # Center coordinates by CA atom
        bb_coords = centering(bb_coords)

        with (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast(device_type=device.type, enabled=False)
        ):
            # Run VQVAE encoder to get quantized indices
            struct_ids = self.model(bb_coords, residue_index)
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
        tok.model.load_state_dict(model_states, strict=True)
        return tok.eval().to(device)
