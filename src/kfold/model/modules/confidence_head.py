from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.folding.pairformer import PairformerStack
from kfold.model.primitives import LayerNorm, LinearNoBias
from kfold.model.primitives.utils import gather_dim, get_context_dtype
from kfold.utils.registry import CONFIDENCE_HEAD, BaseConfig


def to_atom_layout(
    x: torch.Tensor,
    num_token_atoms: torch.Tensor,
    max_total_atoms: int,
) -> torch.Tensor:
    """Convert a tensor from token representation to dense representation.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (*, Ntoken, 24, C).
    num_token_atoms : torch.Tensor
        Tensor of shape (*, Ntoken) containing the number of atoms
        for each token.
    max_total_atoms : int
        The maximum total number of atoms across all tokens. This is used to
        determine the output shape.

    Returns
    -------
    x_atom: torch.Tensor
        Tensor of shape (B, Natom, C).
    """
    *batch_dims, L, _, C = x.shape

    arange = torch.arange(24, device=x.device)
    mask = arange < num_token_atoms.unsqueeze(-1)
    offsets = torch.cumsum(num_token_atoms, dim=-1) - num_token_atoms
    target_idx = offsets.unsqueeze(-1) + arange
    target_idx = target_idx.masked_fill(~mask, 0)
    x_flat = x.reshape(*batch_dims, -1, C)
    target_idx_flat = target_idx.reshape(*batch_dims, -1, 1).expand_as(x_flat)
    x_flat_masked = x_flat.masked_fill(~mask.reshape(*batch_dims, -1, 1), 0)

    # Scatter add the valid atoms into the dense atom representation
    out = torch.zeros(*batch_dims, max_total_atoms, C, device=x.device, dtype=x.dtype)
    out.scatter_add_(dim=-2, index=target_idx_flat, src=x_flat_masked)

    return out


@CONFIDENCE_HEAD.register()
class ConfidenceHead(torch.nn.Module):
    """Base class for confidence head modules.
    See Section 4.3.5 Algorithm 31 Confidence head
    """

    @dataclass
    class Config(BaseConfig):
        """Base configuration class for confidence head modules.

        Parameters
        ----------
        channel_s : int
            The channel of single representation.
        channel_z : int
            The channel of pair representation.
        """

        channel_s: int = 384
        channel_z: int = 128
        num_heads_attn: int = 16
        num_heads_tri_attn: int = 4
        num_blocks: int = 4
        dropout: float = 0.25
        num_bins: int = 39
        min_dist: float = 3.25
        max_dist: float = 50.75

        # head dimensions
        min_pae_dist: float = 0.0
        max_pae_dist: float = 32.0
        num_pae_bins: int = 64
        min_pde_dist: float = 0.0
        max_pde_dist: float = 32.0
        num_pde_bins: int = 64
        num_plddt_bins: int = 50

        # For training
        blocks_per_ckpt: int | None = None

    def __init__(self, cfg: Config, kernel_config: dict):
        super().__init__()
        self.config = cfg
        self.num_pae_bins = cfg.num_pae_bins
        self.num_pde_bins = cfg.num_pde_bins
        self.num_plddt_bins = cfg.num_plddt_bins
        self.num_resolved_bins = 2
        self.kernel_config = kernel_config
        self.is_compiled = False

        self.linear_s1 = LinearNoBias(cfg.channel_s, cfg.channel_z, init="default")
        self.linear_s2 = LinearNoBias(cfg.channel_s, cfg.channel_z, init="default")

        self.num_bins = cfg.num_bins
        boundaries = torch.linspace(cfg.min_dist, cfg.max_dist, self.num_bins - 1)
        self.register_buffer("boundaries", boundaries, persistent=False)
        self.linear_distogram = LinearNoBias(self.num_bins, cfg.channel_z, init="default")

        self.pairformer_stack = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads_attn=cfg.num_heads_attn,
            num_heads_tri_attn=cfg.num_heads_tri_attn,
            num_blocks=cfg.num_blocks,
            dropout=cfg.dropout,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

        self.pae_head = torch.nn.Sequential(
            LayerNorm(cfg.channel_z),
            LinearNoBias(cfg.channel_z, self.num_pae_bins, init="final"),
        )
        self.pde_head = torch.nn.Sequential(
            LayerNorm(cfg.channel_z),
            LinearNoBias(cfg.channel_z, self.num_pde_bins, init="final"),
        )
        self.plddt_head = torch.nn.Sequential(
            LayerNorm(cfg.channel_s),
            LinearNoBias(cfg.channel_s, 24 * self.num_plddt_bins, init="final"),
        )
        self.resolved_head = torch.nn.Sequential(
            LayerNorm(cfg.channel_s),
            LinearNoBias(cfg.channel_s, 24 * 2, init="final"),
        )

    def do_compile(self, **kwargs):
        """Compile the score model module."""
        self._compile(**kwargs)
        self.is_compiled = True

    def _compile(self, **kwargs):
        """Compile the pairformer."""
        self.pairformer_stack = torch.compile(self.pairformer_stack, **kwargs)

    def get_pairformer_stack(self) -> PairformerStack:
        """Get the PairformerStack."""
        if self.is_compiled and not self.training:
            return self.pairformer_stack._orig_mod  # type: ignore
        return self.pairformer_stack

    def forward_inference(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of confidence head module.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input features.
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single representation
        s: torch.Tensor
            Tensor of shape (B, L, C_s) containing single representation
        z: torch.Tensor
            Tensor of shape (B, L, L, C_s) containing pair representation
        x_pred: torch.Tensor
            Tensor of shape (B, N, Latom, 3) containing predicted coordinates

        Returns
        -------
        pae_logits: torch.Tensor
            Tensor of shape (B, N, L, L) containing PAE logits.
        pae_bin_centers: torch.Tensor
            Tensor of shape (num_pae_bins,) containing PAE bin centers.
        pde_logits: torch.Tensor
            Tensor of shape (B, N, L, L) containing PDE logits.
        pde_bin_centers: torch.Tensor
            Tensor of shape (num_pde_bins,) containing PDE bin centers.
        plddt_logits: torch.Tensor
            Tensor of shape (B, N, Natom, num_plddt_bins) containing pLDDT logits.
        plddt_bin_centers: torch.Tensor
            Tensor of shape (num_plddt_bins,) containing pLDDT bin centers.
        """
        pae_logits, pde_logits, plddt_logits, _ = self(f_input, s_inputs, s, z, x_pred)
        cfg = self.config
        device = pae_logits.device

        def create_bins(d_min: float, d_max: float, n_bin: int):
            d_bin = (d_max - d_min) / n_bin
            return torch.linspace(
                d_min + d_bin / 2, d_max - d_bin / 2, n_bin, device=device
            )

        return {
            "pae_logits": pae_logits,
            "pae_bin_centers": create_bins(
                cfg.min_pae_dist, cfg.max_pae_dist, cfg.num_pae_bins
            ),
            "pde_logits": pde_logits,
            "pde_bin_centers": create_bins(
                cfg.min_pde_dist, cfg.max_pde_dist, cfg.num_pde_bins
            ),
            "plddt_logits": plddt_logits,
            "plddt_bin_centers": create_bins(0.0, 1.0, cfg.num_plddt_bins),
        }

    def forward(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of confidence head module.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input features.
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single representation
        s: torch.Tensor
            Tensor of shape (B, L, C_s) containing single representation
        z: torch.Tensor
            Tensor of shape (B, L, L, C_s) containing pair representation
        x_pred: torch.Tensor
            Tensor of shape (B, N, Latom, 3) containing predicted coordinates

        Returns
        -------
        pae_logits: torch.Tensor
            Tensor of shape (B, N, L, L, num_pae_bins) containing predicted aligned
            error logits.
        pde_logits: torch.Tensor
            Tensor of shape (B, N, L, L, num_pde_bins) containing predicted distance
            error logits.
        plddt_logits: torch.Tensor
            Tensor of shape (B, N, Natom, num_lddt_bins) containing predicted lddt
            logits.
        resolved_logits: torch.Tensor
            Tensor of shape (B, N, Natom, 2) containing predicted resolved atom
            logits.
        """
        # Get the device and dtype for computations
        device = s_inputs.device
        dtype = get_context_dtype(device.type)

        # Extract the representative atom coordinates
        # [B, Natom, 3] -> [B, L, 3]
        B, N, Natom, _ = x_pred.shape
        L = s_inputs.shape[1]
        repr_idx = f_input.token.repr_index.unsqueeze(-2)  # [B, 1, Ntoken]
        x_repr = gather_dim(x_pred, dim=-2, index=repr_idx[..., None])

        s_inputs, s, z = s_inputs.to(dtype), s.to(dtype), z.to(dtype)

        # Line 1
        z = (
            z
            + self.linear_s1(s_inputs)[..., None, :, :]
            + self.linear_s2(s_inputs)[..., :, None, :]
        )

        # Prepare output tensors
        pae_logits = torch.zeros(
            (B, N, L, L, self.num_pae_bins), device=device, dtype=dtype
        )
        pde_logits = torch.zeros(
            (B, N, L, L, self.num_pde_bins), device=device, dtype=dtype
        )
        plddt_logits = torch.zeros(
            (B, N, L, 24, self.num_plddt_bins), device=device, dtype=torch.float32
        )
        resolved_logits = torch.zeros(
            (B, N, L, 24, 2), device=device, dtype=torch.float32
        )
        # Process each sample in the batch separately to save memory
        for i in range(N):
            _pae_logits, _pde_logits, _plddt_logits, _resolved_logits = (
                self.forward_single(
                    s,
                    z,
                    x_repr[:, i],
                    mask=f_input.token.pad_mask,
                )
            )
            pae_logits[:, i] = _pae_logits
            pde_logits[:, i] = _pde_logits
            plddt_logits[:, i] = _plddt_logits
            resolved_logits[:, i] = _resolved_logits

        # Reshape plddt and resolved logits to (B, N, Natom, ...)
        num_token_atoms = f_input.token.num_atoms.unsqueeze(-2).expand(-1, N, -1)
        plddt_logits = to_atom_layout(plddt_logits, num_token_atoms, f_input.num_atoms)
        resolved_logits = to_atom_layout(
            resolved_logits, num_token_atoms, f_input.num_atoms
        )

        # Mask out padding
        token_mask = f_input.token.pad_mask[..., None, :]  # [B, 1, L]
        pair_mask = token_mask[..., :, None] & token_mask[..., None, :]  # [B, 1, L, L]
        pair_mask = pair_mask.to(dtype)
        pae_logits = pae_logits * pair_mask[..., None]
        pde_logits = pde_logits * pair_mask[..., None]

        atom_mask = f_input.atom.pad_mask[..., None, :]  # [B, 1, Natom]
        atom_mask = atom_mask.to(torch.float32)
        plddt_logits = plddt_logits * atom_mask[..., None]
        resolved_logits = resolved_logits * atom_mask[..., None]

        return pae_logits, pde_logits, plddt_logits, resolved_logits

    def forward_single(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of confidence head module.

        Parameters
        ----------
        s: torch.Tensor
            Tensor of shape (B, L, C_s) containing single conditioning
        z: torch.Tensor
            Tensor of shape (B, L, L, C_s) containing pair conditioning
        x: torch.Tensor
            Tensor of shape (B, N, L, 3) containing predicted coordinates of
            representative atoms.
        mask: torch.Tensor
            Tensor of shape (B, L) containing the mask for valid tokens.

        Returns
        -------
        pae_logits: torch.Tensor
            Tensor of shape (B, L, L, num_pae_bins) containing predicted aligned
            error logits.
        pde_logits: torch.Tensor
            Tensor of shape (B, L, L, num_pde_bins) containing predicted distance
            error logits.
        lddt_logits: torch.Tensor
            Tensor of shape (B, L, 24, num_lddt_bins) containing predicted lddt logits.
        resolved_logits: torch.Tensor
            Tensor of shape (B, L, 24, 2) containing predicted resolved atom logits.
        """
        # Line 2
        with torch.autocast(x.device.type, dtype=torch.float32), torch.no_grad():
            d = (x[..., :, None, :] - x[..., None, :, :]).norm(dim=-1)

        # Line 3
        distogram = F.one_hot(
            (d[..., None] > self.boundaries).sum(dim=-1), self.num_bins
        )  # [B, Ntoken, Ntoken, Nbin]
        z = z + self.linear_distogram(distogram.to(z.dtype))

        # Line 4
        pairformer_stack = self.get_pairformer_stack()
        use_cuequiv_kernels = self.kernel_config.get("cuequivariance", False)

        s = s.clone()  # clone s to avoid in-place modification.
        s, z = pairformer_stack(s, z, mask, use_cuequiv_kernels=use_cuequiv_kernels)

        # Line 5
        pae_logits = self.pae_head(z)  # [B, L, L, num_pae_bins]

        # Line 6
        pde_logits = self.pde_head(z)  # [B, L, L, num_pde_bins]
        pde_logits = pde_logits + pde_logits.transpose(-2, -3)  # symmetrize

        with torch.autocast(x.device.type, dtype=torch.float32):
            s = s.to(torch.float32)

            # Line 7: plddt head
            plddt_logits = self.plddt_head(s).unflatten(-1, (24, self.num_plddt_bins))

            # Line 8: resolved head
            resolved_logits = self.resolved_head(s).unflatten(-1, (24, 2))

        return pae_logits, pde_logits, plddt_logits, resolved_logits
