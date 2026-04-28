import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import CONFIDENCE_HEAD, BaseConfig
from kfold.utils.torch import get_context_dtype

NUM_ATOM_TYPES = 37 + 29 + 1  # 67: 37 for protein, 29 for dna/rna, and 1 for ligand


def to_atom_layout(
    x: torch.Tensor,
    token_idcs: torch.Tensor,
    atom_idcs: torch.Tensor,
) -> torch.Tensor:
    """Convert a tensor from token representation to dense representation.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (*, Ntoken, 67, C).
    token_idcs : torch.Tensor
        Tensor of shape (*, Natom) containing the token indices.
    atom_idcs : torch.Tensor
        Tensor of shape (*, Natom) containing the atom type indices (0 to 66).

    Returns
    -------
    x_dense: torch.Tensor
        Tensor of shape (B, Natom, C).
    """
    *batch_dims, L, _, C = x.shape
    Natom = token_idcs.shape[-1]
    B = math.prod(batch_dims)
    device = x.device

    # Compute indices for gathering.
    _batch_idcs = torch.arange(B, device=device)[:, None]
    _token_idcs = token_idcs.view(B, Natom)
    _atom_idcs = atom_idcs.view(B, Natom)

    # Gathers features.
    x_flat = x.view(B, L, NUM_ATOM_TYPES, C)
    x_dense_flat = x_flat[_batch_idcs, _token_idcs, _atom_idcs]
    x_dense = x_dense_flat.view(*batch_dims, Natom, C)
    return x_dense


@CONFIDENCE_HEAD.register()
class ConfidenceHead(torch.nn.Module):
    """Base class for confidence head modules.
    See Section 4.3.5 Algorithm 31 Confidence head

    NOTE: Differ to original AF3, we use 67 unique atom types (37 for protein,
    29 for dna/rna, 1 for ligand) instead of 24 max atoms per token.
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
            LinearNoBias(
                cfg.channel_s, NUM_ATOM_TYPES * self.num_plddt_bins, init="final"
            ),
        )
        self.resolved_head = torch.nn.Sequential(
            LayerNorm(cfg.channel_s),
            LinearNoBias(cfg.channel_s, NUM_ATOM_TYPES * 2, init="final"),
        )

        # Create bins
        def create_bins(d_min: float, d_max: float, n_bin: int) -> torch.Tensor:
            d_bin = (d_max - d_min) / n_bin
            return torch.linspace(d_min + d_bin / 2, d_max - d_bin / 2, n_bin)

        self.pae_bins: torch.Tensor
        self.pde_bins: torch.Tensor
        self.plddt_bins: torch.Tensor
        self.register_buffer(
            "pae_bins",
            create_bins(cfg.min_pae_dist, cfg.max_pae_dist, cfg.num_pae_bins),
            persistent=False,
        )
        self.register_buffer(
            "pde_bins",
            create_bins(cfg.min_pde_dist, cfg.max_pde_dist, cfg.num_pde_bins),
            persistent=False,
        )
        self.register_buffer(
            "plddt_bins",
            create_bins(0.0, 1.0, cfg.num_plddt_bins),
            persistent=False,
        )

    def do_compile(self, **kwargs):
        """Compile the score model module."""
        self._compile(**kwargs)
        self.is_compiled = True

    def _compile(self, **kwargs):
        """Compile the pairformer."""
        self.pairformer_stack = torch.compile(self.pairformer_stack, **kwargs)

    def get_pairformer_stack(self, no_compile: bool = False) -> PairformerStack:
        """Get the PairformerStack."""
        if self.is_compiled and no_compile:
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
        pae_score: torch.Tensor
            Tensor of shape (B, N, L, L) containing PAE score.
        pde_score: torch.Tensor
            Tensor of shape (B, N, L, L) containing PDE score.
        plddt_score: torch.Tensor
            Tensor of shape (B, N, Natom) containing pLDDT score.
        """
        pae_logits, pde_logits, plddt_logits, _ = self(f_input, s_inputs, s, z, x_pred)
        p_pae = F.softmax(pae_logits, dim=-1)
        p_pde = F.softmax(pde_logits, dim=-1)
        p_plddt = F.softmax(plddt_logits, dim=-1)

        pae = (p_pae * self.pae_bins).sum(dim=-1)
        pde = (p_pde * self.pde_bins).sum(dim=-1)
        plddt = (p_plddt * self.plddt_bins).sum(dim=-1)
        return {
            "pae_score": pae,
            "pde_score": pde,
            "plddt_score": plddt,
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
        repr_idc = f_input.token.repr_index  # [B, Ntoken]
        x_repr = x_pred[
            torch.arange(B, device=device)[:, None, None],
            torch.arange(N, device=device)[None, :, None],
            repr_idc[:, None, :],
        ]  # [B, N, L, 3]

        s_inputs, s, z = s_inputs.to(dtype), s.to(dtype), z.to(dtype)

        # Line 1
        z = (
            z
            + self.linear_s1(s_inputs)[..., None, :, :]
            + self.linear_s2(s_inputs)[..., :, None, :]
        )

        # Prepare output tensors
        pae_logits = torch.empty(
            (B, N, L, L, self.num_pae_bins), device=device, dtype=dtype
        )
        pde_logits = torch.empty(
            (B, N, L, L, self.num_pde_bins), device=device, dtype=dtype
        )
        plddt_logits = torch.empty(
            (B, N, Natom, self.num_plddt_bins), device=device, dtype=torch.float32
        )
        resolved_logits = torch.empty(
            (B, N, Natom, 2), device=device, dtype=torch.float32
        )
        # Process each sample in the batch separately to save memory
        for i in range(N):
            _pae_logits, _pde_logits, _plddt_logits, _resolved_logits = (
                self.forward_single(
                    s,
                    z,
                    x_repr[:, i],
                    mask=f_input.token.pad_mask,
                    token_idcs=f_input.atom.token_index,
                    atom_idcs=f_input.atom.atom_type,
                )
            )
            pae_logits[:, i] = _pae_logits
            pde_logits[:, i] = _pde_logits
            plddt_logits[:, i] = _plddt_logits
            resolved_logits[:, i] = _resolved_logits
        return pae_logits, pde_logits, plddt_logits, resolved_logits

    def forward_single(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
        token_idcs: torch.Tensor,
        atom_idcs: torch.Tensor,
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
            Tensor of shape (B, L, 67, num_lddt_bins) containing predicted lddt logits.
        resolved_logits: torch.Tensor
            Tensor of shape (B, L, 67, 2) containing predicted resolved atom logits.
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
            plddt_logits = self.plddt_head(s).unflatten(
                -1, (NUM_ATOM_TYPES, self.num_plddt_bins)
            )
            plddt_logits = to_atom_layout(plddt_logits, token_idcs, atom_idcs)

            # Line 8: resolved head
            resolved_logits = self.resolved_head(s).unflatten(-1, (NUM_ATOM_TYPES, 2))
            resolved_logits = to_atom_layout(resolved_logits, token_idcs, atom_idcs)

        return pae_logits, pde_logits, plddt_logits, resolved_logits
