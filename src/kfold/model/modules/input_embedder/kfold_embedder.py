import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.embeddings import RelativePositionEncoding
from kfold.model.layers.kfold.input_encoder import InputEmbedderWithApo
from kfold.model.layers.primitives import LinearNoBias
from kfold.utils.registry import INPUT_EMBEDDER, BaseConfig
from kfold.utils.torch import gather_dim

from .base import BaseInputEmbedder


class RBF(torch.nn.Module):
    """Radial basis function encoding for distances.

    Parameters
    ----------
    d_min : float
        The minimum distance for RBF encoding.
    d_max : float
        The maximum distance for RBF encoding.
    num_bins : int
        The number of bins for RBF encoding.
    """

    def __init__(
        self, d_min: float = 2.00, d_max: float = 50.75, num_bins: int = 40
    ) -> None:
        super().__init__()
        self.d_min: float = d_min
        self.d_max: float = d_max
        self.d_sigma: float = (d_max - d_min) / (num_bins - 2)
        self.register_buffer(
            "d_mu", torch.linspace(d_min, d_max, num_bins - 1), persistent=False
        )
        self.num_bins: int = num_bins

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Forward pass of RBF encoding.

        Parameters
        ----------
        dist : torch.Tensor
            Tensor of shape (...,) containing distances.
        Returns
        -------
        rbf : torch.Tensor
            Tensor of shape (..., num_bins) containing RBF encoded distances.
            last bin is for distances greater than d_max.
        """
        d_mu: torch.Tensor = self.d_mu
        rbf = torch.exp(-((dist.unsqueeze(-1) - d_mu) ** 2) / (2 * self.d_sigma**2))
        last_bin = (dist > self.d_max).float().unsqueeze(-1)
        rbf = torch.cat([rbf, last_bin], dim=-1)
        return rbf


@INPUT_EMBEDDER.register()
class KFoldInputEmbedder(BaseInputEmbedder):
    """Input embedding module for KFold model."""

    class Config(BaseConfig):
        """Configuration for the Input embedding module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        channel_atom : int
            The atom single embedding size.
        channel_atompair : int
            The atom pairwise embedding size.
        atom_encoder_blocks: int
            The atom encoder blocks.
        atom_encoder_heads: int
            The atom encoder heads.

        # Apo-related parameters
        apo_min_dist : float
            The minimum distance for apo distance map encoding.
        apo_max_dist : float
            The maximum distance for apo distance map encoding.
        apo_num_bins : int
            The number of bins for apo distance map encoding.
        """

        channel_s: int = 384
        channel_z: int = 128
        channel_atom: int = 128
        channel_atompair: int = 16
        atom_encoder_blocks: int = 3
        atom_encoder_heads: int = 4
        # Apo-related parameters
        apo_num_bins: int = 48
        apo_min_dist: float = 2.0
        apo_max_dist: float = 49.0

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.channel_s: int = cfg.channel_s
        self.channel_z: int = cfg.channel_z
        self.channel_atom: int = cfg.channel_atom
        self.channel_atompair: int = cfg.channel_atompair

        self.input_embedder = InputEmbedderWithApo(
            channel_s=cfg.channel_s,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            atom_encoder_blocks=cfg.atom_encoder_blocks,
            atom_encoder_heads=cfg.atom_encoder_heads,
        )

        # Initial linear layers for single and pair representations
        self.linear_s_init = LinearNoBias(cfg.channel_s, cfg.channel_s)
        self.linear_z_init1 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        self.linear_z_init2 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        self.rel_pos_encoding = RelativePositionEncoding(r_max=32, s_max=2)
        self.linear_rel_pos = LinearNoBias(self.rel_pos_encoding.dimension, cfg.channel_z)
        self.linear_bond = LinearNoBias(1, cfg.channel_z)

        # Apo-related
        self.distmap = RBF(cfg.apo_min_dist, cfg.apo_max_dist, cfg.apo_num_bins)
        # Pair representation
        self.linear_apo_pdist = LinearNoBias(self.distmap.num_bins, cfg.channel_z)

    def forward(
        self,
        f_input: FoldingInput,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of embedding module.
        See Section 3 Algorithm 1 and Algorithm 2 of AlphaFold3 paper.
        Algorithm 1 Line[1-5]

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.

        Returns
        -------
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single features
        s_init: torch.Tensor
            Tensor of shape (B, L, C_s) containing initial single representation
            before trunk.
        z_init: torch.Tensor
            Tensor of shape (B, L, L, C_z) containing initial pair representation
            before trunk.
        """
        inplace = not self.training
        add = (lambda x, y: x.add_(y)) if inplace else (lambda x, y: x + y)  # noqa

        # Get input single representation
        s_inputs = self.input_embedder(f_input)  # [B, L, c_s]

        # Get initial single representation
        s_init = self.linear_s_init(s_inputs)  # [B, L, c_s]

        # Get initial pair representation
        z_init = (
            self.linear_z_init1(s_inputs)[..., None, :, :]
            + self.linear_z_init2(s_inputs)[..., :, None, :]
        )  # [B, L, L, c_z]
        dtype = z_init.dtype

        # Add relative positional encoding
        z_init = add(z_init, self.linear_rel_pos(self.rel_pos_encoding(f_input, dtype)))

        # Add bond adjacency matrix
        z_init = add(z_init, self.linear_bond(self.get_adj(f_input, dtype).unsqueeze(-1)))

        # Add apo distance embedding
        z_init = add(z_init, self.linear_apo_pdist(self.get_apo_distmap(f_input, dtype)))

        return s_inputs, s_init, z_init

    def get_adj(
        self,
        f_input: FoldingInput,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get the adjacency bond matrix from the input features.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.

        Returns
        -------
        adj : torch.Tensor
            Tensor of shape (B, L, L) containing the adjacency matrix.
        """
        bond_index = f_input.bond.token_index  # [B, num_bonds, 2]
        mask = f_input.bond.pad_mask  # [B, num_bonds]

        B, L = f_input.batch_size, f_input.num_tokens
        dev = f_input.device

        # Masking; (0, 0) is padding index
        bond_index = bond_index.masked_fill(~mask[..., None], 0)

        # Create adjacency matrix
        src, dst = bond_index[:, :, 0], bond_index[:, :, 1]
        adj = torch.zeros((B, L, L), device=dev, dtype=dtype)
        b_idc = torch.arange(B, device=dev).unsqueeze(-1)
        adj[b_idc, src, dst] = 1.0
        adj[b_idc, dst, src] = 1.0  # undirected

        # Padding is always located at index (0,)
        adj[:, 0, 0] = 0.0
        return adj

    def get_apo_distmap(
        self,
        f_input: FoldingInput,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get apo embedding for the input features.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.

        Returns
        -------
        rbf_distmap : torch.Tensor
            Tensor of shape (B, L, L, num_bins) containing RBF-encoded apo distance map.
        """
        # Extract apo C-beta coordinates and mask
        repr_idx = f_input.token.repr_index
        coords = gather_dim(f_input.atom.apo_coords, -2, repr_idx[..., None])  # [B, L, 3]
        mask = gather_dim(f_input.atom.apo_mask, -1, repr_idx)  # [B, L]
        mask &= f_input.token.pad_mask  # ensure padding tokens are masked out

        # Create pairwise mask for valid tokens
        pair_mask = mask[..., :, None] & mask[..., None, :]  # [B, L, L]

        # Chain identity mask (no inter-chain apo distances)
        asym_id = f_input.token.asym_id  # [B, L]
        chain_mask = asym_id[:, :, None] == asym_id[:, None, :]
        pair_mask &= chain_mask

        with torch.autocast(f_input.device.type, enabled=False):
            # Compute pairwise distance map and apply RBF encoding
            pdist = (coords[..., :, None, :] - coords[..., None, :, :]).norm(dim=-1)
            distmap = self.distmap(pdist).to(dtype)  # [B, L, L, num_bins]
            distmap.masked_fill_(~pair_mask[..., None], 0.0)  # mask out invalid pairs

        return distmap
