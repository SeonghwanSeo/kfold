import torch
import torch.nn.functional as F

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.embeddings import RelativePositionEncoding
from kfold.model.layers.alphafold3.input_encoder import InputFeatureEmbedder
from kfold.model.layers.kfold.encoder import InputEmbedderWithApo
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import INPUT_EMBEDDER, BaseConfig

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
        self, d_min: float = 2.0, d_max: float = 22.0, num_bins: int = 64
    ) -> None:
        super().__init__()
        self.d_sigma: float = (d_max - d_min) / num_bins
        self.register_buffer(
            "d_mu", torch.linspace(d_min, d_max, num_bins), persistent=False
        )

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
        """
        d_mu: torch.Tensor = self.d_mu
        rbf = torch.exp(-((dist.unsqueeze(-1) - d_mu) ** 2) / (2 * self.d_sigma**2))
        return rbf


class Distogram(torch.nn.Module):
    """One-hot contact map encoding for distances.

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
        self, d_min: float = 2.0, d_max: float = 22.0, num_bins: int = 64
    ) -> None:
        super().__init__()
        bin_size = (d_max - d_min) / num_bins
        first_bin = d_min + bin_size  # =2.3125
        last_bin = d_max - bin_size  # =21.6875

        boundaries = torch.linspace(first_bin, last_bin, num_bins - 1)  # [num_bins - 1]
        self.register_buffer("boundaries", boundaries, persistent=False)
        self.num_bins: int = num_bins

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Forward pass of RBF encoding.
        Parameters
        ----------
        dist : torch.Tensor
            Tensor of shape (...,) containing distances.
        Returns
        -------
        distogram : torch.Tensor
            Tensor of shape (..., num_bins) containing one-hot distance mp
        """
        boundaries: torch.Tensor = self.boundaries  # type: ignore
        distogram = (dist.unsqueeze(-1) > boundaries).sum(dim=-1).long()

        # One-hot encoding
        return F.one_hot(distogram, num_classes=self.num_bins).float()


@INPUT_EMBEDDER.register()
class PretrainedInputEmbedder(BaseInputEmbedder):
    """Input embedding module using pre-trained embeddings."""

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
        channel_seq_encoder : int | None
            The pre-trained sequence encoder output channel size.
        atoms_per_window_queries: int
            The number of atoms per window for queries.
        atoms_per_window_keys: int
            The number of atoms per window for keys.
        atom_encoder_blocks: int
            The atom encoder blocks.
        atom_encoder_heads: int
            The atom encoder heads.
        max_relative_token : int
            The maximum relative residue distance for relative position encoding.
        max_relative_chain : int
            The maximum relative chain distance for relative position encoding.
        use_apo : bool
            Whether to embed apo structure.
        apo_distmap_type : str
            options: 'rbf', 'distogram'
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
        channel_seq_encoder: int | None = None
        channel_struct_encoder: int | None = None
        atoms_per_window_queries: int = 32
        atoms_per_window_keys: int = 128
        atom_encoder_blocks: int = 3
        atom_encoder_heads: int = 4
        max_relative_token: int = 32
        max_relative_chain: int = 2
        use_apo: bool = True
        apo_distmap_type: str = "rbf"
        apo_num_bins: int = 64
        apo_min_dist: float = 2.0
        apo_max_dist: float = 22.0

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.channel_s: int = cfg.channel_s
        self.channel_z: int = cfg.channel_z
        self.channel_atom: int = cfg.channel_atom
        self.channel_atompair: int = cfg.channel_atompair
        self.use_apo: bool = cfg.use_apo

        assert cfg.apo_distmap_type in ["rbf", "distogram"], (
            f"Invalid distmap_type: {cfg.apo_distmap_type}. "
            "Choose from 'rbf' or 'distogram'."
        )

        if self.use_apo:
            self.encoder = InputEmbedderWithApo(
                channel_s=cfg.channel_s,
                channel_atom=cfg.channel_atom,
                channel_atompair=cfg.channel_atompair,
                atoms_per_window_queries=cfg.atoms_per_window_queries,
                atoms_per_window_keys=cfg.atoms_per_window_keys,
                atom_encoder_blocks=cfg.atom_encoder_blocks,
                atom_encoder_heads=cfg.atom_encoder_heads,
            )
        else:
            self.encoder = InputFeatureEmbedder(
                channel_s=cfg.channel_s,
                channel_atom=cfg.channel_atom,
                channel_atompair=cfg.channel_atompair,
                atoms_per_window_queries=cfg.atoms_per_window_queries,
                atoms_per_window_keys=cfg.atoms_per_window_keys,
                atom_encoder_blocks=cfg.atom_encoder_blocks,
                atom_encoder_heads=cfg.atom_encoder_heads,
            )

        # Pre-trained encoders
        self.use_seq_enc: bool = cfg.channel_seq_encoder is not None
        if cfg.channel_seq_encoder is not None:
            self.proj_seq_emb = torch.nn.Sequential(
                LayerNorm(cfg.channel_seq_encoder, create_offset=False),
                LinearNoBias(cfg.channel_seq_encoder, cfg.channel_s, init="relu"),
                torch.nn.ReLU(),
                LinearNoBias(cfg.channel_s, cfg.channel_s, init="zero"),
            )

        # Initial linear layers for single and pair representations
        self.linear_s_init = LinearNoBias(cfg.channel_s, cfg.channel_s)
        self.linear_z_init1 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        self.linear_z_init2 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        self.relative_pos_encoding = RelativePositionEncoding(
            r_max=cfg.max_relative_token, s_max=cfg.max_relative_chain
        )
        self.linear_rel_pos = LinearNoBias(
            self.relative_pos_encoding.dimension, cfg.channel_z
        )
        self.linear_bond = LinearNoBias(1, cfg.channel_z)

        # Token-level apo embedding
        if cfg.use_apo:
            if cfg.apo_distmap_type == "rbf":
                # rbf
                self.distmap = RBF(cfg.apo_min_dist, cfg.apo_max_dist, cfg.apo_num_bins)
            else:
                # distogram
                self.distmap = Distogram(
                    cfg.apo_min_dist, cfg.apo_max_dist, cfg.apo_num_bins
                )

            # Pair representation
            self.linear_apo_pdist = LinearNoBias(cfg.apo_num_bins, cfg.channel_z)

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

        # Get input single representation
        s_inputs = self.encoder(f_input)  # [B, L, c_s]

        # Add pre-trained sequence/structure embedding if available
        if self.use_seq_enc:
            assert f_input.pretrained.has_sequence_embedding
            seq_emb = f_input.pretrained.sequence_embedding  # [B, Lt, c_seq_enc]
            s_seq = self.proj_seq_emb(seq_emb)  # [B, Lt, c_s]
            s_inputs = s_inputs + s_seq

        # Get initial single representation
        s_init = self.linear_s_init(s_inputs)  # [B, L, c_s]

        # Get initial pair representation
        z_init = (
            self.linear_z_init1(s_inputs)[:, None, :, :]
            + self.linear_z_init2(s_inputs)[:, :, None, :]
        )  # [B, L, L, c_z]

        # Add relative positional encoding
        rel_feat = self.relative_pos_encoding(f_input)
        z_init = z_init + self.linear_rel_pos(rel_feat)  # [B, L, L, c_z]

        # Add bond adjacency matrix
        z_init = z_init + self.linear_bond(
            self.get_adjacency_matrix(
                f_input.bond.token_index, f_input.num_tokens, f_input.bond.pad_mask
            ).unsqueeze(-1)  # [B, L, L, 1]
        )  # [B, L, L, c_z]

        # Add apo distance embedding
        if self.use_apo:
            z_init = z_init + self.get_apo_embedding(f_input)  # [B, L, L, c_z]

        return s_inputs, s_init, z_init

    def get_apo_embedding(self, f_input: FoldingInput) -> torch.Tensor:
        """Get apo embedding for the input features.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.

        Returns
        -------
        z_apo : torch.Tensor
            Pair representation containing apo information. Shape: (B, L, L, c_z)
        """
        batch_index = torch.arange(f_input.batch_size, device=f_input.device)[:, None]
        center_index = f_input.token.center_index

        # Extract apo C-alpha coordinates and mask
        apo_coords = f_input.atom.apo_coords[batch_index, center_index]  # [B, L, 3]
        mask = f_input.atom.apo_mask[batch_index, center_index]  # [B, L]
        pair_mask = mask[:, :, None] & mask[:, None, :]

        # Chain identity mask (no inter-chain apo distances)
        asym_id = f_input.token.asym_id  # [B, L]
        chain_mask = asym_id[:, :, None] == asym_id[:, None, :]

        pair_mask = pair_mask & chain_mask

        # Pair representation: pairwise distance RBF
        with torch.autocast("cuda", enabled=False):
            pdist = torch.cdist(apo_coords, apo_coords, p=2)  # [B, L, L]
            pdist_map = self.distmap(pdist)  # [B, L, L, num_bin]
        pdist_map = pdist_map * pair_mask.unsqueeze(-1)  # apply mask

        z_apo = self.linear_apo_pdist(pdist_map)  # [B, L, L, c_z]

        return z_apo

    def get_adjacency_matrix(
        self, bond_index: torch.Tensor, num_tokens: int, mask: torch.Tensor
    ) -> torch.Tensor:
        """Get the adjacency bond matrix from the input features."""

        # Masking; (0, 0) is padding index
        bond_index = bond_index * mask.unsqueeze(-1)

        src, dst = bond_index[:, :, 0], bond_index[:, :, 1]

        batch_size = bond_index.shape[0]
        adj = torch.zeros(
            (batch_size, num_tokens, num_tokens),
            device=bond_index.device,
            dtype=torch.float32,
        )

        batch_indices = (
            torch.arange(batch_size, device=bond_index.device)
            .unsqueeze(-1)
            .expand_as(src)
        )

        adj[batch_indices, src, dst] = 1.0
        adj[batch_indices, dst, src] = 1.0  # undirected

        # Padding is always located at index (0,)
        adj[:, 0, 0] = 0
        return adj
