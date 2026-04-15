import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.embeddings import RelativePositionEncoding
from kfold.model.layers.alphafold3.input_encoder import InputFeatureEmbedder
from kfold.model.layers.primitives import LinearNoBias
from kfold.utils.registry import INPUT_EMBEDDER, BaseConfig

from .base import BaseInputEmbedder


@INPUT_EMBEDDER.register()
class AF3InputEmbedder(BaseInputEmbedder):
    """Input embedding module based on AlphaFold3.
    See Section 3 Algorithm 1 and Algorithm 2 of AlphaFold3 paper.
    Algorithm 1 Line[1-5]
    """

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
        atom_encoder_blocks: int,
            The atom encoder blocks.
        atom_encoder_heads: int,
            The atom encoder heads.
        """

        channel_s: int = 384
        channel_z: int = 128
        channel_atom: int = 128
        channel_atompair: int = 16
        atom_encoder_blocks: int = 3
        atom_encoder_heads: int = 4

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.channel_s = cfg.channel_s
        self.channel_z = cfg.channel_z
        self.channel_atom = cfg.channel_atom
        self.channel_atompair = cfg.channel_atompair

        self.input_embedder = InputFeatureEmbedder(
            channel_s=cfg.channel_s,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            atom_encoder_blocks=cfg.atom_encoder_blocks,
            atom_encoder_heads=cfg.atom_encoder_heads,
        )

        # Project to model dimension
        # Line 2
        self.linear_s_init = LinearNoBias(cfg.channel_s, cfg.channel_s)
        # Line 3
        self.linear_z_init1 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        self.linear_z_init2 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        # Line 4
        self.rel_pos_encoding = RelativePositionEncoding(r_max=32, s_max=2)
        self.linear_rel_pos = LinearNoBias(self.rel_pos_encoding.dimension, cfg.channel_z)
        # Line 5
        self.linear_bond = LinearNoBias(1, cfg.channel_z)

    def forward(
        self,
        f_input: FoldingInput,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of embedding module.

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
            Tensor of shape (B, L, L, C_s) containing initial pair representation
            before trunk.
        """
        inplace = not self.training
        add = (lambda x, y: x.add_(y)) if inplace else (lambda x, y: x + y)  # noqa

        # Line 1
        s_inputs = self.input_embedder(f_input)  # [B, L, c_s]

        # Get initial single and pair representations
        # Line 2
        s_init = self.linear_s_init(s_inputs)  # [B, L, c_s]

        # Line 3
        z_init = (
            self.linear_z_init1(s_inputs)[:, None, :, :]
            + self.linear_z_init2(s_inputs)[:, :, None, :]
        )  # [B, L, L, c_z]
        dtype = z_init.dtype

        # Line 4
        z_init = add(z_init, self.linear_rel_pos(self.rel_pos_encoding(f_input, dtype)))

        # Line 5
        z_init = add(z_init, self.linear_bond(self.get_adj(f_input, dtype).unsqueeze(-1)))

        return s_inputs, s_init, z_init

    def get_adj(
        self, f_input: FoldingInput, dtype: torch.dtype = torch.float32
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
