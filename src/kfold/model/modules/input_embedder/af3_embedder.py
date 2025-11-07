from collections.abc import Callable

import torch
import torch.nn.functional as F

import kfold.constants as C
from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.primitives import LinearNoBias
from kfold.model.layers.alphafold3.transformers import AtomAttentionEncoder
from kfold.utils.registry import INPUT_EMBEDDER, BaseConfig

from .base import BaseInputEmbedder


@INPUT_EMBEDDER.register()
class AF3InputEmbedder(BaseInputEmbedder):
    class Config(BaseConfig):
        """Configuration for the Input embedding module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_atom : int
            The token single embedding size.
        channel_atompair : int
            The token pairwise embedding size.
        atoms_per_window_queries: int,
            The number of atoms per window for queries.
        atoms_per_window_keys: int,
            The number of atoms per window for keys.
        atom_encoder_depth: int,
            The atom encoder depth.
        atom_encoder_heads: int,
            The atom encoder heads.
        """

        channel_s: int = 384
        channel_atom: int = 128
        channel_atompair: int = 16
        atoms_per_window_queries: int = 32
        atoms_per_window_keys: int = 128
        atom_encoder_depth: int = 3
        atom_encoder_heads: int = 4

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

        self.encoder = AtomAttentionEncoderWithoutStructure(
            channel_s=cfg.channel_s,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            channel_token=cfg.channel_s,  # Same to channel_s
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            atoms_per_window_keys=cfg.atoms_per_window_keys,
            num_blocks=cfg.atom_encoder_depth,
            num_heads=cfg.atom_encoder_heads,
        )

        # residue info
        self.num_res_types: int = C.NUM_RES_TYPES

        # out projection
        s_input_dim = cfg.channel_s + self.num_res_types + 32 + 1
        self.s_init = LinearNoBias(s_input_dim, cfg.channel_s)

    def forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        input : FoldingInput
            Input features

        Returns
        -------
        Tensor
            The embedded tokens. [L, c_s]
        """
        # FIXME: add more

        # Atom attention encoder forward
        a, *_ = self.encoder(f_input)  # [L, c_s]

        # Concatenate additional token features
        res_type = f_input.token.res_type  # [L,]
        profile = f_input.msa.profile  # [L,]
        deletion_mean = f_input.msa.deletion_mean  # [L,]
        s = torch.cat(
            [
                a,
                F.one_hot(res_type, self.num_res_types).float(),
                F.one_hot(profile, 32).float(),
                deletion_mean.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Project to model dimension
        # NOTE: (SeonghwanSeo) I introduce additional linear layer to unify the dimension.
        s = self.s_init(s)  # [L, c_s]

        return s


class AtomAttentionEncoderWithoutStructure(AtomAttentionEncoder):
    """Atom attention encoder without structure information.
    AlphaFold3 Algorithm 5 without noisy structure r_l.
    """

    def __init__(
        self,
        channel_s: int,
        channel_atom: int,
        channel_atompair: int,
        channel_token: int,
        num_blocks: int = 3,
        num_heads: int = 4,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        activation_checkpointing=False,
    ):
        super().__init__(
            channel_s=channel_s,
            channel_z=0,  # no pair embedding used in input embedding
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            channel_token=channel_token,
            num_blocks=num_blocks,
            num_heads=num_heads,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            use_structure=False,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        f_input: FoldingInput,
        s_trunk: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
        model_cache: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Callable]:
        assert s_trunk is None and z is None and r is None, (
            "s_trunk, z_trunk, r must be None"
        )
        assert model_cache is None, "model_cache must be None in input embedding."
        a, q, c, p, to_keys = super().forward(f_input, s_trunk, z, r, model_cache)

        assert a.shape[0] == 1, "Batch size must be 1 for input embedding."

        # Squeeze batch dimension
        a, q, c, p = a.squeeze(0), q.squeeze(0), c.squeeze(0), p.squeeze(0)
        return a, q, c, p, to_keys
