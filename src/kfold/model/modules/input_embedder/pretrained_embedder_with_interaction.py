"""Input embedder with interaction-aware pair representation."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

import kfold.constants as C
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.primitives import LinearNoBias
from kfold.utils.registry import INPUT_EMBEDDER

from .pretrained_embedder import PretrainedInputEmbedder


def compute_pair_interactions(
    interaction_type: torch.Tensor, chain_type: torch.Tensor | None = None
) -> torch.Tensor:
    """Project primitive interaction types to 5 symmetric pair features.

    Parameters
    ----------
    interaction_type : torch.Tensor
        Multi-hot primitive interaction types of shape [B, L, T] or [L, T].
        Values are clamped to [0, 1] before projection.
    chain_type : torch.Tensor | None, optional
        Chain types of shape [B, L] or [L]. When provided, ligand tokens that
        still hold the placeholder "all-ones" primitive types are zeroed out to
        avoid producing dense, meaningless pair labels.

    Returns
    -------
    torch.Tensor
        Pair interaction features of shape [B, L, L, 5] (or [L, L, 5]).
        The last dimension follows ``C.PairInteractionType`` order.
    """
    interaction_type = interaction_type.float().clamp(min=0.0, max=1.0)
    if chain_type is not None:
        ligand_mask = chain_type == C.chain.ChainType.LIGAND.value
        num_active = interaction_type[..., 1:].sum(-1)
        placeholder_ligand = ligand_mask & (num_active == (C.NUM_INTERACTION_TYPES - 1))
        # Legacy NPZ files may encode ligand primitives as "all ones". Treat
        # those as unknown rather than as every possible interaction type.
        interaction_type = interaction_type.masked_fill(
            placeholder_ligand.unsqueeze(-1), 0.0
        )

    hi = interaction_type[..., C.InteractionType.HI]
    hbd = interaction_type[..., C.InteractionType.HBD]
    hba = interaction_type[..., C.InteractionType.HBA]
    sbc = interaction_type[..., C.InteractionType.SBC]
    sba = interaction_type[..., C.InteractionType.SBA]
    pp = interaction_type[..., C.InteractionType.PP]
    pc = interaction_type[..., C.InteractionType.PC]

    hydrophobic = hi[:, :, None] * hi[:, None, :]
    hydrogen_bond = hbd[:, :, None] * hba[:, None, :] + hba[:, :, None] * hbd[:, None, :]
    salt_bridge = sbc[:, :, None] * sba[:, None, :] + sba[:, :, None] * sbc[:, None, :]
    pi_pi = pp[:, :, None] * pp[:, None, :]
    pi_cation = pp[:, :, None] * pc[:, None, :] + pc[:, :, None] * pp[:, None, :]

    pair_interactions = torch.stack(
        [hydrophobic, hydrogen_bond, salt_bridge, pi_pi, pi_cation], dim=-1
    )
    return pair_interactions.clamp(min=0.0, max=1.0)


@INPUT_EMBEDDER.register()
class PretrainedInputEmbedderWithInteraction(PretrainedInputEmbedder):
    """Input embedding module with interaction-aware pair representation.

    This extends the pretrained embedder by:
    - adding a token-level interaction embedding into ``s_inputs``
    - optionally injecting pair interaction features into ``z_init``
    """

    class Config(PretrainedInputEmbedder.Config):
        """Configuration for interaction-aware input embeddings."""

        use_interaction: bool = True
        interaction_mode: str = "fused"
        channel_z_interaction: int = 32
        debug_interaction: bool = False
        debug_interaction_max_logs: int = 1

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.use_interaction = cfg.use_interaction
        self.interaction_mode = cfg.interaction_mode
        self.channel_z_interaction = cfg.channel_z_interaction
        self.debug_interaction = cfg.debug_interaction or (
            os.getenv("KFOLD_DEBUG_INTERACTION") == "1"
        )
        self.debug_interaction_max_logs = cfg.debug_interaction_max_logs
        self._interaction_debug_logs = 0

        if self.use_interaction:
            # Initialize to zero so that enabling the module does not
            # immediately perturb the pretrained baseline before training.
            self.plip_embedder = LinearNoBias(
                C.NUM_INTERACTION_TYPES, cfg.channel_s, init="zero"
            )

            if self.interaction_mode == "fused":
                # "Fused" mode adds a projected interaction bias directly into
                # the existing pair representation channel dimension.
                self.linear_z_interaction = LinearNoBias(
                    C.NUM_PAIR_INTERACTION_TYPES,
                    cfg.channel_z,
                    init="zero",
                )
            elif self.interaction_mode == "separate":
                # "Separate" mode returns an auxiliary pair tensor that the
                # trunk can merge with an explicit projection layer.
                self.linear_z_interaction = LinearNoBias(
                    C.NUM_PAIR_INTERACTION_TYPES,
                    cfg.channel_z_interaction,
                    init="default",
                )
            else:
                raise ValueError(
                    f"Unknown interaction_mode: {cfg.interaction_mode}. "
                    "Expected 'fused' or 'separate'."
                )

    def _add_token_interaction_embedding(
        self, s_inputs: torch.Tensor, f_input: FoldingInput
    ) -> torch.Tensor:
        """Add token-level interaction embedding into the single representation."""
        if not self.use_interaction:
            return s_inputs
        # Match dtype/device with the current representation for safe addition.
        plip_feat = f_input.token.interaction_type.to(dtype=s_inputs.dtype)
        plip_emb = self.plip_embedder(plip_feat)
        return s_inputs + plip_emb

    def forward(
        self,
        f_input: FoldingInput,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Compute initial single/pair representations with interaction features."""
        s_inputs, s_init, z_init = super().forward(f_input, **kwargs)

        if not self.use_interaction:
            return s_inputs, s_init, z_init

        interaction_type = f_input.token.interaction_type
        pair_interactions = compute_pair_interactions(
            interaction_type, f_input.token.chain_type
        )

        if self.interaction_mode == "fused":
            z_interaction = self.linear_z_interaction(pair_interactions)
            self._log_interaction_stats(
                f_input,
                interaction_type,
                pair_interactions,
                z_interaction,
                z_init,
            )
            # Inject interaction-aware bias into the base pair representation.
            z_init = z_init + z_interaction
            return s_inputs, s_init, z_init

        z_interaction_proj = self.linear_z_interaction(pair_interactions)
        # Keep raw pair features alongside their projection for downstream use.
        z_interaction = torch.cat([pair_interactions, z_interaction_proj], dim=-1)
        self._log_interaction_stats(
            f_input,
            interaction_type,
            pair_interactions,
            z_interaction,
            z_init,
        )
        return s_inputs, s_init, z_init, z_interaction

    def get_interaction_mask(self, f_input: FoldingInput) -> torch.Tensor:
        """Return pair interaction features suitable for masking/analysis."""
        interaction_type = f_input.token.interaction_type
        return compute_pair_interactions(interaction_type, f_input.token.chain_type)

    def _log_interaction_stats(
        self,
        f_input: FoldingInput,
        interaction_type: torch.Tensor,
        pair_interactions: torch.Tensor,
        z_interaction: torch.Tensor,
        z_init: torch.Tensor,
    ) -> None:
        """Log lightweight interaction statistics for debugging on rank 0."""
        if not self.debug_interaction:
            return
        if self._interaction_debug_logs >= self.debug_interaction_max_logs:
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        with torch.no_grad():
            pad_mask = f_input.token.pad_mask
            valid_tokens = int(pad_mask.sum().item())
            if valid_tokens == 0:
                return

            raw_min = int(interaction_type.min().item())
            raw_max = int(interaction_type.max().item())

            interaction_type_clean = interaction_type.float().clamp(min=0.0, max=1.0)
            per_token = interaction_type_clean.sum(-1)
            active_tokens = int((per_token > 0).masked_select(pad_mask).sum().item())

            pair_mask = pad_mask[:, :, None] & pad_mask[:, None, :]
            valid_pairs = int(pair_mask.sum().item())
            pair_any = pair_interactions.sum(-1) > 0
            active_pairs = int((pair_any & pair_mask).sum().item())

            pair_counts = (pair_interactions * pair_mask.unsqueeze(-1)).sum(dim=(0, 1, 2))
            pair_density = (pair_counts / max(valid_pairs, 1)).tolist()
            pair_density_str = ",".join(f"{v:.3e}" for v in pair_density)

            z_abs_mean = float(z_interaction.abs().mean().item())
            z_abs_max = float(z_interaction.abs().max().item())
            z_init_abs_mean = float(z_init.abs().mean().item())
            z_ratio = z_abs_mean / (z_init_abs_mean + 1e-8)

            token_density = active_tokens / max(valid_tokens, 1)
            pair_density_any = active_pairs / max(valid_pairs, 1)

            print(
                "[DEBUG][interaction] "
                f"tokens={valid_tokens}, tokens_with_type={active_tokens} "
                f"(density={token_density:.3e}), "
                f"pairs={valid_pairs}, pairs_with_type={active_pairs} "
                f"(density={pair_density_any:.3e}), "
                f"interaction_type_min={raw_min}, interaction_type_max={raw_max}, "
                f"pair_type_density=[{pair_density_str}], "
                f"z_interaction_abs_mean={z_abs_mean:.3e}, "
                f"z_interaction_abs_max={z_abs_max:.3e}, "
                f"z_interaction_to_z_init_ratio={z_ratio:.3e}"
            )

        self._interaction_debug_logs += 1
