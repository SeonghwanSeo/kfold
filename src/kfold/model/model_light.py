import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import MAIN_MODULE

from .model import KFold, KFoldConfig

logger = logging.getLogger(__name__)


@MAIN_MODULE.register()
class KFold_Light(KFold):
    def __init__(self, config: KFoldConfig):
        super().__init__(config)
        del self.prot_struct_encoder
        del self.prot_struct_to_s_lm

    def do_compile(self, mode: str = "default", dynamic: bool = False):
        """Compile the light trunk and heads."""
        opts = {"mode": mode, "dynamic": dynamic}
        self.is_compiled = True
        self.prot_seq_encoder = torch.compile(self.prot_seq_encoder, **opts)
        self.rna_seq_encoder = torch.compile(self.rna_seq_encoder, **opts)

        self.prot_seq_to_s_lm = torch.compile(self.prot_seq_to_s_lm, **opts)
        self.rna_seq_to_s_lm = torch.compile(self.rna_seq_to_s_lm, **opts)

        self.lm_to_pair = torch.compile(self.lm_to_pair, **opts)
        self.lm_stack = torch.compile(self.lm_stack, **opts)
        self.main_stack = torch.compile(self.main_stack, **opts)
        self.refine_stack = torch.compile(self.refine_stack, **opts)
        self.score_model.do_compile(**opts)
        self.confidence_head.do_compile(**opts)

    @torch.inference_mode()
    def inference(
        self,
        f_input: FoldingInput,
        apo_dict: dict[int, dict],
        num_recycles: int = 10,
        num_steps: int = 200,
        num_samples: int = 5,
        return_embeddings: bool = False,
        return_traj: bool = False,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
        """Run light-model inference without apo structure tokenization."""
        return self.sample(
            f_input,
            num_recycles,
            num_steps,
            num_samples,
            return_embeddings=return_embeddings,
            return_traj=return_traj,
        )

    def run_trunk(
        self,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
        grad_recurrence_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk using only sequence encoders for the LM single state."""
        use_cuequiv_kernels = self.kernel_config.get("cuequivariance", False)

        def get_model(mod: torch.nn.Module) -> torch.nn.Module:
            return mod._orig_mod if (self.is_compiled and not self.training) else mod

        prot_seq_encoder = get_model(self.prot_seq_encoder)
        rna_seq_encoder = get_model(self.rna_seq_encoder)
        prot_seq_to_s_lm = get_model(self.prot_seq_to_s_lm)
        rna_seq_to_s_lm = get_model(self.rna_seq_to_s_lm)
        lm_to_pair = get_model(self.lm_to_pair)
        lm_stack = get_model(self.lm_stack)
        main_stack = get_model(self.main_stack)
        refine_stack = get_model(self.refine_stack)

        s_lm = prot_seq_to_s_lm(prot_seq_encoder(f_input)) + rna_seq_to_s_lm(
            rna_seq_encoder(f_input)
        )
        z_lm = lm_to_pair(s_lm)

        z = self._init_parcae_pair_state(z_init)
        token_mask = f_input.token.pad_mask
        pair_mask = token_mask[..., None] & token_mask[..., None, :]

        a, b = self._parcae_discretized_dynamics()
        a = a.view(*((1,) * (z_init.ndim - 1)), -1).to(
            device=z_init.device, dtype=z_init.dtype
        )
        b = b.to(device=z_init.device, dtype=z_init.dtype)

        grad_recurrence_steps = max(1, int(grad_recurrence_steps))
        grad_start = max(0, num_recycles + 1 - grad_recurrence_steps)

        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i >= grad_start
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()
                _z_lm = F.dropout(z_lm, p=self.dropout)
                u_t = z_init + lm_stack(_z_lm, pair_mask, use_cuequiv_kernels)
                z = a * z + F.linear(self.layernorm_z(u_t), b)
                z = main_stack(z, pair_mask, use_cuequiv_kernels)

        z = refine_stack(self.linear_refine(z), pair_mask, use_cuequiv_kernels)

        return z, s_lm

    def get_pretrained_module_names(self) -> list[str]:
        """Get the names of pretrained modules used by the light model."""
        return [
            "prot_seq_encoder",
            "rna_seq_encoder",
        ]

    def get_trunk_module_names(self) -> list[str]:
        """Get the names of trunk modules used by the light model."""
        return [
            "input_embedder",
            "prot_seq_to_s_lm",
            "rna_seq_to_s_lm",
            "lm_to_pair",
            "layernorm_z",
            "lm_stack",
            "main_stack",
            "linear_refine",
            "refine_stack",
            "patch_pair_geometry_head",
        ]

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load checkpoints while ignoring structure-only light model omissions."""
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith(("prot_struct_encoder.", "prot_struct_to_s_lm."))
        }
        incompatible_keys: Any = super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )
        return incompatible_keys
