import logging
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules import (
    confidence_head,
    distogram_head,
    input_embedder,
    sequence_encoder,
    tri_stack,
)
from kfold.model.modules.structure import score_model
from kfold.model.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import MAIN_MODULE, Registry

from .model import KFold, KFoldConfig, LMToPair

logger = logging.getLogger(__name__)


@MAIN_MODULE.register()
class KFold_Light(KFold):
    def __init__(self, config: KFoldConfig):
        torch.nn.Module.__init__(self)
        self.config: KFoldConfig = config
        self.channel_s: int = config.channel_s
        self.channel_z: int = config.channel_z
        self.dropout: float = config.dropout

        kernel_config = {
            "cuequivariance": config.kernel_cuequivariance,
        }
        self.kernel_config = kernel_config

        # Initialize input featurizer.
        self.input_embedder = input_embedder.InputEmbedder(config.input_embedder)

        # Initialize pre-trained sequence and structure encoders.
        self.prot_seq_encoder = sequence_encoder.SequenceEncoder(
            config.protein_sequence_encoder
        )
        self.rna_seq_encoder = sequence_encoder.SequenceEncoder(
            config.rna_sequence_encoder
        )

        self.prot_seq_to_pair = LMToPair(
            self.prot_seq_encoder.d_model, self.prot_seq_encoder.n_layers, self.channel_z
        )
        self.rna_seq_to_pair = LMToPair(
            self.rna_seq_encoder.d_model, self.rna_seq_encoder.n_layers, self.channel_z
        )

        # Initialize trunk
        self.layernorm_z = LayerNorm(self.channel_z)
        self.linear_z = LinearNoBias(self.channel_z, self.channel_z, init="final")
        self.lm_stack = tri_stack.TrianglularStack(
            self.channel_z,
            config.trunk.num_lm_blocks,
            config.trunk.dropout,
        )
        self.main_stack = tri_stack.TrianglularStack(
            self.channel_z,
            config.trunk.num_main_blocks,
            config.trunk.dropout,
            blocks_per_ckpt=config.trunk.blocks_per_ckpt,
        )
        # Recyling
        self.linear_refine = LinearNoBias(self.channel_z, self.channel_z, init="identity")
        self.refine_stack = tri_stack.TrianglularStack(
            self.channel_z,
            config.trunk.num_refine_blocks,
            config.trunk.dropout,
        )

        # Initialize prediction heads
        self.score_model = score_model.DiffusionModule(
            config.score_model, kernel_config=kernel_config
        )
        # NOTE: diffusion_head is not a torch.nn.Module
        # TODO: After we fix the diffusion algorith, remove Registry.instantiate
        self.diffusion_head = Registry.instantiate(
            config.diffusion_head, score_model=self.score_model
        )
        self.distogram_head = distogram_head.DistogramHead(config.distogram_head)
        self.confidence_head = confidence_head.ConfidenceHead(
            config.confidence_head, kernel_config=kernel_config
        )

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
        """Forward pass of KFold model for model training.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model.
        apo_dict : dict[int, dict]
            Dictionary mapping entity_id to apo structure information.
        num_recycles : int
            Number of recycling cycles in trunk.
        num_steps : int
            Number of diffusion steps for training.
        num_samples : int
            Number of diffusion samples for training.

        Returns
        -------
        model_out : dict[str, dict[str, torch.Tensor]]
            Output dictionary containing sampled structures and intermediate features:
            - trunk: intermediate trunk outputs. (optional)
            - distogram: predicted distogram logits.
            - diffusion: sampled structures from diffusion head.
            - confidence: predicted confidence metrics from confidence head.

        time_logs : dict[str, float]
            Dictionary containing time taken for each module during sampling.
        """
        # If input is not batched, add batch dimension for processing
        # and remove it from output at the end.
        if f_input.is_batched:
            return_batched_output = True
        else:
            f_input = f_input.add_batch_dim()
            return_batched_output = False

        if f_input.batch_size != 1:
            # TODO: Support batched inference.
            raise NotImplementedError(
                "Batched input with batch_size > 1 is not supported for inference yet."
            )

        # Sample structures
        model_out, time_logs = self.sample(
            f_input,
            num_recycles,
            num_steps,
            num_samples,
            return_embeddings=return_embeddings,
            return_traj=return_traj,
        )

        # remove batch dimension
        if not return_batched_output:
            model_out = {
                k: {kk: vv.squeeze(0) for kk, vv in v.items()}
                for k, v in model_out.items()
            }

        return model_out, time_logs

    def run_trunk(
        self,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z_init: torch.Tensor
            Tensor of shape (B, L, L, C_z) containing initial pair representation
        f_input : FoldingInput
            The input features.
        num_recycles : int
            The number of recycling steps.

        Returns
        -------
        z: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """
        use_cuequiv_kernels = self.kernel_config.get("cuequivariance", False)

        # Initial pairwise representation from pretrained encoders.
        z_lm = self.prot_seq_to_pair(self.prot_seq_encoder(f_input))
        z_lm += self.rna_seq_to_pair(self.rna_seq_encoder(f_input))

        # === Main trunk iteration with recycling === #
        z = torch.zeros_like(z_init)
        token_mask = f_input.token.pad_mask
        pair_mask = token_mask[..., None] & token_mask[..., None, :]

        for _ in range(0, num_recycles + 1):
            _z_lm = F.dropout(z_lm, p=self.dropout)
            _z = z_init + self.lm_stack(_z_lm, pair_mask, use_cuequiv_kernels)
            z += self.linear_z(self.layernorm_z(_z))
            z = self.main_stack(z, pair_mask, use_cuequiv_kernels)

        # Refinement iteration
        z = self.linear_refine(z)
        z = self.refine_stack(self.linear_refine(z), pair_mask, use_cuequiv_kernels)

        return z

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load state dict without pretrained sequence encoder"""
        # If strict is False, it is fine to have missing keys (e.g., pretrained model)
        incompatible_keys = super().load_state_dict(state_dict, strict=False)
        if strict:
            missing_keys = incompatible_keys.missing_keys
            unexpected_keys = incompatible_keys.unexpected_keys
            # If the sequence encoder is pretrained and not included in the state dict,
            # missing keys starting with "sequence_encoder." or "structure_encoder." are
            # allowed.
            missing_keys = {
                k
                for k in missing_keys
                if not k.startswith(("prot_seq_encoder.", "rna_seq_encoder."))
            }
            # Light model does not have a structure encoder.
            unexpected_keys = {
                k for k in unexpected_keys if not k.startswith(("prot_struct_to_pair.",))
            }
            if missing_keys:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")
        return incompatible_keys
