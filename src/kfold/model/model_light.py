"""Sequence-only KFold model variant."""

import torch

from kfold.data.types.model_input import FoldingInput

from .model import KFold, KFoldConfig


class KFold_Light(KFold):
    """KFold without protein structure-encoder conditioning."""

    def __init__(self, config: KFoldConfig):
        super().__init__(config)
        del self.prot_struct_encoder
        del self.prot_struct_to_s_lm

    def _encode_lm_single(self, f_input: FoldingInput) -> torch.Tensor:
        """Merge sequence encoder features into the shared LM single."""
        prot_seq_encoder = self._get_model_module(self.prot_seq_encoder)
        rna_seq_encoder = self._get_model_module(self.rna_seq_encoder)
        prot_seq_to_s_lm = self._get_model_module(self.prot_seq_to_s_lm)
        rna_seq_to_s_lm = self._get_model_module(self.rna_seq_to_s_lm)
        return prot_seq_to_s_lm(prot_seq_encoder(f_input)) + rna_seq_to_s_lm(
            rna_seq_encoder(f_input)
        )

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
