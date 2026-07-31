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
        return (self.prot_seq_to_s_lm(self.prot_seq_encoder(f_input))) + (
            self.rna_seq_to_s_lm(self.rna_seq_encoder(f_input))
        )
