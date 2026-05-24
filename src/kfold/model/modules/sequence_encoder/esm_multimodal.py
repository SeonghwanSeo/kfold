import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import SEQUENCE_ENCODER

from .esm import ESM


@SEQUENCE_ENCODER.register()
class ESM_MultiModal(ESM):
    """ESMC sequence encoder for protein, dna, and rna sequences."""

    def prepare_emb_mask(self, f_input: FoldingInput) -> torch.Tensor:
        """Prepare output mask"""
        is_polymer = ~f_input.token.is_ligand
        return f_input.token.pad_mask & is_polymer
