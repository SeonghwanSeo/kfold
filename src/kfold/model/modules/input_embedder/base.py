from abc import ABC, abstractmethod

import torch

from kfold.data.model_input import FoldingInput
from kfold.utils.registry import INPUT_EMBEDDER


@INPUT_EMBEDDER.register()
class BaseInputEmbedder(torch.nn.Module, ABC):
    """Base class for input feature embedding modules.
    See Section 3.1.1: InputEmbedder, Algorithm 2 of AlphaFold3 paper.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of embedding module.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.

        Returns
        -------
        s : torch.Tensor
            Tensor of shape (L, C_s) containing sequence embeddings.
        """
