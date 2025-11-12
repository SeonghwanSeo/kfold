from abc import ABC, abstractmethod

import torch

from kfold.data.model_input import FoldingInput
from kfold.utils.registry import INPUT_EMBEDDER


@INPUT_EMBEDDER.register()
class BaseInputEmbedder(torch.nn.Module, ABC):
    """Base class for input feature embedding modules.
    See Section 3 Algorithm 1 Line [1-5] of AlphaFold3 paper.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        f_input: FoldingInput,
        model_cache: dict | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of embedding module.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        model_cache : dict, optional
            Dictionary for caching intermediate results, by default None.
        **kwargs : dict
            Additional keyword arguments.

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
