from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from kfold.data.model_input import FoldingInput
from kfold.utils.registry import DISTOGRAM_HEAD, BaseConfig


@dataclass
class BaseConfidenceHeadConfig(BaseConfig):
    """Base configuration class for confidence head modules."""

    c_s: int = 384
    c_z: int = 128
    max_distance: float = 22.0
    num_plddt_bins: int = 50
    num_pde_bins: int = 64
    num_pae_bins: int = 64


@DISTOGRAM_HEAD.register(config_cls=BaseConfidenceHeadConfig)
class BaseConfidenceHead(torch.nn.Module, ABC):
    def __init__(self, cfg: BaseConfidenceHeadConfig):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        input: FoldingInput,
        s_input: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of distogram head module.

        Parameters
        ----------
        input : FoldingInput
            FoldingInput object containing model inputs.
        s_input : torch.Tensor
            Tensor of shape (B, N, c_s) containing input single feature.
        s_trunk : torch.Tensor
            Tensor of shape (B, N, c_s) containing trunk single feature.
        z_trunk : torch.Tensor
            Tensor of shape (B, N, N, c_z) containing trunk pair feature.
        x_pred : torch.Tensor
            Tensor of shape (B, N, 3) containing predicted coordinates.

        Returns
        -------
        confidences: dict[str, torch.Tensor]
            A dictionary containing:
            - "plddt": Tensor of shape (B, Natom, plddt_bins) containing pLDDT logits.
            - "pde": Tensor of shape (B, N, N, pde_bins) containing pDE logits.
            - "pae": Tensor of shape (B, N, N, pae_bins) containing pAE logits.
            - "resolved": Tensor of shape (B, Natom) containing resolved scores.
        """
