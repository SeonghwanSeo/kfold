from abc import ABC, abstractmethod

import torch

from kfold.utils.registry import STRUCTURE_ENCODER


@STRUCTURE_ENCODER.register()
class BaseStructureEncoder(torch.nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        coords: torch.Tensor,
        atom_type: torch.Tensor,
        token_type: torch.Tensor,
        token_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of structure representation module.

        Parameters
        ----------
        coords : torch.Tensor (float)
            Tensor of shape (B, Nconf, Natom, 3) containing atomic structure.
        atom_type : torch.Tensor (int)
            Tensor of shape (B, Natom) containing atom types.
        token_type : torch.Tensor (int)
            Tensor of shape (B, Natom) containing token types.
        token_id : torch.Tensor (int)
            Tensor of shape (B, Natom) containing token IDs.
        mask : torch.Tensor (bool)
            Tensor of shape (B, Natom) containing mask for valid atoms.

        Returns
        -------
        s: torch.Tensor
            Tensor of shape (B, L, c_s) containing single feature
        z: torch.Tensor
            Tensor of shape (B, L, L, c_z) containing pair feature

        # NOTE (seonghwanseo):
        1. If you want to use more features, you can consider to use
            `kfold.data.model_input.FoldingInput` or `kfold.data.model_input.ChainInput`,
            like `kfold.model.modules.transformer.BaseTransformer`.

        2. You don't have to match c_s and c_z with transformer module. We will
            use projection layers to match the dimensions.

        # TODO (seonghwanseo):
        1. Do we have to return both atom-level and token-level
            embedding? We may want to discuss more.
        """
