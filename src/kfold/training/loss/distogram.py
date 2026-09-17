# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from kfold.data.types.model_input import FoldingInput


class DistogramLoss(torch.nn.Module):
    def forward(
        self,
        distogram_out: dict[str, torch.Tensor],
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the  distogram loss.

        Parameters
        ----------
        distogram_out : dict[str, torch.Tensor]
            Distogram logits and distance-bin boundaries returned by the model.
        f_input : FoldingInput
            The input features containing the target distogram and masks.

        Returns
        -------
        distogram_loss : torch.Tensor
            The computed distogram loss of shape (B,).
        """

        logits = distogram_out["logits"]
        bin_boundaries = distogram_out["bin_boundaries"]
        num_bins = logits.shape[-1]

        with torch.autocast("cuda", enabled=False), torch.no_grad():
            gt_coords = f_input.token.repr_coords
            diff = gt_coords[..., None, :, :] - gt_coords[..., :, None, :]
            d_repr = diff.norm(dim=-1)  # [B, Lt, Lt]
            target_distogram = (d_repr.unsqueeze(-1) > bin_boundaries).sum(dim=-1).long()

        # Compute the distogram loss
        B, L, L = target_distogram.shape
        distogram_loss = torch.nn.functional.cross_entropy(
            logits.view(B * L * L, num_bins),
            target_distogram.view(B * L * L),
            reduction="none",
        ).view(B, L, L)

        # Mask out invalid distogram
        mask = f_input.token.repr_mask  # [B, Lt]
        pair_mask = mask[..., None, :] & mask[..., :, None]  # [B, Lt, Lt]
        pair_mask.diagonal(dim1=-2, dim2=-1).zero_()  # zero out diagonal
        pair_mask = pair_mask.float()

        # Compute mean loss
        sum_loss = (distogram_loss * pair_mask).sum((-1, -2))  # [B,]
        n_valid = pair_mask.sum((-1, -2)).clamp(1)  # [B,]
        return sum_loss / n_valid  # [B,]
