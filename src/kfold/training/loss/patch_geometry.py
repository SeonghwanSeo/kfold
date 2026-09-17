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
import torch.nn.functional as F


class PatchPairGeometryLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        patch_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = patch_output["logits"]
        target = patch_output["target"]
        weight = patch_output["weight"]
        hard_negative = patch_output["hard_negative"]

        log_prob = F.log_softmax(logits, dim=-1)
        ce = -log_prob.gather(dim=-1, index=target[:, None]).squeeze(-1)

        denom = weight.sum().clamp(min=1.0)
        ce_loss = (ce * weight).sum() / denom

        with torch.no_grad():
            pair_count = torch.ones_like(weight).sum()
            hard_count = hard_negative.to(logits.dtype).sum()

        metrics = {
            "patch_geometry_loss": ce_loss.detach(),
            "patch_geometry_ce_loss": ce_loss.detach(),
            "patch_geometry_valid_pairs": pair_count.detach(),
            "patch_geometry_hard_negative_pairs": hard_count.detach(),
        }
        return ce_loss, metrics
