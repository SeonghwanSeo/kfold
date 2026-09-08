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
