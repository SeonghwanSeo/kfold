import torch
import torch.nn.functional as F


class PatchPairGeometryLoss(torch.nn.Module):
    def __init__(
        self,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        near_cutoff: float = 12.0,
        hard_negative_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.num_bins: int = num_bins
        self.near_cutoff: float = near_cutoff
        self.hard_negative_weight: float = hard_negative_weight
        self.eps: float = eps

        bin_size = (max_dist - min_dist) / num_bins
        first_bin = min_dist + bin_size
        self.near_bin = int((near_cutoff - first_bin) / bin_size)
        self.near_bin = max(0, min(self.near_bin, num_bins - 1))

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

        p_near = torch.logsumexp(
            log_prob[..., : self.near_bin + 1],
            dim=-1,
        ).exp()
        hard_weight = weight * hard_negative.to(weight.dtype)
        hard_denom = hard_weight.sum().clamp(min=1.0)
        hard_loss = (
            -torch.log1p(-p_near.clamp(max=1.0 - self.eps)) * hard_weight
        ).sum() / hard_denom

        loss = ce_loss + self.hard_negative_weight * hard_loss
        with torch.no_grad():
            near_target = target <= self.near_bin
            far_target = ~near_target
            pair_count = torch.ones_like(weight).sum()
            near_count = near_target.to(weight.dtype).sum()
            far_count = far_target.to(weight.dtype).sum()
            hard_count = hard_negative.to(weight.dtype).sum()

        metrics = {
            "patch_geometry_loss": loss.detach(),
            "patch_geometry_ce_loss": ce_loss.detach(),
            "patch_geometry_hard_negative_loss": hard_loss.detach(),
            "patch_geometry_valid_pairs": pair_count.detach(),
            "patch_geometry_hard_negative_pairs": hard_count.detach(),
            "patch_geometry_near_pairs": near_count.detach(),
            "patch_geometry_far_pairs": far_count.detach(),
            "patch_geometry_target_near_rate": (
                near_count / pair_count.clamp(min=1.0)
            ).detach(),
            "patch_geometry_pred_near_mass": (
                p_near.sum() / pair_count.clamp(min=1.0)
            ).detach(),
            "patch_geometry_p_near_true_near": (
                (p_near * near_target).sum() / near_count.clamp(min=1.0)
            ).detach(),
            "patch_geometry_false_positive_near_mass": (
                (p_near * far_target).sum() / far_count.clamp(min=1.0)
            ).detach(),
            "patch_geometry_hard_negative_p_near": (
                (p_near * hard_negative).sum() / hard_count.clamp(min=1.0)
            ).detach(),
        }
        return loss, metrics
