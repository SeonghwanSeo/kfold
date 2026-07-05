import torch
import torch.nn.functional as F


def _binary_average_precision(
    score: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    score = score[mask]
    target = target[mask]
    if score.numel() == 0 or not target.any() or target.all():
        return None

    order = torch.argsort(score, descending=True)
    sorted_target = target[order].to(score.dtype)
    rank = torch.arange(
        1, sorted_target.numel() + 1, device=score.device, dtype=score.dtype
    )
    precision = sorted_target.cumsum(dim=0) / rank
    return (precision * sorted_target).sum() / sorted_target.sum().clamp(min=1.0)


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
        valid_mask = patch_output["valid_mask"]
        hard_negative = patch_output["hard_negative"] & valid_mask

        if logits.numel() == 0 or not valid_mask.any():
            zero = logits.sum() * 0.0
            return zero, {
                "patch_geometry_loss": zero.detach(),
                "patch_geometry_ce_loss": zero.detach(),
                "patch_geometry_hard_negative_loss": zero.detach(),
                "patch_geometry_valid_pairs": zero.detach(),
            }

        b, m, _ = logits.shape
        ce = F.cross_entropy(
            logits.reshape(b * m, self.num_bins),
            target.reshape(b * m),
            reduction="none",
        ).view(b, m)

        valid_weight = weight * valid_mask.to(weight.dtype)
        denom = valid_weight.sum().clamp(min=1.0)
        ce_loss = (ce * valid_weight).sum() / denom

        prob = torch.softmax(logits, dim=-1)
        p_near = prob[..., : self.near_bin + 1].sum(dim=-1)
        hard_weight = valid_weight * hard_negative.to(valid_weight.dtype)
        hard_denom = hard_weight.sum().clamp(min=1.0)
        hard_loss = (
            -torch.log1p(-p_near.clamp(max=1.0 - self.eps)) * hard_weight
        ).sum() / hard_denom

        loss = ce_loss + self.hard_negative_weight * hard_loss
        with torch.no_grad():
            near_target = (target <= self.near_bin) & valid_mask
            far_target = (target > self.near_bin) & valid_mask
            ap = _binary_average_precision(p_near.detach(), near_target, valid_mask)

        metrics = {
            "patch_geometry_loss": loss.detach(),
            "patch_geometry_ce_loss": ce_loss.detach(),
            "patch_geometry_hard_negative_loss": hard_loss.detach(),
            "patch_geometry_valid_pairs": valid_mask.float().sum().detach(),
            "patch_geometry_hard_negative_pairs": hard_negative.float().sum().detach(),
            "patch_geometry_near_pairs": near_target.float().sum().detach(),
            "patch_geometry_far_pairs": far_target.float().sum().detach(),
            "patch_geometry_target_near_rate": near_target.float().sum().detach()
            / valid_mask.float().sum().clamp(min=1.0).detach(),
            "patch_geometry_pred_near_mass": p_near[valid_mask].mean().detach(),
        }
        if ap is not None:
            metrics["patch_geometry_near_ap"] = ap.detach()
        if near_target.any():
            metrics["patch_geometry_p_near_true_near"] = (
                p_near[near_target].mean().detach()
            )
        if far_target.any():
            metrics["patch_geometry_false_positive_near_mass"] = (
                p_near[far_target].mean().detach()
            )
        if hard_negative.any():
            metrics["patch_geometry_hard_negative_p_near"] = (
                p_near[hard_negative].mean().detach()
            )
        return loss, metrics
