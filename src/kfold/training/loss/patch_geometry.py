import torch
import torch.nn.functional as F


class PatchPairGeometryLoss(torch.nn.Module):
    def __init__(
        self,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        near_cutoff: float = 12.0,
    ) -> None:
        super().__init__()
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.num_bins: int = num_bins
        self.near_cutoff: float = near_cutoff

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

        log_prob = F.log_softmax(logits, dim=-1)
        ce = -log_prob.gather(dim=-1, index=target[:, None]).squeeze(-1)

        pair_count = logits.new_tensor(target.numel())
        ce_loss = ce.sum() / pair_count.clamp(min=1.0)

        p_near = torch.logsumexp(
            log_prob[..., : self.near_bin + 1],
            dim=-1,
        ).exp()
        with torch.no_grad():
            near_target = target <= self.near_bin
            far_target = ~near_target
            near_count = near_target.to(logits.dtype).sum()
            far_count = far_target.to(logits.dtype).sum()

        metrics = {
            "patch_geometry_loss": ce_loss.detach(),
            "patch_geometry_ce_loss": ce_loss.detach(),
            "patch_geometry_valid_pairs": pair_count.detach(),
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
        }
        return ce_loss, metrics
