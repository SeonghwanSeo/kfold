import torch
import torch.nn.functional as F


class PatchPairGeometryLoss(torch.nn.Module):
    def __init__(
        self,
        near_cutoff: float = 12.0,
        ranking_weight: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.near_cutoff: float = near_cutoff
        self.ranking_weight: float = ranking_weight
        self.eps: float = eps
        if ranking_weight < 0:
            raise ValueError("ranking_weight must be non-negative.")

    def _ranking_loss(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank contact-density targets within each unordered chain pair."""
        group_losses: list[torch.Tensor] = []
        weighted_correct = score.new_zeros(())
        ranking_weight_sum = score.new_zeros(())
        ranking_pair_count = score.new_zeros(())

        for group_id in torch.unique(group):
            in_group = group == group_id
            group_score = score[in_group]
            if group_score.numel() < 2:
                continue
            group_target = target[in_group]

            target_diff = group_target[:, None] - group_target[None, :]
            score_diff = group_score[:, None] - group_score[None, :]
            upper_triangle = torch.ones_like(target_diff, dtype=torch.bool).triu(
                diagonal=1
            )
            comparable = upper_triangle & (target_diff != 0)
            if not comparable.any():
                continue

            comparison_weight = target_diff.abs() * comparable.to(score.dtype)
            denom = comparison_weight.sum().clamp(min=self.eps)
            signed_score_diff = target_diff.sign() * score_diff
            group_losses.append(
                (F.softplus(-signed_score_diff) * comparison_weight).sum() / denom
            )
            weighted_correct = (
                weighted_correct
                + ((signed_score_diff > 0).to(score.dtype) * comparison_weight).sum()
            )
            ranking_weight_sum = ranking_weight_sum + comparison_weight.sum()
            ranking_pair_count = ranking_pair_count + comparable.to(score.dtype).sum()

        ranking_loss = (
            torch.stack(group_losses).mean() if group_losses else score.sum() * 0.0
        )
        ranking_accuracy = weighted_correct / ranking_weight_sum.clamp(min=self.eps)
        ranking_group_count = score.new_tensor(len(group_losses))
        return ranking_loss, ranking_accuracy, ranking_pair_count, ranking_group_count

    def forward(
        self,
        patch_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = patch_output["logits"]
        bin_boundaries = patch_output["bin_boundaries"]
        target = patch_output["target"]
        near_bin = int((bin_boundaries < self.near_cutoff).sum().item()) - 1
        near_bin = max(0, min(near_bin, logits.shape[-1] - 1))

        log_prob = F.log_softmax(logits, dim=-1)
        ce = -log_prob.gather(dim=-1, index=target[:, None]).squeeze(-1)

        pair_count = logits.new_tensor(target.numel())
        ce_loss = ce.sum() / pair_count.clamp(min=1.0)

        ranking_loss = logits.sum() * 0.0
        ranking_accuracy = logits.new_zeros(())
        ranking_pair_count = logits.new_zeros(())
        ranking_group_count = logits.new_zeros(())
        if self.ranking_weight > 0:
            try:
                (
                    ranking_loss,
                    ranking_accuracy,
                    ranking_pair_count,
                    ranking_group_count,
                ) = self._ranking_loss(
                    score=patch_output["rank_score"],
                    target=patch_output["contact_strength"],
                    group=patch_output["rank_group"],
                )
            except KeyError as exc:
                raise KeyError(
                    "Patch ranking requires rank_score, contact_strength, and "
                    "rank_group model outputs."
                ) from exc

        loss = ce_loss + self.ranking_weight * ranking_loss

        p_near = torch.logsumexp(
            log_prob[..., : near_bin + 1],
            dim=-1,
        ).exp()
        with torch.no_grad():
            near_target = target <= near_bin
            far_target = ~near_target
            near_count = near_target.to(logits.dtype).sum()
            far_count = far_target.to(logits.dtype).sum()

        metrics = {
            "patch_geometry_loss": loss.detach(),
            "patch_geometry_ce_loss": ce_loss.detach(),
            "patch_geometry_ranking_loss": ranking_loss.detach(),
            "patch_geometry_ranking_accuracy": ranking_accuracy.detach(),
            "patch_geometry_ranking_pairs": ranking_pair_count.detach(),
            "patch_geometry_ranking_groups": ranking_group_count.detach(),
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
        return loss, metrics
