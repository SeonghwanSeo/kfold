import math

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput


class InterfaceContactBalancedLoss(torch.nn.Module):
    """Reweight exact-bin distogram CE on hard inter-chain pairs."""

    def __init__(
        self,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        contact_cutoff: float = 8.0,
        positive_weight: float = 0.75,
        negative_weight: float = 0.25,
        focal_gamma: float = 2.0,
        fp_budget_ratio: float = 4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not min_dist < contact_cutoff < max_dist:
            raise ValueError("Expected min_dist < contact_cutoff < max_dist.")
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than one.")
        if positive_weight <= 0 or negative_weight < 0:
            raise ValueError("Loss weights must be non-negative with positive FN weight.")
        if not math.isclose(positive_weight + negative_weight, 1.0):
            raise ValueError("positive_weight and negative_weight must sum to one.")
        if focal_gamma < 0 or fp_budget_ratio <= 0 or eps <= 0:
            raise ValueError(
                "Gamma must be non-negative; budget and eps must be positive."
            )

        self.num_bins = num_bins
        self.contact_cutoff = contact_cutoff
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.focal_gamma = focal_gamma
        self.fp_budget_ratio = fp_budget_ratio
        self.eps = eps

        bin_size = (max_dist - min_dist) / num_bins
        first_bin = min_dist + bin_size
        last_bin = max_dist - bin_size
        boundaries = torch.linspace(first_bin, last_bin, num_bins - 1)
        self.register_buffer("boundaries", boundaries, persistent=False)

        contact_bin = int((contact_cutoff - first_bin) / bin_size)
        self.contact_bin = max(0, min(contact_bin, num_bins - 1))

    @staticmethod
    def _scatter_sum(
        values: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        return values.new_zeros(num_groups).scatter_add(0, group_index, values)

    @staticmethod
    def _segmented_tail_indices(
        probabilities: torch.Tensor,
        group_index: torch.Tensor,
        group_budget: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        """Select the highest-probability entries within each group."""
        if probabilities.numel() == 0:
            return group_index.new_empty(0)

        probability_order = torch.argsort(
            probabilities,
            descending=True,
            stable=True,
        )
        probability_sorted_group = group_index[probability_order]
        group_order = torch.argsort(probability_sorted_group, stable=True)
        sorted_pair_index = probability_order[group_order]
        sorted_group = group_index[sorted_pair_index]
        sorted_position = torch.arange(
            probabilities.numel(),
            device=probabilities.device,
            dtype=torch.long,
        )
        first_position = group_index.new_full(
            (num_groups,),
            probabilities.numel(),
        )
        first_position.scatter_reduce_(
            0,
            sorted_group,
            sorted_position,
            reduce="amin",
            include_self=True,
        )
        rank_within_group = sorted_position - first_position[sorted_group]
        return sorted_pair_index[rank_within_group < group_budget[sorted_group]]

    def forward(
        self,
        logits: torch.Tensor,
        f_input: FoldingInput,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute example-macro hard-pair distogram supervision."""
        if logits.shape[-1] != self.num_bins:
            raise ValueError(
                f"Expected {self.num_bins} distogram bins, got {logits.shape[-1]}."
            )

        logits_float = logits.float()
        batch_size, num_tokens, _, _ = logits_float.shape
        device = logits.device
        log_prob = F.log_softmax(logits_float, dim=-1)
        p_contact = log_prob[..., : self.contact_bin + 1].logsumexp(dim=-1).exp()

        with torch.no_grad():
            coords = f_input.token.repr_coords.float()
            displacement = coords[..., None, :, :] - coords[..., :, None, :]
            distance = displacement.norm(dim=-1)
            target = (distance.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()

            repr_mask = f_input.token.repr_mask
            pair_mask = repr_mask[..., None, :] & repr_mask[..., :, None]
            asym_id = f_input.token.asym_id
            asym_i = asym_id[..., :, None]
            asym_j = asym_id[..., None, :]
            inter_chain = asym_i != asym_j
            upper_triangle = torch.ones(
                num_tokens,
                num_tokens,
                dtype=torch.bool,
                device=device,
            ).triu(diagonal=1)
            valid = pair_mask & inter_chain & upper_triangle
            positive = valid & (distance < self.contact_cutoff)
            negative = valid & ~positive

            dense_asym_id = torch.stack(
                [
                    torch.unique(
                        example_asym_id,
                        sorted=True,
                        return_inverse=True,
                    )[1]
                    for example_asym_id in asym_id
                ],
                dim=0,
            )
            dense_asym_i = dense_asym_id[..., :, None]
            dense_asym_j = dense_asym_id[..., None, :]
            chain_stride = num_tokens
            group_stride = chain_stride * chain_stride
            batch_offset = (
                torch.arange(batch_size, device=device, dtype=torch.long) * group_stride
            )[:, None, None]
            asym_low = torch.minimum(dense_asym_i, dense_asym_j)
            asym_high = torch.maximum(dense_asym_i, dense_asym_j)
            group_id = batch_offset + asym_low * chain_stride + asym_high
            num_groups = batch_size * group_stride

        exact_bin_ce = -log_prob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        positive_group = group_id[positive]
        negative_group = group_id[negative]
        positive_probability = p_contact[positive]
        negative_probability = p_contact[negative]
        positive_ce = exact_bin_ce[positive]
        negative_ce = exact_bin_ce[negative]
        positive_focal_weight = (
            (1.0 - positive_probability).pow(self.focal_gamma).detach()
        )
        negative_focal_weight = negative_probability.pow(self.focal_gamma).detach()

        positive_count = self._scatter_sum(
            torch.ones_like(positive_ce),
            positive_group,
            num_groups,
        )
        positive_numerator = self._scatter_sum(
            positive_focal_weight * positive_ce,
            positive_group,
            num_groups,
        )
        active_interface = positive_count > 0
        positive_loss = positive_numerator / (positive_count + self.eps)

        with torch.no_grad():
            fp_budget = torch.ceil(positive_count * self.fp_budget_ratio).long()
            fp_budget = torch.where(
                active_interface,
                fp_budget,
                torch.zeros_like(fp_budget),
            )
            selected_negative_index = self._segmented_tail_indices(
                negative_probability,
                negative_group,
                fp_budget,
                num_groups,
            )

        selected_negative_group = negative_group[selected_negative_index]
        selected_negative_numerator = self._scatter_sum(
            (negative_focal_weight * negative_ce)[selected_negative_index],
            selected_negative_group,
            num_groups,
        )
        selected_negative_count = self._scatter_sum(
            torch.ones_like(negative_ce)[selected_negative_index],
            selected_negative_group,
            num_groups,
        )
        negative_loss = selected_negative_numerator / (selected_negative_count + self.eps)

        active_group_index = torch.nonzero(
            active_interface,
            as_tuple=False,
        ).flatten()
        active_batch = torch.div(
            active_group_index,
            group_stride,
            rounding_mode="floor",
        )
        example_interface_count = self._scatter_sum(
            torch.ones_like(positive_loss[active_group_index]),
            active_batch,
            batch_size,
        )
        example_positive_loss = self._scatter_sum(
            positive_loss[active_group_index],
            active_batch,
            batch_size,
        ) / example_interface_count.clamp(min=1.0)
        example_negative_loss = self._scatter_sum(
            negative_loss[active_group_index],
            active_batch,
            batch_size,
        ) / example_interface_count.clamp(min=1.0)
        example_loss = (
            self.positive_weight * example_positive_loss
            + self.negative_weight * example_negative_loss
        )
        loss = example_loss.mean() + logits_float.sum() * 0.0

        active_interface_float = active_interface.to(logits_float.dtype)
        active_interface_count = active_interface_float.sum().clamp(min=1.0)
        positive_float = positive.to(logits_float.dtype)
        negative_float = negative.to(logits_float.dtype)
        selected_negative_probability = negative_probability[selected_negative_index]
        selected_negative_focal_weight = negative_focal_weight[selected_negative_index]
        metrics = {
            "interface_contact_loss": loss.detach(),
            "interface_contact_fn_loss": (
                (positive_loss * active_interface_float).sum() / active_interface_count
            ).detach(),
            "interface_contact_fp_loss": (
                (negative_loss * active_interface_float).sum() / active_interface_count
            ).detach(),
            "interface_contact_active_interfaces": (
                active_interface_float.sum().detach()
            ),
            "interface_contact_active_examples": (
                (example_interface_count > 0).float().sum().detach()
            ),
            "interface_contact_positive_pairs": positive_float.sum().detach(),
            "interface_contact_negative_pairs": negative_float.sum().detach(),
            "interface_contact_selected_negative_pairs": (
                selected_negative_count.sum().detach()
            ),
            "interface_contact_fp_tail_budget": (
                (fp_budget * active_interface).sum().detach()
            ),
            "interface_contact_p_true_contact": (
                positive_probability.sum() / positive_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_p_true_negative": (
                negative_probability.sum() / negative_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_p_selected_negative": (
                selected_negative_probability.mean()
                if selected_negative_probability.numel() > 0
                else negative_probability.sum() * 0.0
            ).detach(),
            "interface_contact_fn_focal_weight": (
                positive_focal_weight.sum() / positive_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_fp_focal_weight": (
                negative_focal_weight.sum() / negative_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_selected_fp_focal_weight": (
                selected_negative_focal_weight.mean()
                if selected_negative_focal_weight.numel() > 0
                else negative_focal_weight.sum() * 0.0
            ).detach(),
        }
        return loss, metrics
