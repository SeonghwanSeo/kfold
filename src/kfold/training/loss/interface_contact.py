import math

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput

_CHAIN_TYPE_COUNT = 4
_MODALITIES: tuple[tuple[int, int, str], ...] = (
    (0, 0, "protein_protein"),
    (0, 1, "protein_dna"),
    (0, 2, "protein_rna"),
    (0, 3, "protein_ligand"),
    (1, 1, "dna_dna"),
    (1, 2, "dna_rna"),
    (1, 3, "dna_ligand"),
    (2, 2, "rna_rna"),
    (2, 3, "rna_ligand"),
    (3, 3, "ligand_ligand"),
)


class InterfaceContactBalancedLoss(torch.nn.Module):
    """FN-priority contact loss with count-sensitive far-pair supervision."""

    def __init__(
        self,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        contact_cutoff: float = 8.0,
        far_cutoff: float = 20.0,
        positive_weight: float = 0.75,
        negative_weight: float = 0.25,
        focal_gamma: float = 2.0,
        fp_budget_ratio: float = 4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not min_dist < contact_cutoff < far_cutoff <= max_dist:
            raise ValueError(
                "Expected min_dist < contact_cutoff < far_cutoff <= max_dist."
            )
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than one.")
        if positive_weight <= 0 or negative_weight < 0:
            raise ValueError("Loss weights must be non-negative with positive FN weight.")
        if not math.isclose(positive_weight + negative_weight, 1.0):
            raise ValueError("positive_weight and negative_weight must sum to one.")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative.")
        if fp_budget_ratio <= 0:
            raise ValueError("fp_budget_ratio must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.contact_cutoff = contact_cutoff
        self.far_cutoff = far_cutoff
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.focal_gamma = focal_gamma
        self.fp_budget_ratio = fp_budget_ratio
        self.eps = eps

        bin_size = (max_dist - min_dist) / num_bins
        first_bin = min_dist + bin_size
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
        selected = rank_within_group < group_budget[sorted_group]
        return sorted_pair_index[selected]

    def forward(
        self,
        logits: torch.Tensor,
        f_input: FoldingInput,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute an example-macro, interface-balanced contact loss."""
        logits_float = logits.float()
        batch_size, num_tokens, _, _ = logits_float.shape
        device = logits.device

        log_normalizer = torch.logsumexp(logits_float, dim=-1)
        log_contact = torch.logsumexp(
            logits_float[..., : self.contact_bin + 1],
            dim=-1,
        )
        log_noncontact = torch.logsumexp(
            logits_float[..., self.contact_bin + 1 :],
            dim=-1,
        )
        log_p_contact = log_contact - log_normalizer
        p_contact = log_p_contact.exp()

        with torch.no_grad():
            coords = f_input.token.repr_coords.float()
            displacement = coords[..., None, :, :] - coords[..., :, None, :]
            distance = displacement.norm(dim=-1)

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
            far = valid & (distance >= self.far_cutoff)

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

        positive_group = group_id[positive]
        far_group = group_id[far]
        positive_probability = p_contact[positive]
        positive_bce = -log_p_contact[positive]
        positive_focal_weight = (
            (1.0 - positive_probability).pow(self.focal_gamma).detach()
        )
        far_probability = p_contact[far]
        far_log_odds = (log_contact - log_noncontact)[far]
        far_bce = F.softplus(far_log_odds)
        far_focal_weight = far_probability.pow(self.focal_gamma).detach()

        positive_numerator = self._scatter_sum(
            positive_focal_weight * positive_bce,
            positive_group,
            num_groups,
        )
        positive_denominator = self._scatter_sum(
            positive_focal_weight,
            positive_group,
            num_groups,
        )
        positive_count = self._scatter_sum(
            torch.ones_like(positive_bce),
            positive_group,
            num_groups,
        )
        active_interface = positive_count > 0

        with torch.no_grad():
            fp_budget = torch.ceil(positive_count * self.fp_budget_ratio).long()
            fp_budget = torch.where(
                active_interface,
                fp_budget,
                torch.zeros_like(fp_budget),
            )

        selected_far_index = self._segmented_tail_indices(
            far_probability.detach(),
            far_group,
            fp_budget,
            num_groups,
        )
        selected_far_group = far_group[selected_far_index]
        selected_far_numerator = self._scatter_sum(
            (far_focal_weight * far_bce)[selected_far_index],
            selected_far_group,
            num_groups,
        )
        selected_far_count = self._scatter_sum(
            torch.ones_like(far_bce)[selected_far_index],
            selected_far_group,
            num_groups,
        )

        positive_loss = positive_numerator / (positive_denominator + self.eps)
        far_loss = selected_far_numerator / (positive_count + self.eps)

        active_group_index = torch.nonzero(active_interface, as_tuple=False).flatten()
        active_batch = torch.div(
            active_group_index,
            group_stride,
            rounding_mode="floor",
        )
        example_positive_loss_sum = self._scatter_sum(
            positive_loss[active_group_index],
            active_batch,
            batch_size,
        )
        example_interface_count = self._scatter_sum(
            torch.ones_like(positive_loss[active_group_index]),
            active_batch,
            batch_size,
        )
        example_positive_loss = example_positive_loss_sum / example_interface_count.clamp(
            min=1.0
        )

        example_far_loss_sum = self._scatter_sum(
            far_loss[active_group_index],
            active_batch,
            batch_size,
        )
        example_far_loss = example_far_loss_sum / example_interface_count.clamp(min=1.0)
        example_loss = (
            self.positive_weight * example_positive_loss
            + self.negative_weight * example_far_loss
        )
        loss = example_loss.mean() + log_p_contact.sum() * 0.0

        active_example = example_interface_count > 0
        active_interface_float = active_interface.to(logits_float.dtype)
        positive_float = positive.to(logits_float.dtype)
        far_float = far.to(logits_float.dtype)
        selected_far_probability = far_probability[selected_far_index]
        selected_far_focal_weight = far_focal_weight[selected_far_index]
        metrics = {
            "interface_contact_loss": loss.detach(),
            "interface_contact_fn_loss": (
                (positive_loss * active_interface_float).sum()
                / active_interface_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_fp_loss": (
                (far_loss * active_interface_float).sum()
                / active_interface_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_active_interfaces": (
                active_interface_float.sum().detach()
            ),
            "interface_contact_active_examples": active_example.float().sum().detach(),
            "interface_contact_positive_pairs": positive_float.sum().detach(),
            "interface_contact_far_pairs": far_float.sum().detach(),
            "interface_contact_selected_far_pairs": (selected_far_count.sum().detach()),
            "interface_contact_fp_tail_budget": (
                (fp_budget * active_interface).sum().detach()
            ),
            "interface_contact_p_true_contact": (
                (p_contact * positive_float).sum() / positive_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_p_true_far": (
                far_probability.sum() / far_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_p_selected_far": (
                selected_far_probability.sum() / selected_far_probability.numel()
                if selected_far_probability.numel() > 0
                else far_probability.sum() * 0.0
            ).detach(),
            "interface_contact_fn_focal_weight": (
                positive_focal_weight.sum() / positive_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_fp_focal_weight": (
                far_focal_weight.sum() / far_float.sum().clamp(min=1.0)
            ).detach(),
            "interface_contact_selected_fp_focal_weight": (
                selected_far_focal_weight.sum() / selected_far_focal_weight.numel()
                if selected_far_focal_weight.numel() > 0
                else far_focal_weight.sum() * 0.0
            ).detach(),
        }

        with torch.no_grad():
            chain_type = f_input.token.chain_type
            type_i = chain_type[..., :, None]
            type_j = chain_type[..., None, :]
            type_low = torch.minimum(type_i, type_j)
            type_high = torch.maximum(type_i, type_j)
            modality = type_low * _CHAIN_TYPE_COUNT + type_high
            group_modality = torch.full(
                (num_groups,),
                -1,
                dtype=torch.long,
                device=device,
            )
            group_modality.scatter_reduce_(
                0,
                group_id[valid],
                modality[valid],
                reduce="amax",
                include_self=True,
            )
            for low, high, name in _MODALITIES:
                modality_id = low * _CHAIN_TYPE_COUNT + high
                in_modality = valid & (modality == modality_id)
                metrics[f"interface_contact_active_interfaces_{name}"] = (
                    (active_interface & (group_modality == modality_id))
                    .float()
                    .sum()
                    .detach()
                )
                metrics[f"interface_contact_positive_pairs_{name}"] = (
                    (positive & in_modality).float().sum().detach()
                )
                metrics[f"interface_contact_far_pairs_{name}"] = (
                    (far & in_modality).float().sum().detach()
                )
                metrics[f"interface_contact_selected_far_pairs_{name}"] = (
                    (group_modality[selected_far_group] == modality_id)
                    .float()
                    .sum()
                    .detach()
                )

        return loss, metrics
