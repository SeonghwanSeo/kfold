"""Loss functions for confidence prediction heads, including experimentally resolved
prediction, Predicted Distance Error (PDE), pLDDT, and Predicted Aligned Error (PAE).

The GT structures used for loss computation are aligned to the predicted structures
using `kfold.training.utils.permutation_alignment.align_train`.

"""

import torch

from kfold.data.types import FoldingInput, RefStructure
from kfold.model.primitives.utils import expand_dim, gather_dim
from kfold.training.utils.permutation_alignment.align_train import get_aligned_true_coords


def get_one_hot_from_bins(
    tensor: torch.Tensor, bin_centers: torch.Tensor
) -> torch.Tensor:
    """Get one-hot encoding of a tensor based on the provided bins.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be one-hot encoded of shape (*,).
    bin_centers : torch.Tensor
        A tensor containing the centers of the bins for one-hot encoding
        of shape (num_bins,).

    Returns
    -------
    torch.Tensor
        A one-hot encoded tensor of shape (*, num_bins).
    """
    num_bins = bin_centers.shape[0]
    d = torch.abs(tensor[..., None] - bin_centers)  # [*, num_bins]
    indices = d.argmin(dim=-1)  # [*]
    return torch.nn.functional.one_hot(indices, num_classes=num_bins)


def cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances between two sets of points."""
    d = x[..., :, None, :] - y[..., None, :, :]  # [*, Lx, Ly, 3]
    return d.norm(dim=-1)


def get_aligned_gt_structure(
    x_pred: torch.Tensor,
    f_input: FoldingInput,
    struct_info: list,
    align_only_for_confidence_loss: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the aligned ground truth coordinates for the confidence prediction losses.

    Parameters
    ----------
    x_pred : torch.Tensor
        Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
    f_input : FoldingInput
        The input features containing masks and representative atom indices.
    struct_info : list
        A list of length B containing structure information for each sample, including
        the reference structure and symmetry information.
    align_only_for_confidence_loss : bool, optional
        Whether to only compute aligned GT coordinates for samples that contribute to
        the confidence loss (i.e., RCSB structures with resolution <= 4.0A).

    Returns
    -------
    x_gt_aligned : torch.Tensor
        Tensor of shape (B, N, Natom, 3) containing the aligned ground truth coordinates.
    mask : torch.Tensor
        Tensor of shape (B, N, Natom) containing boolean masks for valid atoms.
    """
    # Ground truth coordinates for confidence loss computation.
    x_gt: torch.Tensor = torch.full_like(x_pred, torch.nan)
    f_input_list = f_input.to_list(deepcopy=False)
    batch_size, num_sample, _, _ = x_pred.shape
    for b_i in range(batch_size):
        f_input_i = f_input_list[b_i]
        struct_info_i = struct_info[b_i]
        ref_struct_i: RefStructure = struct_info_i["structure"]

        # Check if we should compute aligned GT coords for this sample
        if align_only_for_confidence_loss and not struct_info_i["train_confidence_head"]:
            _x_gt_i = f_input_i.atom.label_coords.clone()  # [Natom, 3]
            _mask_i = f_input_i.atom.resolved_mask
            _x_gt_i[~_mask_i] = torch.nan  # Mask unresolved atoms to NaN
            x_gt[b_i] = expand_dim(_x_gt_i, num_sample, dim=0)
            continue

        symmetry_dict_i: dict = struct_info_i["symmetry"]
        num_atoms: int = f_input_i.atom.pad_mask.sum().item()
        for s_j in range(num_sample):
            x_gt_ij = get_aligned_true_coords(
                ref_struct=ref_struct_i,
                f_input=f_input_i,
                pred_coords=x_pred[b_i, s_j, :num_atoms],
                symmetry_dict=symmetry_dict_i,
            )
            x_gt[b_i, s_j, :num_atoms] = x_gt_ij
    mask = x_gt.isfinite().all(-1)  # [B, N, Natom]
    x_gt[~mask] = 0.0  # Mask unresolved atoms to zero
    return x_gt, mask


class ExperimentallyResolvedPredictionLoss(torch.nn.Module):
    """Loss for predicting whether each atom is experimentally resolved or not."""

    def forward(
        self,
        logits: torch.Tensor,
        resolved_mask: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the  distogram loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, N, Natom, 2) containing experimentally resolved prediction
            logits.
        resolved_mask : torch.Tensor
            Tensor of shape (B, N, Natom) containing boolean masks for experimentally
            resolved atoms.
        pad_mask : torch.Tensor
            Tensor of shape (B, Natom) containing boolean masks for padded atoms.

        Returns
        -------
        exp_resolved_loss : torch.Tensor
            The computed experimentally resolved prediction loss of shape (B, N).
        """
        B, N, Natom, _ = logits.shape

        # Compute the loss
        label = resolved_mask.long()  # [B, N, Natom]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(B * N * Natom, 2),
            label.reshape(B * N * Natom),
            reduction="none",
        ).view(B, N, Natom)

        mask = pad_mask.unsqueeze(1).float()  # [B, 1, Natom]
        n_valid = mask.sum(dim=-1).clamp(1)  # [B, 1]
        loss_mean = (loss * mask).sum(dim=-1) / n_valid  # [B, N]
        return loss_mean  # [B, N]


class PDELoss(torch.nn.Module):
    """Loss for predicting pairwise distances (PDE) between representative atoms."""

    def forward(
        self,
        logits: torch.Tensor,
        bin_centers: torch.Tensor,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask_gt: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the Predicted Distance Error (PDE) loss between representative atoms.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, N, L, L, num_bins) containing distogram logits.
        bin_centers : torch.Tensor
            PDE bin centers returned by the confidence head.
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_gt : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates.
        mask : torch.Tensor
            Tensor of shape (B, N, Natom) containing boolean masks for valid atoms
            in the GT structure.
        f_input : FoldingInput
            The input features containing the target distogram and masks.

        Returns
        -------
        pde_loss : torch.Tensor
            The computed pde loss of shape (B,).
        """
        with torch.no_grad():
            e = self.get_distance_error(x_pred, x_gt, mask_gt, f_input)  # [B, N, L, L]

        # Compute loss
        e_bins = get_one_hot_from_bins(e, bin_centers)  # [B, N, L, L, num_bins]
        loss = -torch.sum(e_bins.float() * logits.log_softmax(-1), dim=-1)  # [B, N, L, L]

        # Reduce loss
        repr_idx = f_input.token.repr_index[:, None, :]  # [B, 1, L]
        repr_mask = gather_dim(mask_gt, -1, repr_idx)  # [B, N, L]
        repr_mask &= f_input.token.pad_mask[:, None, :]  # [B, N, L]
        pair_mask = repr_mask[..., None, :] & repr_mask[..., :, None]  # [B, N, L, L]
        n_pairs = pair_mask.sum((-1, -2)).clamp(1)  # [B, N]
        loss_mean = (loss * pair_mask).sum((-1, -2)) / n_pairs  # [B, N]
        return loss_mean

    def get_distance_error(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask_gt: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the pairwise distance error between predicted and GT coordinates
        for representative atoms.

        Parameters
        ----------
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_gt : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates.
        mask_gt : torch.Tensor
            Tensor of shape (B, N, Natom) containing boolean masks for valid atoms in
            the GT structure.
        f_input : FoldingInput
            The input features.

        Returns
        -------
        distance_error : torch.Tensor
            Tensor of shape (B, N, L, L) containing the pairwise distance error
        """
        # [*, Natom, 3] -> [*, L, 3]
        repr_idx = f_input.token.repr_index[:, None, :]  # [B, 1, L]
        _x_pred = gather_dim(x_pred, 2, repr_idx[..., None])  # [B, N, L, 3]
        _x_gt = gather_dim(x_gt, 2, repr_idx[..., None])  # [B, N, L, 3]

        # Compute pairwise distances
        d_pred = cdist(_x_pred, _x_pred)  # [B, N, L, L]
        d_gt = cdist(_x_gt, _x_gt)  # [B, N, L, L]

        # Compute distance error
        e = torch.abs(d_gt - d_pred)  # [B, N, L, L]

        # Mask invalid pairs (where either atom is invalid)
        repr_mask = gather_dim(mask_gt, -1, repr_idx)  # [B, N, L]
        pair_mask = repr_mask[..., None, :] & repr_mask[..., :, None]
        e.masked_fill_(~pair_mask, 0.0)

        return e


class PLDDTLoss(torch.nn.Module):
    """Loss for predicting the Local Distance Difference Test (pLDDT) for each atom."""

    def forward(
        self,
        logits: torch.Tensor,
        bin_centers: torch.Tensor,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask_gt: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the pLDDT loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, N, Natom, num_bins) containing pLDDT logits.
        bin_centers : torch.Tensor
            pLDDT bin centers returned by the confidence head.
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_gt : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates.
        mask : torch.Tensor
            Tensor of shape (B, N, Natom) containing boolean masks for valid atoms
                in the GT structure.
        f_input : FoldingInput
            The input features containing masks, token flags, and representative indices.

        Returns
        -------
        plddt_loss : torch.Tensor
            The computed pLDDT loss of shape (B, N).
        """
        with torch.no_grad():
            lddt = self.get_lddt_score(x_pred, x_gt, mask_gt, f_input)  # [B, N, Natom]

        lddt_bins = get_one_hot_from_bins(
            lddt,
            bin_centers,
        )  # [B, N, Natom, num_bins]
        loss = -(lddt_bins.float() * logits.log_softmax(-1)).sum(-1)  # [B, N, Natom]

        n_valid = mask_gt.sum(dim=-1).clamp(min=1)  # [B, 1]
        loss_mean = (loss * mask_gt).sum(dim=-1) / n_valid  # [B, N]
        return loss_mean

    def get_lddt_score(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask_gt: torch.Tensor,
        f_input: FoldingInput,
    ):
        """Compute the ground truth LDDT score for each atom based on predicted and
        true coordinates.

        Parameters
        ----------
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_true : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates
        mask_gt : torch.Tensor
            Tensor of shape (B, N, Natom) containing boolean masks for valid atoms.
        f_input : FoldingInput
            The input features containing masks, token flags, and representative indices.

        Returns
        -------
        lddt_score : torch.Tensor
            Tensor of shape (B, N, Natom) containing the ground truth LDDT
            score for each atom.
        """
        B, Nrepr, Nall = f_input.batch_size, f_input.num_tokens, f_input.num_atoms  # noqa
        device = x_pred.device

        # === Create loss masks ===
        # Create mask for valid atom pairs.
        repr_idx = f_input.token.repr_index[:, None, :]
        atom_mask = mask_gt  # [B, N, Natom]
        repr_mask = gather_dim(mask_gt, -1, repr_idx)  # [B, N, Nrepr]
        pair_mask = (
            atom_mask[..., :, None] & repr_mask[..., None, :]
        )  # [B, N, Nall, Nrepr]

        # === Extract representative coordinates ===
        x_pred_rep = gather_dim(x_pred, 2, repr_idx[..., None])  # [B, N, Nrepr, 3]
        x_gt_rep = gather_dim(x_gt, 2, repr_idx[..., None])  # [B, N, Nrepr, 3]

        # === Compute pairwise distances (All Atoms -> Rep Atoms) ===
        d_pred = cdist(x_pred, x_pred_rep)  # [B, N, Nall, Nrepr]
        d_gt = cdist(x_gt, x_gt_rep)  # [B, N, Nall, Nrepr]

        # Protein: cutoff 15A, Nucleic Acids: cutoff 30A
        is_prot = f_input.token.is_protein[:, None, None, :]
        is_nuc = (f_input.token.is_rna | f_input.token.is_dna)[:, None, None, :]
        pair_mask &= ((d_gt < 15.0) & is_prot) | ((d_gt < 30.0) & is_nuc)

        # Mask non-standard residues (e.g., modified, ligand)
        is_nonstandard = ~f_input.token.is_standard[:, None, None, :]
        pair_mask &= ~is_nonstandard

        # Mask self-pairs
        atom_index = torch.arange(Nall, device=device)
        pair_mask &= (
            atom_index[None, None, :, None] != f_input.token.repr_index[:, None, None, :]
        )  # [B, 1, Nall, Nrepr]

        # Compute LDDT Score
        e = torch.abs(d_gt - d_pred)  # [B, N, Natom, Nrepr]
        score = torch.zeros_like(e)
        for cutoff in [0.5, 1.0, 2.0, 4.0]:
            score += (e < cutoff).float()
        score *= 0.25
        score.masked_fill_(~pair_mask, 0.0)

        # Aggregate
        lddt_score = score.sum(dim=-1) / pair_mask.sum(dim=-1).clamp(min=1)
        return lddt_score


class PAELoss(torch.nn.Module):
    """Loss on Predicted Aligned Error (PAE)."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps: float = eps

    def forward(
        self,
        logits: torch.Tensor,
        bin_centers: torch.Tensor,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask_gt: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the PAE loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, N, L, L, num_bins) containing PAE logits.
        bin_centers : torch.Tensor
            PAE bin centers returned by the confidence head.
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_gt : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates.
        mask : torch.Tensor
            Tensor of shape (B, N, Natom) containing boolean masks for valid atoms
            in the GT structure.
        f_input : FoldingInput
            The input features containing masks and atom indices.

        Returns
        -------
        pae_loss : torch.Tensor
            The computed PAE loss of shape (B, N).
        """
        with torch.no_grad():
            e = self.get_alignment_error(x_pred, x_gt, mask_gt, f_input)  # [B, N, L, L]

        # Compute Cross Entropy Error
        e_bins = get_one_hot_from_bins(e, bin_centers)
        loss = -(e_bins.float() * logits.log_softmax(-1)).sum(-1)  # [B, N, L, L]

        # === Compute validity masks ===
        frame_idx = f_input.token.frame_index[:, None, :, :]  # [B, 1, L, 3]
        repr_idx = f_input.token.repr_index[:, None, :]  # [B, 1, L]
        frame_mask = (
            gather_dim(mask_gt, -1, frame_idx.flatten(-2)).unflatten(-1, (-1, 3)).all(-1)
        )  # [B, N, L]
        repr_mask = gather_dim(mask_gt, -1, repr_idx)  # [B, N, L]
        mask_i = (
            f_input.token.frame_mask.unsqueeze(1) & repr_mask & frame_mask
        )  # [B, N, L]
        mask_j = repr_mask  # [B, N, L]
        pair_mask = mask_i[..., :, None] & mask_j[..., None, :]  # [B, N, L, L]

        # Reduce
        n_valid = pair_mask.sum(dim=(-1, -2)).clamp(min=1)  # [B, N]
        loss_mean = (loss * pair_mask).sum(dim=(-1, -2)) / n_valid  # [B, N]

        return loss_mean

    def get_alignment_error(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask_gt: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the ground truth alignment error for each pair of representative atoms.

        Parameters
        ----------
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_gt : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates.
        mask_gt : torch.Tensor
            Tensor of shape (B, N, Natom) containing boolean masks for valid atoms in
            the GT structure.
        f_input : FoldingInput
            The input features containing masks and representative atom indices.

        Returns
        -------
        alignment_error : torch.Tensor
            Tensor of shape (B, N, L, L) containing the ground truth alignment error for
            each pair of representative atoms.
        """
        # Extract representative coordinates (the 'j' tokens)
        repr_idx = f_input.token.repr_index[:, None, :]  # [B, 1, Nrepr, 1]
        x_pred_rep = gather_dim(x_pred, -2, repr_idx[..., None])  # [B, N, Nrepr, 3]
        x_gt_rep = gather_dim(x_gt, -2, repr_idx[..., None])  # [B, N, Nrepr, 3]

        # Extract frame coordinates (the 'i' tokens)
        frame_idx = f_input.token.frame_index[:, None, :, :]  # [B, 1, L, 3]
        a_i, b_i, c_i = frame_idx.unbind(-1)  # Each of shape [B, 1, L]
        a_gt = gather_dim(x_gt, -2, a_i[..., None])  # [B, N, L, 3]
        b_gt = gather_dim(x_gt, -2, b_i[..., None])  # [B, N, L, 3]
        c_gt = gather_dim(x_gt, -2, c_i[..., None])  # [B, N, L, 3]
        a_pred = gather_dim(x_pred, -2, a_i[..., None])  # [B, N, L, 3]
        b_pred = gather_dim(x_pred, -2, b_i[..., None])  # [B, N, L, 3]
        c_pred = gather_dim(x_pred, -2, c_i[..., None])  # [B, N, L, 3]

        # Project coords into frames
        xij_gt = self.express_coords_in_frames(x_gt_rep, a_gt, b_gt, c_gt)
        xij_pred = self.express_coords_in_frames(x_pred_rep, a_pred, b_pred, c_pred)

        # Compute Euclidean distance between alignments
        e = torch.sqrt((xij_pred - xij_gt).pow(2).sum(-1) + self.eps)

        # Apply mask to set pad atoms to zero
        repr_mask = gather_dim(mask_gt, -1, repr_idx)  # [B, N, L]
        pair_mask = repr_mask[..., :, None] & repr_mask[..., None, :]  # [B, 1, L, L]
        e.masked_fill_(~pair_mask, 0.0)
        return e

    @staticmethod
    def express_coords_in_frames(
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Project coordinates `x` into the local frames defined by atoms `a, b, c`.
        See Section 4.3.2 Algorithm 29 of the AlphaFold paper.
        """
        # Line 2
        w1 = a - b
        w1 /= w1.norm(dim=-1, keepdim=True) + 1e-8

        # Line 3
        w2 = c - b
        w2 /= w2.norm(dim=-1, keepdim=True) + 1e-8

        # Build orthogonal frame basis (e1, e2, e3)
        # Line 4
        e1 = w1 + w2
        e1 /= e1.norm(dim=-1, keepdim=True) + 1e-8

        # Line 5
        e2 = w2 - w1
        e2 /= e2.norm(dim=-1, keepdim=True) + 1e-8

        # Line 6
        e3 = torch.linalg.cross(e1, e2, dim=-1)

        # Project onto frame basis
        # Line 7
        d = x.unsqueeze(-3) - b.unsqueeze(-2)

        # Line 8
        x_transformed = torch.stack(
            [
                torch.einsum("...id,...ijd->...ij", e1, d),
                torch.einsum("...id,...ijd->...ij", e2, d),
                torch.einsum("...id,...ijd->...ij", e3, d),
            ],
            dim=-1,
        )
        return x_transformed
