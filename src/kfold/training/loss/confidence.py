import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.kernels.cdist import cdist as kernel_cdist
from kfold.utils.torch import get_one_hot_from_bins


class ExperimentallyResolvedPredictionLoss(torch.nn.Module):
    """Loss for predicting whether each atom is experimentally resolved or not."""

    def forward(
        self,
        logits: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the  distogram loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, N, Natom, 2) containing experimentally resolved prediction
            logits.
        f_input : FoldingInput
            The input features.

        Returns
        -------
        exp_resolved_loss : torch.Tensor
            The computed experimentally resolved prediction loss of shape (B, N).
        """
        B, N, Natom, _ = logits.shape
        mask = f_input.atom.pad_mask  # [B, Natom]
        label = f_input.atom.resolved_mask  # [B, Natom]

        # Compute the loss
        label = label.unsqueeze(1).expand(-1, N, -1).long()  # [B, N, Natom]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(B * N * Natom, 2),
            label.reshape(B * N * Natom),
            reduction="none",
        ).view(B, N, Natom)

        mask = mask.unsqueeze(1).float()  # [B, 1, Natom]
        n_valid = mask.sum(dim=-1).clamp(1)  # [B, 1]
        loss_mean = (loss * mask).sum(dim=-1) / n_valid  # [B, N]
        return loss_mean  # [B, N]


class PDELoss(torch.nn.Module):
    """Loss for predicting pairwise distances (PDE) between representative atoms."""

    def __init__(
        self,
        min_dist: float = 0.0,
        max_dist: float = 32.0,
        num_bins: int = 64,
        use_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.num_bins: int = num_bins
        self.use_kernel: bool = use_kernel

        bin_size: float = (max_dist - min_dist) / num_bins
        bins = torch.linspace(
            min_dist + bin_size / 2, max_dist - bin_size / 2, num_bins
        )  # [num_bins]
        self.register_buffer("bins", bins, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the Predicted Distance Error (PDE) loss between representative atoms.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, N, L, L, num_bins) containing distogram logits.
        x_pred : torch.Tensor
            Tensor of shape (B, N, L, 3) containing predicted coordinates.
        x_true : torch.Tensor
            Tensor of shape (B, N, L, 3) containing ground truth coordinates.
        f_input : FoldingInput
            The input features containing the target distogram and masks.

        Returns
        -------
        pde_loss : torch.Tensor
            The computed pde loss of shape (B,).
        """
        with torch.no_grad():
            e = self.get_distance_error(x_pred, x_true, f_input)  # [B, N, L, L]

        # Compute loss
        e_bins = get_one_hot_from_bins(e, self.bins)  # [B, N, L, L, num_bins]
        loss = -torch.sum(e_bins.float() * logits.log_softmax(-1), dim=-1)  # [B, N, L, L]

        # Reduce loss
        mask = f_input.token.repr_mask  # [B, N, L]
        pair_mask = mask[..., None, :] & mask[..., :, None]  # [B, N, L, L]
        n_pairs = pair_mask.sum((-1, -2)).clamp(1)  # [B, N]
        loss_mean = (loss * pair_mask).sum((-1, -2)) / n_pairs  # [B, N]
        return loss_mean

    def get_distance_error(
        self, x_pred: torch.Tensor, x_true: torch.Tensor, f_input: FoldingInput
    ) -> torch.Tensor:
        """Compute the pairwise distance error between predicted and true coordinates
        for representative atoms.

        Parameters
        ----------
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_true : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates.
        f_input : FoldingInput
            The input features.

        Returns
        -------
        distance_error : torch.Tensor
            Tensor of shape (B, N, L, L) containing the pairwise distance error
        """
        # [*, Natom, 3] -> [*, L, 3]
        B, N, _, _ = x_pred.shape
        device = x_pred.device
        b_idcs = torch.arange(B, device=device)[:, None, None]  # [B, 1, 1]
        n_idcs = torch.arange(N, device=device)[None, :, None]  # [1, N, 1]
        repr_idc = f_input.token.repr_index[:, None, :]  # [B, 1, L]
        _x_pred = x_pred[b_idcs, n_idcs, repr_idc]  # [B, N, L, 3]
        _x_true = x_true[b_idcs, n_idcs, repr_idc]  # [B, N, L, 3]

        # Compute pairwise distances
        d_pred = kernel_cdist(_x_pred, _x_pred)  # [B, N, L, L]
        d_true = kernel_cdist(_x_true, _x_true)  # [B, N, L, L]

        # Compute distance error
        e = torch.abs(d_true - d_pred)  # [B, N, L, L]
        return e


class PLDDTLoss(torch.nn.Module):
    """Loss for predicting the Local Distance Difference Test (pLDDT) for each atom."""

    def __init__(self, num_bins: int = 50) -> None:
        super().__init__()
        self.num_bins: int = num_bins

        # Standard AlphaFold/OpenFold-3 bins
        bin_size: float = 1 / num_bins
        bins = torch.arange(bin_size / 2, 1.0, bin_size)  # [num_bins]
        self.register_buffer("bins", bins, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the pLDDT loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, N, Natom, num_bins) containing pLDDT logits.
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_true : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates.
        f_input : FoldingInput
            The input features containing masks, token flags, and representative indices.

        Returns
        -------
        plddt_loss : torch.Tensor
            The computed pLDDT loss of shape (B, N).
        """
        with torch.no_grad():
            lddt = self.get_lddt_score(x_pred, x_true, f_input)  # [B, N, Natom]

        lddt_bins = get_one_hot_from_bins(lddt, self.bins)  # [B, N, Natom, num_bins]
        loss = -(lddt_bins.float() * logits.log_softmax(-1)).sum(-1)  # [B, N, Natom]

        # We only apply loss to atoms that are experimentally resolved
        atom_mask = f_input.atom.resolved_mask.unsqueeze(1).float()  # [B, 1, Natom]
        n_valid = atom_mask.sum(dim=-1).clamp(min=1)  # [B, 1]

        loss_mean = (loss * atom_mask).sum(dim=-1) / n_valid  # [B, N]

        return loss_mean

    def get_lddt_score(
        self, x_pred: torch.Tensor, x_true: torch.Tensor, f_input: FoldingInput
    ):
        """Compute the ground truth LDDT score for each atom based on predicted and
        true coordinates.

        Parameters
        ----------
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_true : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates
        f_input : FoldingInput
            The input features containing masks, token flags, and representative indices.

        Returns
        -------
        lddt_score : torch.Tensor
            Tensor of shape (B, N, Natom) containing the ground truth LDDT
            score for each atom.
        """
        B, L, Nall = f_input.batch_size, f_input.num_tokens, f_input.num_atoms  # noqa
        N = x_pred.shape[1]
        device = x_pred.device

        # === Extract representative coordinates ===
        b_idcs = torch.arange(B, device=device)[:, None, None]  # [B, 1, 1]
        n_idcs = torch.arange(N, device=device)[None, :, None]  # [1, N, 1]
        repr_idc = f_input.token.repr_index[:, None, :]  # [B, 1, L]

        x_pred_rep = x_pred[b_idcs, n_idcs, repr_idc]  # [B, N, L, 3]
        x_true_rep = x_true[b_idcs, n_idcs, repr_idc]  # [B, N, L, 3]

        # === Compute pairwise distances (All Atoms -> Rep Atoms) ===
        d_pred = kernel_cdist(x_pred, x_pred_rep)  # [B, N, Nall, L]
        d_true = kernel_cdist(x_true, x_true_rep)  # [B, N, Nall, L]

        # === Create loss masks ===
        # Protein: cutoff 15A, Nucleic Acids: cutoff 30A
        is_prot = f_input.token.is_protein[:, None, None, :]
        is_nuc = (f_input.token.is_rna | f_input.token.is_dna)[:, None, None, :]
        mask = ((d_true < 15.0) & is_prot) | ((d_true < 30.0) & is_nuc)  # [B, N, Nall, L]

        # Mask unresolved atoms.
        mask &= f_input.atom.resolved_mask[:, None, :, None]
        mask &= f_input.token.repr_mask[:, None, None, :]

        # Mask non-standard residues (e.g., modified, ligand)
        mask &= f_input.token.is_standard[:, None, None, :]

        # Mask self-pairs
        pair_atom_mask = (
            torch.arange(Nall, device=device)[None, None, :, None]
            != f_input.token.repr_index[:, None, None, :]
        )  # [B, 1, Nall, L]
        mask &= pair_atom_mask

        # Compute LDDT Score
        e = torch.abs(d_true - d_pred)  # [B, N, Natom, L]
        score = torch.zeros_like(e)
        for cutoff in [0.5, 1.0, 2.0, 4.0]:
            score += (e < cutoff).float()
        score *= 0.25
        score.masked_fill_(~mask, 0.0)

        # Aggregate
        lddt_score = score.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        return lddt_score


class PAELoss(torch.nn.Module):
    """Loss on Predicted Aligned Error (PAE)."""

    def __init__(
        self,
        min_dist: float = 0.0,
        max_dist: float = 32.0,
        num_bins: int = 64,
        eps: float = 1e-8,
        return_zero: bool = False,
    ) -> None:
        super().__init__()
        self.num_bins: int = num_bins
        self.eps: float = eps
        self.return_zero: bool = return_zero
        bin_size: float = (max_dist - min_dist) / num_bins
        bins = torch.linspace(
            min_dist + bin_size / 2, max_dist - bin_size / 2, num_bins
        )  # [num_bins]
        self.register_buffer("bins", bins, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the PAE loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, N, L, L, num_bins) containing PAE logits.
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_true : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates.
        f_input : FoldingInput
            The input features containing masks and atom indices.

        Returns
        -------
        pae_loss : torch.Tensor
            The computed PAE loss of shape (B, N).
        """
        if self.return_zero:
            # Keep gradients flowing but return zero loss.
            return (logits * 0.0).sum(dim=-1).mean(dim=(-1, -2))

        with torch.no_grad():
            e = self.get_alignment_error(x_pred, x_true, f_input)  # [B, N, L, L]

        # Compute Cross Entropy Error
        e_bins = get_one_hot_from_bins(e, self.bins)
        loss = -(e_bins.float() * logits.log_softmax(-1)).sum(-1)  # [B, N, L, L]

        # === Compute validity masks ===
        mask_i = f_input.token.frame_mask & f_input.token.repr_mask  # [B, L]
        mask_j = f_input.token.repr_mask  # [B, L]
        pair_mask = mask_i[..., :, None] & mask_j[..., None, :]  # [B, L, L]
        pair_mask = pair_mask.unsqueeze(1)  # [B, 1, L, L] -> broadcasts over N

        # Reduce
        n_valid = pair_mask.sum(dim=(-1, -2)).clamp(min=1)  # [B, 1]
        loss_mean = (loss * pair_mask).sum(dim=(-1, -2)) / n_valid  # [B, N]

        return loss_mean

    def get_alignment_error(
        self, x_pred: torch.Tensor, x_true: torch.Tensor, f_input: FoldingInput
    ) -> torch.Tensor:
        """Compute the ground truth alignment error for each pair of representative atoms.

        Parameters
        ----------
        x_pred : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing predicted coordinates.
        x_true : torch.Tensor
            Tensor of shape (B, N, Natom, 3) containing ground truth coordinates.
        f_input : FoldingInput
            The input features containing masks and representative atom indices.

        Returns
        -------
        alignment_error : torch.Tensor
            Tensor of shape (B, N, L, L) containing the ground truth alignment error for
            each pair of representative atoms.
        """
        # [*, Natom, 3] -> [*, L, 3]
        B, N, _ = x_pred.shape
        device = x_pred.device

        # Extract representative coordinates (the 'j' tokens)
        b_idcs = torch.arange(B, device=device)[:, None, None]  # [B, 1, 1]
        n_idcs = torch.arange(N, device=device)[None, :, None]  # [1, N, 1]
        repr_idc = f_input.token.repr_index[:, None, :]  # [B, 1, L]

        x_pred_rep = x_pred[b_idcs, n_idcs, repr_idc]  # [B, N, L, 3]
        x_true_rep = x_true[b_idcs, n_idcs, repr_idc]  # [B, N, L, 3]

        # Extract frame coordinates (the 'i' tokens)
        frame_index = f_input.token.frame_index[:, None, :, :]  # [B, 1, L, 3]

        a_true = x_true[b_idcs, n_idcs, frame_index[..., 0]]
        b_true = x_true[b_idcs, n_idcs, frame_index[..., 1]]
        c_true = x_true[b_idcs, n_idcs, frame_index[..., 2]]

        a_pred = x_pred[b_idcs, n_idcs, frame_index[..., 0]]
        b_pred = x_pred[b_idcs, n_idcs, frame_index[..., 1]]
        c_pred = x_pred[b_idcs, n_idcs, frame_index[..., 2]]

        # Project coords into frames
        xij_true = self.express_coords_in_frames(x_true_rep, a_true, b_true, c_true)
        xij_pred = self.express_coords_in_frames(x_pred_rep, a_pred, b_pred, c_pred)

        # Compute Euclidean distance between alignments
        e = torch.sqrt((xij_pred - xij_true).pow(2).sum(-1) + self.eps)

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
