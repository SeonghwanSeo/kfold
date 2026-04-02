from functools import partial

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.checkpointing import checkpoint_section
from kfold.utils.geometry.rigid_align import weighted_rigid_align


def safe_cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pairwise distances between two sets of points."""
    d = x[..., :, None, :] - y[..., None, :, :]  # [*, Lx, Ly, 3]
    return torch.sqrt(d.pow(2).sum(-1) + eps)


def compute_modality_weights(
    is_protein: torch.Tensor,
    is_dna: torch.Tensor,
    is_rna: torch.Tensor,
    is_ligand: torch.Tensor,
    upweight_protein: float = 0.0,
    upweight_dna: float = 5.0,
    upweight_rna: float = 5.0,
    upweight_ligand: float = 10.0,
) -> torch.Tensor:
    """Compute weights for loss calculation.
    See Section 3.7.1 Equation 4 of the AlphaFold 3 paper.
    """
    return (
        1.0
        + is_protein.float() * upweight_protein
        + is_dna.float() * upweight_dna
        + is_rna.float() * upweight_rna
        + is_ligand.float() * upweight_ligand
    )  # [B, Ltoken]


def get_atom_weights(
    f_input: FoldingInput,
    upweight_protein: float = 0.0,
    upweight_dna: float = 5.0,
    upweight_rna: float = 5.0,
    upweight_ligand: float = 10.0,
) -> torch.Tensor:
    """Compute atom weights for loss calculation.
    See Section 3.7.1 Equation 4 of the AlphaFold 3 paper.
    """
    token_weights = compute_modality_weights(
        f_input.token.is_protein,
        f_input.token.is_dna,
        f_input.token.is_rna,
        f_input.token.is_ligand,
        upweight_protein,
        upweight_dna,
        upweight_rna,
        upweight_ligand,
    )
    batch_indices = torch.arange(f_input.batch_size, device=f_input.device)[:, None]
    atom_weights = token_weights[batch_indices, f_input.atom.token_index]  # [B, Latom]

    return atom_weights


class WeightedMSELoss(torch.nn.Module):
    """Weighted MSE loss of denoised atom positions
    See Section 3.7.1 Equation 2-3 of the AlphaFold 3 paper."""

    def __init__(
        self,
        upweight_protein: float = 0.0,
        upweight_dna: float = 5.0,
        upweight_rna: float = 5.0,
        upweight_ligand: float = 10.0,
        align_true_to_pred: bool = True,
    ):
        """Initialize WeightedMSELoss.
        Parameters
        ----------
        weight_protein: float
            The weight for protein atoms
        weight_dna: float
            The weight for DNA atoms
        weight_rna: float
            The weight for RNA atoms
        weight_ligand: float
            The weight for ligand atoms
        align_true_to_pred: bool
            Whether to align ground truth coordinates to predictions before MSE.
        """
        super().__init__()
        self.upweight_protein: float = upweight_protein
        self.upweight_dna: float = upweight_dna
        self.upweight_rna: float = upweight_rna
        self.upweight_ligand: float = upweight_ligand
        self.align_true_to_pred: bool = align_true_to_pred

    @staticmethod
    def _broadcast_component_weight(
        weight: float | torch.Tensor,
        ref: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Broadcast scalar, per-batch, or full-grid weights to ``ref`` (e.g. [B, N])."""
        if not isinstance(weight, torch.Tensor):
            t = ref.new_tensor(weight)
        else:
            t = weight.to(device=ref.device, dtype=ref.dtype)
        if t.ndim == 0 or t.numel() == 1:
            return t.reshape(())
        if t.shape == ref.shape:
            return t
        if ref.ndim == 2 and t.ndim == 1 and t.shape[0] == ref.shape[0]:
            return t[:, None]
        raise ValueError(
            f"{name} must be scalar, shape {tuple(ref.shape)}, or [B]={ref.shape[0]}, "
            f"got shape {tuple(t.shape)}"
        )

    @staticmethod
    def _get_atom_chain_ids(f_input: FoldingInput, num_atoms: int) -> torch.Tensor:
        """Map each cropped atom to its chain `asym_id`.

        Parameters
        ----------
        f_input : FoldingInput
            Batched model input containing token- and atom-level metadata.
        num_atoms : int
            Number of atom slots kept in the current coordinate tensors after optional
            cropping/pad trimming.

        Returns
        -------
        torch.Tensor
            Atom-wise chain ids of shape [B, L_atom], where each atom is assigned the
            `asym_id` of its parent token.
        """
        assert f_input.token.asym_id.ndim == 2
        assert f_input.atom.token_index.ndim == 2
        atom_token_index = f_input.atom.token_index[:, :num_atoms]
        return f_input.token.asym_id.gather(-1, atom_token_index.clamp(min=0)).clamp(
            min=0
        )

    @staticmethod
    def _assert_unique_chain_type_per_asym_id(f_input: FoldingInput) -> None:
        """Each ``asym_id`` among valid tokens must have a single ``chain_type``."""
        asym = f_input.token.asym_id
        ctype = f_input.token.chain_type
        pad = f_input.token.pad_mask
        assert asym.ndim == 2 and ctype.ndim == 2 and pad.ndim == 2
        batch_size = asym.shape[0]
        for b in range(batch_size):
            valid = pad[b]
            if not valid.any():
                continue
            asym_b = asym[b, valid]
            ctype_b = ctype[b, valid]
            for aid in asym_b.unique():
                types = ctype_b[asym_b == aid].unique()
                assert types.numel() == 1, (
                    f"Inconsistent chain_type for asym_id={aid.item()} in batch row {b}"
                )

    @staticmethod
    def _compute_chain_com(
        coords: torch.Tensor,
        atom_chain_id: torch.Tensor,
        atom_mask_f: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chain centroid per chain (mean over resolved atoms in that chain).

        Parameters
        ----------
        coords : torch.Tensor
            Atom coordinates of shape [B, N, L, 3].
        atom_chain_id : torch.Tensor
            Chain id per atom of shape [B, L].
        atom_mask_f : torch.Tensor
            Resolved mask as float {0, 1}, shape [B, L].

        Returns
        -------
        torch.Tensor
            ``chain_com`` of shape [B, N, C, 3] with C = ``max(atom_chain_id) + 1``.
            Empty chain slots divide by clamped count and are unused in the loss.
        """
        # Shapes for scatter/gather; empty atom dim yields a dummy COM tensor.
        batch_size, num_samples, num_atoms = coords.shape[:3]
        if num_atoms == 0:
            return coords.new_zeros(batch_size, num_samples, 1, 3)

        # One slot per asym_id: resolved atom count per chain (per batch row).
        max_chain_id = int(atom_chain_id.max().item()) + 1
        chain_counts = coords.new_zeros(batch_size, max_chain_id)
        chain_counts.scatter_add_(1, atom_chain_id, atom_mask_f)

        # Sum masked coordinates into each chain bucket for every diffusion sample.
        gather_index = atom_chain_id[:, None, :, None].expand(
            batch_size, num_samples, num_atoms, 3
        )
        chain_coord_sums = coords.new_zeros(batch_size, num_samples, max_chain_id, 3)
        masked_coords = coords * atom_mask_f[:, None, :, None]
        chain_coord_sums.scatter_add_(2, gather_index, masked_coords)

        # Unweighted chain COM: sum/count (clamp avoids div-by-zero on empty slots).
        return chain_coord_sums / chain_counts[:, None, :, None].clamp(min=1.0)

    def _compute_chain_component_losses(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        atom_weights: torch.Tensor,
        atom_mask: torch.Tensor,
        f_input: FoldingInput,
        com_weight: float | torch.Tensor = 1.0,
        internal_weight: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Chain-wise COM and internal-coordinate MSE.

        Atoms are grouped by token ``asym_id`` (cropped batch). Chain centroids are
        unweighted means over resolved atoms; internal errors use modality weights
        ``w_i`` like the original atom MSE.

        Same scalar as the one-term ``WeightedMSELoss`` for the same ``forward`` call
        when ``com_weight`` and ``internal_weight`` are both 1.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted coordinates of shape [B, N, L, 3].
        x_true : torch.Tensor
            Ground-truth coordinates of shape [B, N, L, 3] (typically rigid-aligned to
            ``x_pred`` in ``forward`` when ``align_true_to_pred`` is True).
        atom_weights : torch.Tensor
            Modality weights per atom of shape [B, L], before masking.
        atom_mask : torch.Tensor
            ``resolved_mask`` of shape [B, L]; padding / unresolved atoms are excluded.
        f_input : FoldingInput
            Batched input; ``asym_id``, ``token_index``, ``chain_type``, ``pad_mask``.
        com_weight : float or torch.Tensor
            Coefficient on the COM term; scalar, shape ``[B]``, or ``[B, N]`` matching
            ``com_mse_loss``.
        internal_weight : float or torch.Tensor
            Coefficient on the internal term; same broadcasting rules as ``com_weight``.

        Returns
        -------
        dict[str, torch.Tensor]
            Each tensor has shape [B, N].

            - ``com_mse_loss``: normalized COM term.
            - ``internal_mse_loss``: normalized internal term.
            - ``mse_loss``: ``com_mse_loss + internal_mse_loss``.
            - ``weighted_mse_loss``: ``com_weight * com_mse_loss +
              internal_weight * internal_mse_loss`` (used for backprop when returned
              from ``forward`` without ``return_components``).
        """
        # One modality per asym_id; enables unweighted chain COM without changing the
        # legacy-equivalent total when weights are constant within each chain.
        self._assert_unique_chain_type_per_asym_id(f_input)

        # Per-atom chain id (cropped batch), modality weights on resolved atoms only,
        # and valid-atom count for the same per-atom normalization.
        num_atoms = x_pred.shape[-2]
        atom_chain_id = self._get_atom_chain_ids(f_input, num_atoms=num_atoms)
        atom_mask_f = atom_mask.to(dtype=x_pred.dtype)
        atom_weights = atom_weights * atom_mask_f
        mask_sum = atom_mask_f.sum(-1, keepdim=True).clamp(min=1.0)

        batch_size = x_pred.shape[0]
        max_chain_id = int(atom_chain_id.max().item()) + 1
        chain_weight_sums = x_pred.new_zeros(batch_size, max_chain_id)
        chain_weight_sums.scatter_add_(1, atom_chain_id, atom_weights)

        # Chain centroid per chain for pred and GT (resolved atoms only).
        pred_chain_com = self._compute_chain_com(x_pred, atom_chain_id, atom_mask_f)
        true_chain_com = self._compute_chain_com(x_true, atom_chain_id, atom_mask_f)

        # Broadcast each chain's centroid to its atoms for chain-relative coordinates.
        gather_index = atom_chain_id[:, None, :, None].expand_as(x_pred)
        pred_chain_com_per_atom = torch.gather(pred_chain_com, 2, gather_index)
        true_chain_com_per_atom = torch.gather(true_chain_com, 2, gather_index)

        # Internal motion: displacement from chain centroid (masked).
        rel_pred = (x_pred - pred_chain_com_per_atom) * atom_mask_f[:, None, :, None]
        rel_true = (x_true - true_chain_com_per_atom) * atom_mask_f[:, None, :, None]

        # Chain centroid term: sum_c (sum_{i in c} w_i) ||R_c^pred - R_c^true||^2.
        com_sq = ((pred_chain_com - true_chain_com) ** 2).sum(-1)
        com_loss = (chain_weight_sums[:, None, :] * com_sq).sum(-1)

        # Internal term: sum_i w_i ||r_i^pred - r_i^true||^2.
        internal_sq = ((rel_pred - rel_true) ** 2).sum(-1)
        internal_loss = (atom_weights[:, None, :] * internal_sq).sum(-1)

        # Match legacy WeightedMSE scale; com + internal equals legacy when weights are 1.
        com_loss = (1 / 3) * com_loss / mask_sum
        internal_loss = (1 / 3) * internal_loss / mask_sum
        total_loss = com_loss + internal_loss
        cw = self._broadcast_component_weight(com_weight, com_loss, "com_weight")
        iw = self._broadcast_component_weight(
            internal_weight, internal_loss, "internal_weight"
        )
        weighted_loss = cw * com_loss + iw * internal_loss
        return {
            "weighted_mse_loss": weighted_loss,
            "mse_loss": total_loss,
            "com_mse_loss": com_loss,
            "internal_mse_loss": internal_loss,
        }

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
        remove_pad: bool = False,
        return_components: bool = False,
        com_weight: float | torch.Tensor = 1.0,
        internal_weight: float | torch.Tensor = 1.0,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute the weighted MSE loss.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted coordinates from the model. Shape (B, N, L, 3).
        x_true : torch.Tensor
            Ground truth coordinates. Shape (B, N, L, 3).
        f_input : FoldingInput
            The FoldingInput object containing model inputs.
        remove_pad : bool
            Whether to minimize the padding for memory efficiency.
        com_weight : float or torch.Tensor
            Passed to ``_compute_chain_component_losses`` (e.g. schedule of ``t``).
        internal_weight : float or torch.Tensor
            Same as ``com_weight`` for the internal-coordinate term.

        Returns
        -------
        torch.Tensor
            Computed MSE loss. Shape (B, N).
        """
        assert x_pred.ndim == 4  # [B, N, L, 3]
        assert x_pred.shape == x_true.shape

        w = self.get_atom_weights(f_input)  # [B, L]
        mask = f_input.atom.resolved_mask  # [B, L]

        if remove_pad:
            # Minimize the number of padding
            pad_mask = f_input.atom.pad_mask  # [B, L]
            max_atoms = int(pad_mask.sum(dim=-1).max().clamp(min=1))

            w = w[:, :max_atoms]  # [B, L]
            mask = mask[:, :max_atoms]  # [B, L]
            x_true = x_true[:, :, :max_atoms, :]  # [B, N, L, 3]
            x_pred = x_pred[:, :, :max_atoms, :]  # [B, N, L, 3]

        w = w * mask  # [B, L]

        atom_weights = w
        align_weights = w.unsqueeze(-2)  # [B, 1, L]
        align_mask = mask.unsqueeze(-2)  # [B, 1, L]

        # See Section 3.7.1 Equation 2
        if self.align_true_to_pred:
            with torch.no_grad():
                x_true = weighted_rigid_align(
                    coords=x_true.float(),  # [B, N, L, 3]
                    target=x_pred.float(),  # [B, N, L, 3]
                    weights=align_weights,  # [B, 1, L], broadcasted over N
                    mask=align_mask,  # [B, 1, L]
                )  # [B, N, L, 3]

        loss_terms = self._compute_chain_component_losses(
            x_pred=x_pred,
            x_true=x_true,
            atom_weights=atom_weights,
            atom_mask=mask,
            f_input=f_input,
            com_weight=com_weight,
            internal_weight=internal_weight,
        )

        if return_components:
            return loss_terms
        return loss_terms["weighted_mse_loss"]

    @torch.no_grad()
    def get_atom_weights(self, f_input: FoldingInput) -> torch.Tensor:
        """Compute atom weights for loss calculation.
        See Section 3.7.1 Equation 4 of the AlphaFold 3 paper.
        """
        return get_atom_weights(
            f_input,
            upweight_protein=self.upweight_protein,
            upweight_dna=self.upweight_dna,
            upweight_rna=self.upweight_rna,
            upweight_ligand=self.upweight_ligand,
        )


class BondLoss(torch.nn.Module):
    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute the bond loss.
        See Section 3.7.1 Equation 5

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted pair-wise distance from the model. Shape (B, N, L, 3).
        x_true : torch.Tensor
            Ground truth pair-wise distance. Shape (B, N, L, 3).
        f_input : FoldingInput
            The FoldingInput object containing model inputs.

        Returns
        -------
        bond_loss : torch.Tensor
            Computed Bond loss. Shape (B, N)


        Notes
        -----
        (Seonghwan Seo) Instead of using adjacency matrix, we directly indexing
        since the number of bonds is much smaller than the number of atom pairs.
        e.g., if we consider 512 tokens, the number of atom pairs is (512*24)^2,
        where the number of bonds is (512*10).
        """
        # Return 0 if no polymer-ligand bonds exist
        bond_index = f_input.bond.atom_index  # [B, Nbond, 2]
        is_polymer_ligand_bond = f_input.bond.is_polymer_ligand  # [B, Nbond]

        if not is_polymer_ligand_bond.any():
            return torch.zeros(x_pred.shape[:2], device=x_pred.device)  # [B, N]

        batch_size = x_pred.shape[0]
        src, dst = bond_index[:, :, 0], bond_index[:, :, 1]  # [B, Nbond]
        batch_indices = (
            torch.arange(batch_size, device=bond_index.device)
            .unsqueeze(-1)
            .expand_as(src)
        )  # [B, Nbond]

        # Get bond distances
        src_pred = x_pred[batch_indices, :, src, :].transpose(1, 2)  # [B, N, Nbond, 3]
        dst_pred = x_pred[batch_indices, :, dst, :].transpose(1, 2)  # [B, N, Nbond, 3]
        d_pred = torch.sqrt((src_pred - dst_pred).pow(2).sum(-1) + eps)  # [B, N, Nbond]

        src_true = x_true[batch_indices, :, src, :].transpose(1, 2)  # [B, N, Nbond, 3]
        dst_true = x_true[batch_indices, :, dst, :].transpose(1, 2)  # [B, N, Nbond, 3]
        d_true = torch.sqrt((src_true - dst_true).pow(2).sum(-1) + eps)  # [B, N, Nbond]

        diff = (d_pred - d_true) ** 2  # [B, N, Nbond]

        # Mask non-polymer-ligand bonds and invalid bonds
        mask = is_polymer_ligand_bond  # [B, Nbond]
        # Mask padding
        mask = mask & f_input.bond.pad_mask
        # Mask invalid atom pairs
        atom_mask = f_input.atom.resolved_mask  # [B, Latom]
        mask &= atom_mask[batch_indices, src] & atom_mask[batch_indices, dst]

        diff = diff.masked_fill(~mask[:, None, :], 0.0)  # [B, N, Nbond]
        num_bonds = mask.sum(dim=-1, keepdim=True, dtype=torch.float32)  # [B, 1]
        bond_loss = diff.sum(dim=-1) / num_bonds.clamp(min=1)  # [B, N]
        return bond_loss


class SmoothLDDTLoss(torch.nn.Module):
    """Smooth LDDT loss of denoised atom positions
    See Section 3.7.1 Algorithm 27 Smooth LDDT Loss of the AlphaFold 3 paper."""

    def __init__(
        self,
        cutoff: float = 15.0,
        cutoff_nucleic_acid: float = 30.0,
        repr_atom_only: bool = False,
        chunk_size: int | None = 1,
    ):
        """Initialize SmoothLDDTLoss.

        Parameters
        ----------
        cutoff: float
            The cutoff for non-nucleic acid atoms
        cutoff_nucleic_acid: float
            The cutoff for nucleic acid atoms
        repr_atom_only: bool
            Whether to compute LDDT loss using representative atoms instead of all atoms:
            [Lrepr, L] instead of [L, L]. This is a memory-saving option that can be used
            for larger chunk sizes. The representative atoms are defined as follows:
            - For proteins: Cb atoms
            - For nucleic acids: C4' atoms
            - For ligands: all atoms
        """

        super().__init__()
        self.cutoff: float = cutoff
        self.cutoff_nucleic_acid: float = cutoff_nucleic_acid
        self.repr_atom_only: bool = repr_atom_only
        self.chunk_size: int | None = chunk_size

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute weighted alignment.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted coordinates from the model. Shape (B, N, L, 3).
        x_true : torch.Tensor
            Ground truth coordinates. Shape (B, N, L, 3).
        f_input : FoldingInput
            The FoldingInput object containing model inputs.

        Returns
        -------
        lddt_loss: torch.Tensor
            Computed LDDT loss. Shape (B, N).
        """
        # NOTE: Due to the memory constraint, we change the order of operations
        # from the original paper implementation.

        assert x_pred.ndim == 4  # [B, N, L, 3]
        B, N = x_pred.shape[:2]

        # Mask to compute loss
        mask = f_input.atom.resolved_mask  # [B, Latom]

        # Line 5: is_nucleotide = is_dna | is_rna
        is_nucleotide = f_input.token.is_dna | f_input.token.is_rna  # [B, Ltoken]
        # [B, Ntoken] -> [B, Natom]
        batch_indices = torch.arange(B, device=f_input.device)[:, None]
        is_nucleotide = is_nucleotide[batch_indices, f_input.atom.token_index]

        # Get representative atom indices if needed
        repr_atom_index = f_input.token.repr_index

        losses = []
        for b_i in range(B):
            losses.extend(
                self._forward_single(
                    x_pred[b_i],  # [N, L, 3]
                    x_true[b_i],  # [N, L, 3]
                    mask[b_i],  # [L]
                    is_nucleotide[b_i],  # [L]
                    repr_atom_index[b_i],  # [L]
                )
            )
        lddt_loss = torch.cat(losses, dim=0)  # [B*N]
        lddt_loss = lddt_loss.view(B, N)  # [B, N]
        return lddt_loss

    def _forward_single(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        is_nucleotide: torch.Tensor,
        repr_atom_index: torch.Tensor,
    ) -> list[torch.Tensor]:
        N, L, _ = x_pred.shape
        # NOTE: pairwise distances of ground truth coordinates are shared across N.
        x_true = x_true[0]

        # Create pair mask
        pair_mask = mask[None, :] & mask[:, None]  # [L, L]
        # Mask out self-term
        pair_mask.diagonal(dim1=-2, dim2=-1).zero_()

        if self.repr_atom_only:
            # Extract representative atom indices
            d_true = safe_cdist(x_true[repr_atom_index], x_true)  # [Lrepr, L]
            pair_mask = pair_mask[repr_atom_index]  # [L, L] -> [Lrepr, L]
            is_nucleotide = is_nucleotide[repr_atom_index]  # [L] -> [Lrepr]
        else:
            d_true = safe_cdist(x_true, x_true)  # [L, L]

        # Mask out invalid distances
        pair_mask &= ((d_true < self.cutoff_nucleic_acid) & is_nucleotide[..., None]) | (
            (d_true < self.cutoff) & (~is_nucleotide[..., None])
        )  # [L, L] or [Lrepr, L]

        loss_fn = partial(
            self._chunk_forward,
            d_true=d_true,  # [L, L] or [Lrepr, L]
            pair_mask=pair_mask.float(),  # [L, L] or [Lrepr, L]
            repr_atom_index=repr_atom_index if self.repr_atom_only else None,
        )

        losses = []
        if self.chunk_size is None:
            losses.append(loss_fn(x_pred))  # [N,]
        else:
            for i in range(0, N, self.chunk_size):
                st, end = i, i + self.chunk_size
                x_chunk = x_pred[st:end]  # [chunk_size, L, 3]
                loss_chunk = checkpoint_section(
                    loss_fn, (x_chunk,), apply_ckpt=True, use_reentrant=False
                )
                losses.append(loss_chunk)
        return losses

    @staticmethod
    def _chunk_forward(
        x_pred: torch.Tensor,
        d_true: torch.Tensor,
        pair_mask: torch.Tensor,
        repr_atom_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Line 1
        if repr_atom_index is not None:
            # Compute predicted distances between representative atoms and all atoms
            x_pred_repr = x_pred[:, repr_atom_index]  # [N, Lrepr, 3]
            d_pred = safe_cdist(x_pred_repr, x_pred)  # [N, Lrepr, L]
        else:
            # Compute predicted pairwise distances (original AF3)
            d_pred = safe_cdist(x_pred, x_pred)  # [N, L, L]

        # Line 2 (outside function): compute true pairwise distances

        # Line 3
        d_diff = torch.abs(d_pred - d_true[None, ...])  # [N, L, L]

        # Line 4
        lddt_score = (1 / 4) * (
            torch.sigmoid(0.5 - d_diff)
            + torch.sigmoid(1.0 - d_diff)
            + torch.sigmoid(2.0 - d_diff)
            + torch.sigmoid(4.0 - d_diff)
        )  # [N, L, L]

        # Line 5: outside function (is_nucleotide = is_dna | is_rna)

        # Line 6: outside function (pair_mask = ...)

        # Line 7
        n_pair = pair_mask.sum((-1, -2)).clamp(min=1)  # scalar
        lddt = (lddt_score * pair_mask[None, ...]).sum((-1, -2)) / n_pair  # [N,]

        # Line 8
        lddt_loss = 1.0 - lddt  # [N,]
        return lddt_loss
