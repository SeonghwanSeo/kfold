from functools import partial

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.checkpointing import checkpoint_fn
from kfold.utils.geometry.rigid_align import weighted_rigid_align
from kfold.utils.kernels.cdist import cdist as kernel_cdist


def safe_cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pairwise distances between two sets of points."""
    d = x[..., :, None, :] - y[..., None, :, :]  # [*, Lx, Ly, 3]
    return torch.sqrt(d.pow(2).sum(-1) + eps)


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
        upweight_protein: float
            The additional weight for protein atoms
        upweight_dna: float
            The additional weight for DNA atoms
        upweight_rna: float
            The additional weight for RNA atoms
        upweight_ligand: float
            The additional weight for ligand atoms
        align_true_to_pred: bool
            Whether to align ground truth coordinates to predictions before MSE.
        """
        super().__init__()
        self.upweight_protein: float = upweight_protein
        self.upweight_dna: float = upweight_dna
        self.upweight_rna: float = upweight_rna
        self.upweight_ligand: float = upweight_ligand
        self.align_true_to_pred: bool = align_true_to_pred

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
        compute_chain_com_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute the weighted MSE loss.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted coordinates from the model. Shape (B, N, L, 3).
        x_true : torch.Tensor
            Ground truth coordinates. Shape (B, N, L, 3).
        f_input : FoldingInput
            The FoldingInput object containing model inputs.
        compute_chain_com_loss: bool
            Whether to compute the additional chain COM loss term.

        Returns
        -------
        weighted_mse_loss : torch.Tensor
            Computed MSE loss. Shape (B, N).
        chain_com_loss : torch.Tensor | None
            Computed chain COM loss if compute_chain_com_loss is True, else None.
        """
        assert x_pred.ndim == 4  # [B, N, L, 3]
        assert x_pred.shape == x_true.shape

        w = self.get_atom_weights(f_input)  # [B, L]
        mask = f_input.atom.resolved_mask  # [B, L]
        w = w * mask  # [B, L]

        # See Section 3.7.1 Equation 2
        if self.align_true_to_pred:
            with torch.no_grad():
                x_true = weighted_rigid_align(
                    coords=x_true.float(),  # [B, N, L, 3]
                    target=x_pred.float(),  # [B, N, L, 3]
                    weights=w.unsqueeze(-2),  # [B, 1, L]
                    mask=mask.unsqueeze(-2),  # [B, 1, L]
                )  # [B, N, L, 3]

        # Compute weighted mse loss
        mse_loss = self.compute_weighted_mse_loss(x_pred, x_true, w, mask)  # [B, N]

        if compute_chain_com_loss:
            # Additional chain COM loss term
            b_i = torch.arange(f_input.batch_size, device=f_input.device)[:, None]
            t_i = f_input.atom.token_index  # [B, Latom]
            asym_id = f_input.token.asym_id[b_i, t_i]  # [B, Latom]
            com_loss = self.compute_chain_com_loss(
                x_pred, x_true, w, mask, asym_id, f_input.num_chains
            )
            return mse_loss, com_loss
        else:
            return mse_loss, None

    def get_atom_weights(self, f_input: FoldingInput) -> torch.Tensor:
        """Compute atom weights for loss calculation.
        See Section 3.7.1 Equation 4 of the AlphaFold 3 paper.
        """
        token_weights = (
            f_input.token.is_protein.float() * (1 + self.upweight_protein)
            + f_input.token.is_dna.float() * (1 + self.upweight_dna)
            + f_input.token.is_rna.float() * (1 + self.upweight_rna)
            + f_input.token.is_ligand.float() * (1 + self.upweight_ligand)
        )  # [B, Ltoken]
        b_i = torch.arange(f_input.batch_size, device=f_input.device)[:, None]
        atom_weights = token_weights[b_i, f_input.atom.token_index]  # [B, Latom]
        return atom_weights

    def compute_weighted_mse_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the weighted MSE loss.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted coordinates of shape [B, N, L, 3].
        x_true : torch.Tensor
            Aligned ground-truth coordinates of shape [B, N, L, 3].
        weights : torch.Tensor
            Modality weights per atom of shape [B, L].
        mask : torch.Tensor
            Resolved mask per atom of shape [B, L].

        Returns
        -------
        mse_loss : torch.Tensor
            Weighted MSE loss of shape [B, N], averaged over atoms with modality weights.
        """
        # Add num-sample dimension
        m, w = mask[:, None, :], weights[:, None, :]  # [B, 1, L]

        # Compute weighted MSE loss
        mask_sum = m.sum(-1).clamp(min=1.0)  # [B, 1]
        d_sq = ((x_pred - x_true) ** 2).sum(-1)  # [B, N, L]
        mse_loss = (1 / 3) * (w * d_sq).sum(-1) / mask_sum  # [B, N]
        return mse_loss

    def compute_chain_com_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        asym_id: torch.Tensor,
        max_chains: int,
    ) -> torch.Tensor:
        """Compute chain-wise COM losses.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted coordinates of shape [B, N, L, 3].
        x_true : torch.Tensor
            Aligned ground-truth coordinates of shape [B, N, L, 3].
        weights : torch.Tensor
            Modality weights per atom of shape [B, L].
        mask : torch.Tensor
            Resolved mask per atom of shape [B, L].
        asym_id : torch.Tensor
            Chain id per atom of shape [B, L].

        Returns
        -------
        com_loss : torch.Tensor
            Chain COM loss of shape [B, N], averaged over atoms in each chain and weighted
            by the modality weights. Empty chain slots are ignored in the loss.
        """
        B, N, L, _ = x_pred.shape

        # Renumber asym id to chain id: 1, 2, 5, 6, ... -> 1, 2, 3, ..., 0 0 0(unresolved)
        chain_id = torch.zeros_like(asym_id)
        for b_i in range(B):
            mask_i = mask[b_i]  # [L]
            asym_id_i = asym_id[b_i]
            uniq_id = torch.sort(torch.unique(asym_id_i[mask_i]))[0]  # [Nchain]
            for i, asym in enumerate(uniq_id, start=1):
                chain_id[b_i, asym_id_i == asym] = i  # [B, Ntoken]
        chain_id[~mask] = 0  # Unresolved atoms get chain id 0

        def compute_chain_com(x: torch.Tensor) -> torch.Tensor:
            """Compute chain-wise COM for each sample in the batch.
            [B, N, L, 3] -> [B, N, Nchain, 3]
            """
            num_chains = max_chains + 1  # 0 is reserved for padding atoms
            idx = chain_id[:, None, :, None].expand(x.shape)
            com = torch.zeros(
                (B, N, num_chains, 3), device=x.device, dtype=x.dtype
            ).scatter_reduce_(-2, idx, x, reduce="mean", include_self=False)
            return com

        com_pred = compute_chain_com(x_pred)  # [B, N, Nchain, 3]
        com_true = compute_chain_com(x_true)  # [B, N, Nchain, 3]

        # Broadcast each chain's centroid to its atoms.
        # This aims to weight the COM error by the number of atoms in each chain.
        gather_index = chain_id[:, None, :, None].expand_as(x_pred)
        com_pred = torch.gather(com_pred, 2, gather_index)
        com_true = torch.gather(com_true, 2, gather_index)

        m, w = mask[:, None, :], weights[:, None, :]  # [B, 1, L]
        mask_sum = m.sum(-1).clamp(min=1.0)  # [B, 1]
        com_sq = ((com_pred - com_true) ** 2).sum(-1)
        com_loss = (w * com_sq).sum(-1) / mask_sum  # [B, N]
        return com_loss


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
        use_kernel: bool = False,
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
        chunk_size: int | None
            The chunk size for computing LDDT loss.
        use_kernel: bool
            Whether to use triton implementation for pairwise distance calculation.
        """

        super().__init__()
        self.cutoff: float = cutoff
        self.cutoff_nucleic_acid: float = cutoff_nucleic_acid
        self.repr_atom_only: bool = repr_atom_only
        self.chunk_size: int | None = chunk_size
        self.use_kernel: bool = use_kernel

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
        use_kernel: bool = False,
    ) -> list[torch.Tensor]:
        N, L, _ = x_pred.shape
        # NOTE: pairwise distances of ground truth coordinates are shared across N.
        x_true = x_true[0]

        # Create pair mask
        pair_mask = mask[None, :] & mask[:, None]  # [L, L]
        # Mask out self-term
        pair_mask.diagonal(dim1=-2, dim2=-1).zero_()

        _cdist = kernel_cdist if self.use_kernel else safe_cdist

        if self.repr_atom_only:
            # Extract representative atom indices
            d_true = _cdist(x_true[repr_atom_index], x_true)  # [Lrepr, L]
            pair_mask = pair_mask[repr_atom_index]  # [L, L] -> [Lrepr, L]
            is_nucleotide = is_nucleotide[repr_atom_index]  # [L] -> [Lrepr]
        else:
            d_true = _cdist(x_true, x_true)  # [L, L]

        # Mask out invalid distances
        pair_mask &= ((d_true < self.cutoff_nucleic_acid) & is_nucleotide[..., None]) | (
            (d_true < self.cutoff) & (~is_nucleotide[..., None])
        )  # [L, L] or [Lrepr, L]

        loss_fn = partial(
            self._chunk_forward,
            d_true=d_true,  # [L, L] or [Lrepr, L]
            pair_mask=pair_mask,  # [L, L] or [Lrepr, L]
            repr_atom_index=repr_atom_index if self.repr_atom_only else None,
            use_kernel=self.use_kernel,
        )

        losses = []
        if self.chunk_size is None:
            losses.append(loss_fn(x_pred))  # [N,]
        else:
            for i in range(0, N, self.chunk_size):
                st, end = i, i + self.chunk_size
                x_chunk = x_pred[st:end]  # [chunk_size, L, 3]
                loss_chunk = checkpoint_fn(
                    loss_fn,
                    x_chunk,
                    use_reentrant=False,
                    determinism_check="none",  # No randomness in loss function
                )
                losses.append(loss_chunk)
        return losses

    @staticmethod
    def _chunk_forward(
        x_pred: torch.Tensor,
        d_true: torch.Tensor,
        pair_mask: torch.Tensor,
        repr_atom_index: torch.Tensor | None = None,
        use_kernel: bool = False,
    ) -> torch.Tensor:
        _cdist = kernel_cdist if use_kernel else safe_cdist

        # Line 1
        if repr_atom_index is not None:
            # Compute predicted distances between representative atoms and all atoms
            x_pred_repr = x_pred[:, repr_atom_index]  # [N, Lrepr, 3]
            d_pred = _cdist(x_pred_repr, x_pred)  # [N, Lrepr, L]
        else:
            # Compute predicted pairwise distances (original AF3)
            d_pred = _cdist(x_pred, x_pred)  # [N, L, L]

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
        pair_mask = pair_mask.float()
        n_pair = pair_mask.sum((-1, -2)).clamp(min=1)  # scalar
        lddt = (lddt_score * pair_mask[None, ...]).sum((-1, -2)) / n_pair  # [N,]

        # Line 8
        lddt_loss = 1.0 - lddt  # [N,]
        return lddt_loss
