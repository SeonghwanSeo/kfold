# started from code from https://github.com/jwohlwend/boltz, MIT License

import warnings

import einops
import torch
import torch.nn.functional as F
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from kfold.data.model_input import FoldingInput


def weighted_rigid_align(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
):
    """Compute weighted alignment.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates of shape (..., L, 3)
    pred_coords: torch.Tensor
        The predicted atom coordinates of shape (..., L, 3)
    weights: torch.Tensor
        The weights for alignment of shape (..., L)

    Returns
    -------
    torch.Tensor
        Aligned coordinates of shape (..., L, 3)
    """

    device = true_coords.device
    with torch.autocast(device.type, enabled=False):
        L = true_coords.shape[-2]
        weights = weights.unsqueeze(-1)  # [..., L, 1]
        weight_sum = weights.sum(dim=-2, keepdim=True).clamp(1)  # [..., 1, 1]

        if L < 4:
            print(
                "Warning: The size of one of the point clouds is <= dim+1. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        # Compute weighted centroids
        true_centroid = (true_coords * weights).sum(
            dim=-2, keepdim=True
        ) / weight_sum  # [..., 1, 3]
        pred_centroid = (pred_coords * weights).sum(
            dim=-2, keepdim=True
        ) / weight_sum  # [..., 1, 3]

        # Center the coordinates
        true_coords_centered = true_coords - true_centroid  # [..., L, 3]
        pred_coords_centered = pred_coords - pred_centroid  # [..., L, 3]

        # Compute the weighted covariance matrix
        cov_matrix = einops.einsum(
            weights * pred_coords_centered,
            true_coords_centered,
            "... n i, ... n j -> ... i j",
        )

        # Compute the SVD of the covariance matrix, required float32 for svd and det
        original_dtype = cov_matrix.dtype
        cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
        U, S, V = torch.linalg.svd(
            cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
        )
        V = V.mH

        # Catch ambiguous rotation by checking the magnitude of singular values
        if (S.abs() <= 1e-15).any() and not (L < 4):
            warnings.warn(
                "Warning: Excessively low rank of "
                + "cross-correlation between aligned point clouds. "
                + "`WeightedRigidAlign` cannot return a unique rotation.",
                stacklevel=2,
            )

        # Compute the rotation matrix
        rot_matrix = torch.einsum("... i j, ... k j -> ... i k", U, V).to(
            dtype=torch.float32
        )

        # Ensure proper rotation matrix with determinant 1
        F = torch.eye(3, dtype=torch.float32, device=cov_matrix.device)  # [3, 3]
        F = F.unsqueeze(0).expand(*rot_matrix.shape[:-2], 3, 3).clone()  # [..., 3, 3]
        F[..., -1, -1] = torch.det(rot_matrix)  # Now broadcasts correctly

        rot_matrix = einops.einsum(U, F, V, "... i j, ... j k, ... l k -> ... i l")
        rot_matrix = rot_matrix.to(dtype=original_dtype)

        # Apply the rotation and translation
        aligned_coords = (
            einops.einsum(true_coords_centered, rot_matrix, "... n i, ... j i -> ... n j")
            + pred_centroid
        )
    return aligned_coords


class WeightedMSELoss(torch.nn.Module):
    """Weighted MSE loss of denoised atom positions
    See Section 3.7.1 Equation 2-3 of the AlphaFold 3 paper."""

    def __init__(
        self,
        weight_protein: float = 1.0,
        weight_dna: float = 5.0,
        weight_rna: float = 5.0,
        weight_ligand: float = 10.0,
        align: bool = True,
    ):
        super().__init__()
        self.weight_protein: float = weight_protein
        self.weight_dna: float = weight_dna
        self.weight_rna: float = weight_rna
        self.weight_ligand: float = weight_ligand
        self.align: bool = align

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
        align: bool = True,
    ) -> torch.Tensor:
        """Compute the weighted MSE loss.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted coordinates from the model. Shape (B, N, L, 3).
        x_true : torch.Tensor
            Ground truth coordinates. Shape (B, N, L, 3).
        align : bool, optional
            Whether to perform weighted rigid alignment before computing loss,

        Returns
        -------
        torch.Tensor
            Computed MSE loss. Shape (B, N)
        """
        weights = self.get_atom_weights(f_input)  # [B, L]
        weights = weights.unsqueeze(-2)  # [B, 1, L]

        # See Section 3.7.1 Equation 2
        if align or self.align:
            with torch.no_grad():
                x_true = weighted_rigid_align(
                    true_coords=x_true.float(),  # [B, N, L, 3]
                    pred_coords=x_pred.float(),  # [B, N, L, 3]
                    weights=weights.float(),  # [B, 1, L], broadcasted over N
                )  # [B, N, L, 3]

                x_true = x_true.to(dtype=x_true.dtype)

        # See Section 3.7.1 Equation 3
        d_sq = ((x_pred - x_true) ** 2).sum(dim=-1)  # [B, N, L]
        mse_loss = (1 / 3) * (weights * d_sq).mean(-1)  # [B, N]
        return mse_loss

    @torch.no_grad()
    def get_atom_weights(self, f_input: FoldingInput) -> torch.Tensor:
        """Compute atom weights for loss calculation.
        See Section 3.7.1 Equation 4 of the AlphaFold 3 paper.
        """
        is_protein = f_input.token.is_protein  # [B, Ltoken]
        is_dna = f_input.token.is_dna  # [B, Ltoken]
        is_rna = f_input.token.is_rna  # [B, Ltoken]
        is_ligand = f_input.token.is_ligand  # [B, Ltoken]

        token_weights = (
            is_protein.float() * self.weight_protein
            + is_dna.float() * self.weight_dna
            + is_rna.float() * self.weight_rna
            + is_ligand.float() * self.weight_ligand
        )  # [B, Ltoken]

        atom_weights = einops.einsum(
            f_input.atom_to_token,  # [B, Latom, Ltoken]
            token_weights,  # [B, Ltoken]
            "b l1 l2, b l2 -> b l1",
        )  # [B, Latom]

        # Masking
        # NOTE: this process
        mask = f_input.atom.resolved_mask  # [B, Latom]

        return atom_weights * mask  # [B, Latom]


class BondLoss(torch.nn.Module):
    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
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
        is_polymer_ligand_bond = f_input.bond.is_polymer_ligand_bond  # [B, Nbond]

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
        src_pred = x_pred[batch_indices, :, src, :]  # [B, N, Nbond, 3]
        dst_pred = x_pred[batch_indices, :, dst, :]  # [B, N, Nbond, 3]
        d_pred = torch.norm(src_pred - dst_pred, dim=-1)  # [B, N, Nbond]

        src_true = x_true[batch_indices, :, src, :]  # [B, N, Nbond, 3]
        dst_true = x_true[batch_indices, :, dst, :]  # [B, N, Nbond, 3]
        d_true = torch.norm(src_true - dst_true, dim=-1)  # [B, N, Nbond]

        d_diff = (d_pred - d_true) ** 2  # [B, N, Nbond]

        # Mask non-polymer-ligand bonds and invalid bonds
        bond_mask = f_input.bond.pad_mask  # [B, Nbond]
        # Mask padding
        mask = is_polymer_ligand_bond & bond_mask  # [B, Nbond]
        # Mask invalid atom pairs
        atom_mask = f_input.atom.resolved_mask  # [B, Latom]
        mask &= (
            atom_mask[batch_indices, src] & atom_mask[batch_indices, dst]
        )  # [B, Nbond]
        num_bonds = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]

        bond_loss = (d_diff * mask[:, None, :]).sum(dim=-1) / num_bonds  # [B, N]
        return bond_loss


class SmoothLDDTLoss(torch.nn.Module):
    """Smooth LDDT loss of denoised atom positions
    See Section 3.7.1 Algorithm 27 Smooth LDDT Loss of the AlphaFold 3 paper."""

    class ChunkedSmoothLDDTLoss(torch.nn.Module):
        """Chunked Smooth LDDT Loss for memory efficiency."""

        def __init__(self, cutoff: float = 15.0, cutoff_nucleic_acid: float = 30.0):
            super().__init__()
            self.cutoff: float = cutoff
            self.cutoff_nucleic_acid: float = cutoff_nucleic_acid

        def forward(
            self,
            x_pred: torch.Tensor,
            x_true: torch.Tensor,
            is_nucleotide: torch.Tensor,
            pair_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Compute weighted alignment.

            Parameters
            ----------
            x_pred : torch.Tensor
                Predicted coordinates from the model. Shape (B, L, 3).
            x_true : torch.Tensor
                Ground truth coordinates. Shape (B, L, 3).
            is_nucleotide : torch.Tensor
                The nucleotide mask for LDDT calculation. Shape (B, L).
            pair_mask : torch.Tensor
                The pair mask for LDDT calculation. Shape (B, L, L).

            Returns
            -------
            lddt_loss: torch.Tensor
                Computed LDDT Loss. Shape (B,)
            """
            # Line 1
            d_pred = torch.cdist(x_pred, x_pred)  # [B, L, L]

            # Line 2
            with torch.no_grad():
                d_true = torch.cdist(x_true, x_true)  # [B, L, L]

            # Line 3
            d_diff = torch.abs(d_true - d_pred)  # [B, L, L]

            # Line 4
            eps = (1 / 4) * (
                F.sigmoid(0.5 - d_diff)
                + F.sigmoid(1.0 - d_diff)
                + F.sigmoid(2.0 - d_diff)
                + F.sigmoid(4.0 - d_diff)
            )  # [B, L, L]

            # Line 5: outside function (is_nucleotide = is_dna | is_rna)

            # Line 6
            with torch.no_grad():
                c = ((d_true < self.cutoff_nucleic_acid) & is_nucleotide[..., None]) | (
                    (d_true < self.cutoff) & (~is_nucleotide[..., None])
                )  # [B, L, L]
                # Mask out invalid distances and self-term (see Line 7)
                c &= pair_mask  # [B, L, L]
            c = c.to(dtype=x_pred.dtype)

            # Line 7
            lddt = (eps * c).sum((-1, -2)) / c.sum((-1, -2)).clamp(1)  # [B,]

            # Line 8
            lddt_loss = 1.0 - lddt  # [B,]
            return lddt_loss

    def __init__(
        self,
        cutoff: float = 15.0,
        cutoff_nucleic_acid: float = 30.0,
        memory_efficient: bool = True,
    ):
        """Initialize SmoothLDDTLoss.

        Parameters
        ----------
        cutoff: float
            The cutoff for non-nucleic acid atoms
        cutoff_nucleic_acid: float
            The cutoff for nucleic acid atoms
        memory_efficient: bool
            Whether to use memory efficient implementation
        """

        super().__init__()
        self.cutoff: float = cutoff
        self.cutoff_nucleic_acid: float = cutoff_nucleic_acid

        # Initialize chunked smooth LDDT loss
        lddt_loss = self.ChunkedSmoothLDDTLoss(cutoff, cutoff_nucleic_acid)
        if memory_efficient:
            lddt_loss = checkpoint_wrapper(lddt_loss)
        self.lddt_loss = lddt_loss

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
        chunk_size: int | None = None,
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
        chunk_size : int | None
            The chunk size for memory efficient implementation.

        Returns
        -------
        lddt_loss: torch.Tensor
            Computed LDDT loss. Shape (B, N)
        """

        # NOTE: Due to the memory constraint, we change the order of operations
        # from the original paper implementation.

        B, N, L = x_pred.shape[:3]

        with torch.no_grad():
            # Line 5: is_nucleotide = is_dna | is_rna
            is_nucleotide = f_input.token.is_dna | f_input.token.is_rna  # [B, Ltoken]
            is_nucleotide = einops.einsum(
                f_input.atom_to_token,  # [B, Latom, Ltoken]
                is_nucleotide.float(),  # [B, Ltoken]
                "b l1 l2, b l2 -> b l1",
            ).bool()  # [B, Latom]
            is_nucleotide = is_nucleotide.unsqueeze(1).repeat(1, N, 1)  # [B, N, L]

            # Prepare masking
            mask = f_input.atom.resolved_mask  # [B, Latom]
            pair_mask = mask[:, None, :] & mask[:, :, None]  # [B, L, L]
            # mask self-distances
            pair_mask.diagonal(dim1=-2, dim2=-1).fill_(0)
            pair_mask = pair_mask.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, L, L]

        # Reshape inputs
        x_pred = x_pred.view(B * N, L, 3)  # [B*N, L, 3]
        x_true = x_true.view(B * N, L, 3)  # [B*N, L, 3]
        is_nucleotide = is_nucleotide.view(B * N, L)  # [B*N, L]
        pair_mask = pair_mask.view(B * N, L, L)  # [B*N, L, L]

        if chunk_size is not None and chunk_size < (B * N):
            losses = []
            BN = x_pred.shape[0]
            for i in range(0, BN, chunk_size):
                st, end = i, min(i + chunk_size, BN)
                loss_chunk = self.lddt_loss(
                    x_pred[st:end],
                    x_true[st:end],
                    is_nucleotide[st:end],
                    pair_mask[st:end],
                )  # [chunk_size,]
                losses.append(loss_chunk)

            lddt_loss = torch.cat(losses, dim=0)  # [B*N]
        else:
            lddt_loss = self.lddt_loss(x_pred, x_true, is_nucleotide, pair_mask)  # [B*N]

        lddt_loss = lddt_loss.view(B, N)  # [B, N]

        return lddt_loss
