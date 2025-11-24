# started from code from https://github.com/jwohlwend/boltz, MIT License
import torch
import torch.nn.functional as F

from kfold.data.model_input import FoldingInput
from kfold.utils.checkpointing import checkpoint_section
from kfold.utils.geometry.rigid_align import weighted_rigid_align


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
        scale: bool = False,
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
        scale: bool
            Whether to divide by the sum of weights.
            Boltz1: scale.
            AlphaFold3, Protenix, OpenFold-3: do not scale.
            NOTE: loss value is lower when scale=True.
        """
        super().__init__()
        self.upweight_protein: float = upweight_protein
        self.upweight_dna: float = upweight_dna
        self.upweight_rna: float = upweight_rna
        self.upweight_ligand: float = upweight_ligand
        self.scale: bool = scale

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute the weighted MSE loss.

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
        torch.Tensor
            Computed MSE loss. Shape (B, N).
        """
        assert x_pred.ndim == 4  # [B, N, L, 3]
        assert x_pred.shape == x_true.shape

        w = self.get_atom_weights(f_input)  # [B, L]
        mask = f_input.atom.resolved_mask  # [B, L]
        w = w * mask  # [B, L]

        w = w.unsqueeze(-2)  # [B, 1, L]
        mask = mask.unsqueeze(-2)  # [B, 1, L]

        # See Section 3.7.1 Equation 2
        with torch.no_grad():
            x_true_aligned = weighted_rigid_align(
                coords=x_true.float(),  # [B, N, L, 3]
                target=x_pred.float(),  # [B, N, L, 3]
                weights=w,  # [B, 1, L], broadcasted over N
                mask=mask,  # [B, 1, L]
            )  # [B, N, L, 3]

        d_sq = ((x_pred - x_true_aligned) ** 2).sum(dim=-1)  # [B, N, L]
        if self.scale:
            weight_sum = w.sum(-1).clamp(min=1)  # [B, 1]
            mse_loss = (1 / 3) * (w * d_sq).sum(-1) / weight_sum  # [B, N]
        else:
            mask_sum = mask.sum(dim=-1).clamp(min=1)  # [B, 1]
            mse_loss = (1 / 3) * (w * d_sq).sum(-1) / mask_sum  # [B, N]

        return mse_loss

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

    def __init__(
        self,
        cutoff: float = 15.0,
        cutoff_nucleic_acid: float = 30.0,
    ):
        """Initialize SmoothLDDTLoss.

        Parameters
        ----------
        cutoff: float
            The cutoff for non-nucleic acid atoms
        cutoff_nucleic_acid: float
            The cutoff for nucleic acid atoms
        """

        super().__init__()
        self.cutoff: float = cutoff
        self.cutoff_nucleic_acid: float = cutoff_nucleic_acid

    def _chunk_forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        is_nucleotide: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
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

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
        chunk_size: int | None = 1,
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
            Computed LDDT loss. Shape (B, N).
        """

        # NOTE: Due to the memory constraint, we change the order of operations
        # from the original paper implementation.

        assert x_pred.ndim == 4  # [B, N, L, 3]
        B, N, L = x_pred.shape[:3]

        # Line 5: is_nucleotide = is_dna | is_rna
        is_nucleotide = f_input.token.is_dna | f_input.token.is_rna  # [B, Ltoken]
        # [B, Ntoken] -> [B, Natom]
        batch_indices = torch.arange(B, device=f_input.device)[:, None]
        is_nucleotide = is_nucleotide[batch_indices, f_input.atom.token_index]

        # Prepare masking
        mask = f_input.atom.resolved_mask  # [B, Latom]
        pair_mask = mask[:, None, :] & mask[:, :, None]  # [B, L, L]
        # mask self-distances
        pair_mask.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Reshape inputs for chunking
        x_pred = x_pred.view(B * N, L, 3)  # [B*N, L, 3]
        x_true = x_true.view(B * N, L, 3)  # [B*N, L, 3]
        is_nucleotide = is_nucleotide.repeat_interleave(N, dim=0)  # [B*N, L]
        pair_mask = pair_mask.repeat_interleave(N, dim=0)  # [B*N, L, L]

        BN = x_pred.shape[0]
        if chunk_size is not None:
            losses = []
            for i in range(0, BN, chunk_size):
                st, end = i, i + chunk_size
                loss_chunk = checkpoint_section(
                    self._chunk_forward,
                    (
                        x_pred[st:end],
                        x_true[st:end],
                        is_nucleotide[st:end],
                        pair_mask[st:end],
                    ),
                    apply_ckpt=True,
                    use_reentrant=False,
                )
                losses.append(loss_chunk)
            lddt_loss = torch.cat(losses, dim=0)  # [B*N]
        else:
            lddt_loss = self._chunk_forward(
                x_pred,
                x_true,
                is_nucleotide,
                pair_mask,
            )  # [B*N]

        lddt_loss = lddt_loss.view(B, N)  # [B, N]

        return lddt_loss
