import torch

from kfold.constants import chain as const
from kfold.training.folding.loss.confidence import lddt_dist

def factored_token_lddt_dist_loss(true_d, pred_d, f_input, cardinality_weighted=False):
    """Compute the distogram lddt factorized into the different modalities.

    Parameters
    ----------
    true_d : torch.Tensor
        Ground truth atom distogram
    pred_d : torch.Tensor
        Predicted atom distogram
    f_input : Dict[str, torch.Tensor]
        Input features

    Returns
    -------
    Tensor
        The lddt for each modality
    Tensor
        The total number of pairs for each modality

    """
    # extract necessary features
    token_type = f_input["mol_type"]

    ligand_mask = (token_type == const.ChainType.LIGAND).float()
    dna_mask = (token_type == const.ChainType.DNA).float()
    rna_mask = (token_type == const.ChainType.RNA).float()
    protein_mask = (token_type == const.ChainType.PROTEIN).float()

    nucleotide_mask = dna_mask + rna_mask

    token_mask = f_input["disto_mask"] # (B, T)
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    token_mask = token_mask * (1 - torch.eye(token_mask.shape[1])[None]).to(token_mask) # (B, T, T)

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # compute different lddts
    dna_protein_mask = token_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )

    rna_protein_mask = token_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )

    ligand_protein_mask = token_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )

    dna_ligand_mask = token_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )

    rna_ligand_mask = token_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )

    chain_id = f_input["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()
    intra_ligand_mask = (
        token_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )

    intra_dna_mask = token_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)

    intra_rna_mask = token_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)

    # chain_id = f_input["asym_id"]
    # same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float() # TODO: delete

    intra_protein_mask = (
        token_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )

    protein_protein_mask = (
        token_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )
    
    # TODO: add Modified, Unresolved type (*AF3 5.7)

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    if not cardinality_weighted: # if cardinality_weighted is False, all total_dict values are set to 1.0 or 0.0 (no matter what atom number is)
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict

def factored_lddt_loss(
    true_atom_coords,
    pred_atom_coords,
    f_input,
    atom_mask,
    num_diffusion_samples=1,
    cardinality_weighted=False,
):
    """Compute the lddt factorized into the different modalities.

    Parameters
    ----------
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates after symmetry correction
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    f_input : Dict[str, torch.Tensor]
        Input features
    atom_mask : torch.Tensor
        Atom mask
    num_diffusion_samples : int
        Diffusion batch size, by default 1

    Returns
    -------
    Dict[str, torch.Tensor]
        The lddt for each modality
    Dict[str, torch.Tensor]
        The total number of pairs for each modality

    """
    # extract necessary features
    # 기존 코드 -----------------------------------------
    # atom_type = (
    #     torch.bmm(
    #         f_input["atom_to_token"].float(), f_input["mol_type"].unsqueeze(-1).float()
    #     ) 
    #     .squeeze(-1)
    #     .long()
    # )
    # -----------------------------------------------------
    mol_type = f_input["mol_type"] # (B, N_token,)
    atom_to_token = f_input["atom_to_token"] # (B, N_atom,)
    atom_type = torch.gather(mol_type, 1, atom_to_token).long() # (B, N_atom,)
    atom_type = atom_type.repeat_interleave(num_diffusion_samples, 0)

    ligand_mask = (atom_type == const.ChainType.LIGAND).float()
    dna_mask = (atom_type == const.ChainType.DNA).float()
    rna_mask = (atom_type == const.ChainType.RNA).float()
    protein_mask = (atom_type == const.ChainType.PROTEIN).float()

    nucleotide_mask = dna_mask + rna_mask

    # TODO: check what is efficient (B, M) or (B*M) ?
    B, M, N, _ = true_atom_coords.shape
    true_atom_coords = true_atom_coords.view(B * M, N, 3)
    pred_atom_coords = pred_atom_coords.view(B * M, N, 3)
    atom_mask = atom_mask.view(B * M, N) # (B, N) -> (B*M, N)
    true_d = torch.cdist(true_atom_coords, true_atom_coords)
    pred_d = torch.cdist(pred_atom_coords, pred_atom_coords)

    pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # compute different lddts ------------------------------------
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )
    del dna_protein_mask

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )
    del rna_protein_mask

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )
    del ligand_protein_mask

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )
    del dna_ligand_mask

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )
    del rna_ligand_mask

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)
    del intra_dna_mask

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)
    del intra_rna_mask

    # 기존 코드 -----------------------------------------
    # chain_id = f_input["asym_id"]
    # atom_chain_id = (
    #     torch.bmm(f_input["atom_to_token"].float(), chain_id.unsqueeze(-1).float())
    #     .squeeze(-1)
    #     .long()
    # )
    # -----------------------------------------------------
    chain_id = f_input["asym_id"]
    atom_chain_id = torch.gather(chain_id, 1, atom_to_token).long()
    atom_chain_id = atom_chain_id.repeat_interleave(num_diffusion_samples, 0)
    same_chain_mask = (atom_chain_id[:, :, None] == atom_chain_id[:, None, :]).float()

    intra_ligand_mask = (
        pair_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )
    del intra_ligand_mask

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )
    del intra_protein_mask

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )
    del protein_protein_mask

    # TODO: add Modified, Unresolved type (*AF3 5.7)
    # -------------------------------------------------------------

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }
    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict

# TODO: find another place to put this function (if use this function in other places (not only validation))
# TODO: revise it
def get_true_coordinates(
    self,
    batch,
    out,
    num_diffusion_samples,
    symmetry_correction,
    lddt_minimization=True,
):
    ## TODO : if symmetry 구현

    true_coords = (
        batch["label_coords"].squeeze(1).repeat_interleave(num_diffusion_samples, 0)
    )

    true_coords_resolved_mask = batch["resolved_mask"].repeat_interleave(
        num_diffusion_samples, 0
    )
    rmsds, best_rmsds = weighted_minimum_rmsd(
        out["sample"]["coordinates"],
        batch,
        multiplicity=num_diffusion_samples,
        nucleotide_weight=self.nucleotide_rmsd_weight,
        ligand_weight=self.ligand_rmsd_weight,
    )

    return true_coords, rmsds, best_rmsds, true_coords_resolved_mask

# TODO: revise it
def weighted_minimum_rmsd(
    pred_atom_coords,
    feats,
    multiplicity=1,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
):
    """Compute rmsd of the aligned atom coordinates.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The rmsds
    Tensor
        The best rmsd

    """
    atom_coords = feats["coords"]
    atom_coords = atom_coords.repeat_interleave(multiplicity, 0)
    atom_coords = atom_coords[:, 0]

    atom_mask = feats["atom_resolved_mask"]
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

    align_weights = atom_coords.new_ones(atom_coords.shape[:2])
    atom_type = (
        torch.bmm(
            feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
        )
        .squeeze(-1)
        .long()
    )
    atom_type = atom_type.repeat_interleave(multiplicity, 0)

    align_weights = align_weights * (
        1
        + nucleotide_weight
        * (
            torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
            + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
        )
        + ligand_weight
        * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
    )

    with torch.no_grad():
        atom_coords_aligned_ground_truth = weighted_rigid_align(
            atom_coords, pred_atom_coords, align_weights, mask=atom_mask
        )

    # weighted MSE loss of denoised atom positions
    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )
    best_rmsd = torch.min(rmsd.reshape(-1, multiplicity), dim=1).values

    return rmsd, best_rmsd