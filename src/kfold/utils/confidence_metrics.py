"""Functions for computing confidence metrics from the model's predicted logits."""

import itertools

import numpy as np
import torch

from kfold.data.types import FoldingInput, RefStructure
from kfold.model.layers.alphafold3.utils import broadcast_tokens_to_atoms


def compute_plddt(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the predicted lDDT from the lDDT logits.

    Parameters
    ----------
    logits: torch.Tensor
        Logits for the predicted lDDT,
        of shape [*, L, num_bins]
    bin_centers: torch.Tensor
        Bin centers for the lDDT bins, of shape [num_bins]
    mask: torch.Tensor | None
        Optional boolean mask of shape [L] indicating valid residues.

    Returns
    -------
    plddt: torch.Tensor
        Predicted lDDT of shape [*, L]
    """
    L = logits.shape[-2]
    if mask is None:
        mask = torch.ones(L, dtype=torch.bool, device=logits.device)
    assert mask.shape == (L,), (
        f"Mask shape {mask.shape} does not match expected shape {(L,)}"
    )
    probs = torch.softmax(logits, dim=-1)  # [*, L, num_bins]
    plddt = (probs * bin_centers).sum(dim=-1)  # [*, L]
    plddt *= mask.float()  # [*, L]
    return plddt  # [*, L]


def compute_pde(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the predicted aligned error (PAE) from the PAE logits.

    Parameters
    ----------
    logits: torch.Tensor
        Logits for the predicted pairwise distance error,
        of shape [*, L, L, num_bins]
    bin_centers: torch.Tensor
        Bin centers for the pairwise distance error bins, of shape [num_bins]
    mask: torch.Tensor | None
        Optional boolean mask of shape [L] indicating valid residues.

    Returns
    -------
    pde: torch.Tensor
        Predicted pairwise distance error of shape [*, L, L]
    """
    L = logits.shape[-2]
    if mask is None:
        mask = torch.ones(L, dtype=torch.bool, device=logits.device)
    assert mask.shape == (L,), (
        f"Mask shape {mask.shape} does not match expected shape {(L,)}"
    )
    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    pde = (probs * bin_centers).sum(dim=-1)  # [*, L, L]
    pair_mask = mask[:, None] & mask[None, :]  # [L, L]
    pde *= pair_mask.float()  # [*, L, L]
    return pde  # [L, L]


def compute_pae(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the predicted aligned error (PAE) from the PAE logits.

    Parameters
    ----------
    logits: torch.Tensor
        Logits for the predicted pairwise aligned error,
        of shape [*, L, L, num_bins]
    bin_centers: torch.Tensor
        Bin centers for the pairwise distance error bins, of shape [num_bins]
    mask: torch.Tensor | None
        Optional boolean mask of shape [L] indicating valid residues.

    Returns
    -------
    pae: torch.Tensor
        Predicted aligned error of shape [*, L, L]
    """
    L = logits.shape[-2]
    if mask is None:
        mask = torch.ones(L, dtype=torch.bool, device=logits.device)
    assert mask.shape == (L,), (
        f"Mask shape {mask.shape} does not match expected shape {(L,)}"
    )
    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    pae = (probs * bin_centers).sum(dim=-1)  # [*, L, L]
    pair_mask = mask[..., :, None] & mask[..., None, :]
    pae *= pair_mask.float()  # [*, L, L]
    return pae  # [*, L, L]


def compute_ptm(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the predicted TM-score from the PAE logits.

    Parameters
    ----------
    logits: torch.Tensor
        Logits for the predicted pairwise aligned error,
        of shape [*, L, L, num_bins]
    bin_centers: torch.Tensor
        Bin centers for the pairwise distance error bins, of shape [num_bins]
    mask: torch.Tensor | None
        Optional boolean mask of shape [L] indicating valid residues.

    Returns
    -------
    ptm: torch.Tensor
        Predicted TM-score of shape [*]
    """
    L = logits.shape[-2]
    if mask is None:
        mask = torch.ones(L, dtype=torch.bool, device=logits.device)
    assert mask.shape == (L,), (
        f"Mask shape {mask.shape} does not match expected shape {(L,)}"
    )

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    n = int(mask.sum().item())
    clipped_n = max(n, 19)
    d0: float = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8
    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]

    # Avoid broadcasting issues when processing batches
    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))  # [*, num_bins]
    ptm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [*, L, L]

    pair_mask = mask[:, None] & mask[None, :]  # [L, L]
    w = pair_mask.float()

    ptm_term = ptm_term * w  # [*, L, L]
    denom = eps + w.sum(-1)  # [*, L]
    per_alignment = torch.sum(ptm_term / denom[..., None], dim=-1)  # [*, L]
    weighted = per_alignment * mask.float()  # [*, L]
    return weighted.max(-1).values  # [*]


def compute_iptm(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    asym_id: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the interface predicted TM-score from the PAE logits.

    Parameters
    ----------
    logits: torch.Tensor
        Logits for the predicted pairwise aligned error,
        of shape [*, L, L, num_bins]
    bin_centers: torch.Tensor
        Bin centers for the pairwise distance error bins, of shape [num_bins]
    asym_id: torch.Tensor
        Asymmetric unit IDs for each residue, of shape [L]
    mask: torch.Tensor | None
        Optional boolean mask of shape [L] indicating valid residues.

    Returns
    -------
    iptm: torch.Tensor
        Predicted interface TM-score of shape [*]
    """
    L = logits.shape[-2]
    if mask is None:
        mask = torch.ones(L, dtype=torch.bool, device=logits.device)
    assert mask.shape == (L,), (
        f"Mask shape {mask.shape} does not match expected shape {(L,)}"
    )

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    n = int(mask.sum().item())  # [L]
    clipped_n = max(n, 19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8
    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]

    # Avoid broadcasting issues when processing batches
    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))  # [*, L, num_bins]
    ptm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [*, L, L]

    pair_mask = mask[:, None] & mask[None, :]  # [L, L]
    pair_mask &= asym_id[:, None] != asym_id[None, :]
    w = pair_mask.float()

    ptm_term = ptm_term * w  # [*, L, L]
    denom = eps + w.sum(-1)  # [*, L]
    per_alignment = torch.sum(ptm_term / denom[..., None], dim=-1)  # [*, L]
    weighted = per_alignment * mask.float()  # [*, L]
    return weighted.max(-1).values  # [*]


def compute_has_clash(
    sample_coords: torch.Tensor,
    asym_id: torch.Tensor,
    is_polymer: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    threshold: float = 1.1,
    max_clash_num: int = 100,
    max_clash_frac: float = 0.1,
) -> torch.Tensor:
    """
    Compute has_clash metric.

    Parameters
    ----------
    sample_coords: torch.Tensor
        Predicted atom coordinates, of shape [*, Natom, 3]
    asym_id: torch.Tensor
        Asymmetric unit IDs for each atom, of shape [Natom]
    atom_mask: torch.Tensor
        Boolean mask indicating valid atoms, of shape [Natom]
    is_polymer: torch.Tensor
        Boolean mask indicating whether each atom belongs to a polymer, of shape [Natom]
    threshold: float
        Distance threshold for considering a clash, in Angstroms.
    max_clash_num: int
        Maximum number of clashes allowed for a chain pair.
    max_clash_frac: float
        Maximum fraction of clashing residues allowed for a chain pair.

    Returns
    -------
    has_clash: torch.Tensor
        Boolean tensor of shape [*] indicating whether each sample has a clash.
    """
    L = sample_coords.shape[-2]
    if atom_mask is None:
        atom_mask = torch.ones(L, dtype=torch.bool, device=sample_coords.device)

    # Remove padding
    n_atoms: int = int(atom_mask.sum().item())
    sample_coords = sample_coords[..., :n_atoms, :]
    asym_id = asym_id[:n_atoms]
    atom_mask = atom_mask[:n_atoms]
    is_polymer = is_polymer[:n_atoms]

    if not (asym_id > 0).all():
        raise ValueError("All asym_id values must be positive integers.")
    if not atom_mask.all():
        raise ValueError("All atoms must be valid (atom_mask must be all True).")

    unique_asym_ids: list[int] = sorted(torch.unique(asym_id).tolist())
    chain_masks: dict[int, torch.Tensor] = {
        aid: (asym_id == aid) for aid in unique_asym_ids
    }
    is_polymer_chain: dict[int, bool] = {
        aid: bool(is_polymer[chain_mask].any().item())
        for aid, chain_mask in chain_masks.items()
    }

    def detect_clash(coords: torch.Tensor) -> bool:
        for ai, aj in itertools.combinations(unique_asym_ids, 2):
            if not (is_polymer_chain[ai] and is_polymer_chain[aj]):
                continue
            chain_i = coords[chain_masks[ai], :]
            chain_j = coords[chain_masks[aj], :]
            ni, nj = chain_i.shape[0], chain_j.shape[0]
            if ni == 0 or nj == 0:
                continue
            d = (chain_i[:, None, :] - chain_j[None, :, :]).norm(-1)
            n_clash = (d < threshold).sum().item()
            if n_clash > max_clash_num:
                return True
            if n_clash / min(ni, nj) > max_clash_frac:
                return True
        return False

    if sample_coords.ndim == 2:
        return torch.tensor(detect_clash(sample_coords), device=sample_coords.device)
    else:
        has_clash = []
        for coords in sample_coords:
            has_clash.append(detect_clash(coords))
        return torch.tensor(has_clash, device=sample_coords.device)


def summarize_confidence_metrics(
    f_input: FoldingInput,
    ref_struct: RefStructure,
    model_out: dict[str, dict[str, torch.Tensor]],
) -> tuple[list[dict], list[dict[str, np.ndarray]]]:
    if f_input.is_batched:
        raise NotImplementedError(
            "full_complex_sample_ranking_metric does not support batched inputs"
        )
    sample_coords = model_out["diffusion"]["coordinates"]  # [Nsample, Natom, 3]
    summary_list = []
    score_list = []
    for i in range(sample_coords.shape[0]):
        summary, scores = summarize_confidence_metrics_single(
            f_input=f_input,
            ref_struct=ref_struct,
            model_out=model_out,
            sample_index=i,
        )
        summary_list.append(summary)
        score_list.append(scores)
    return summary_list, score_list


def summarize_confidence_metrics_single(
    f_input: FoldingInput,
    ref_struct: RefStructure,
    model_out: dict[str, dict[str, torch.Tensor]],
    sample_index: int = 0,
) -> tuple[dict, dict[str, np.ndarray]]:
    """
    AlphaFold3 sample ranking metric for the full complex.
    See Section 5.9.3 of the AF3 SI for details.

    Score: 0.8·ipTM + 0.2·pTM + 0.5·disorder - 100·has_clash
    """
    if f_input.is_batched:
        raise NotImplementedError(
            "full_complex_sample_ranking_metric does not support batched inputs"
        )
    diffusion_out = model_out["diffusion"]
    confidence_out = model_out["confidence"]

    sample_coords = diffusion_out["coordinates"][sample_index]  # [Natom, 3]
    plddt_logits = confidence_out["plddt_logits"][sample_index]  # [Natom, num_bins]
    pae_logits = confidence_out["pae_logits"][sample_index]  # [L, L, num_bins]
    pde_logits = confidence_out["pde_logits"][sample_index]  # [L, L, num_bins]
    plddt_bin_centers = confidence_out["plddt_bin_centers"]
    pae_bin_centers = confidence_out["pae_bin_centers"]
    pde_bin_centers = confidence_out["pde_bin_centers"]

    token_mask = f_input.token.pad_mask
    atom_mask = f_input.atom.pad_mask
    asym_id = f_input.token.asym_id
    frame_mask = f_input.token.frame_mask

    # Remove padding
    n_atoms: int = ref_struct.num_atoms
    n_tokens: int = ref_struct.num_residues
    if not token_mask[:n_tokens].all():
        raise ValueError("All tokens must be valid (token_mask must be all True).")
    if not atom_mask[:n_atoms].all():
        raise ValueError("All atoms must be valid (atom_mask must be all True).")

    sample_coords = sample_coords[:n_atoms, :]
    plddt_logits = plddt_logits[:n_atoms, :]
    pae_logits = pae_logits[:n_tokens, :n_tokens, :]
    pde_logits = pde_logits[:n_tokens, :n_tokens, :]
    asym_id = asym_id[:n_tokens]
    frame_mask = frame_mask[:n_tokens]

    # === Compute confidence scores ===
    plddt = compute_plddt(plddt_logits, plddt_bin_centers)
    pde = compute_pde(pde_logits, pde_bin_centers)
    pae = compute_pae(pae_logits, pae_bin_centers)
    ptm = compute_ptm(pae_logits, pae_bin_centers, frame_mask)
    iptm = compute_iptm(pae_logits, pae_bin_centers, asym_id, frame_mask)

    confidence_scores: dict[str, np.ndarray] = {
        "plddt": plddt.cpu().numpy(),
        "pde": pde.cpu().numpy(),
        "pae": pae.cpu().numpy(),
    }

    # === Compute summary metrics ===
    summary: dict = {}

    # Complex-level metrics
    # Atomize features for clash/disorder
    is_polymer = f_input.token.is_protein | f_input.token.is_rna | f_input.token.is_dna
    is_polymer = is_polymer[:n_tokens]
    # Convert token-level features to atom-level features.
    token_index = f_input.atom.token_index
    token_index = token_index[:n_atoms]
    is_polymer_atom = broadcast_tokens_to_atoms(is_polymer, token_index).bool()
    asym_id_atom = broadcast_tokens_to_atoms(asym_id, token_index).long()
    has_clash = compute_has_clash(
        sample_coords=sample_coords,
        asym_id=asym_id_atom,
        is_polymer=is_polymer_atom,
        atom_mask=atom_mask,
    )
    complex_scores = {
        "plddt": plddt.mean().item(),
        "pde": pde.mean().item(),
        "ptm": ptm.item(),
        "iptm": iptm.item(),
        "has_clash": float(has_clash.item()),
        # TODO: compute disorder ratio using RASA
        "disorder_ratio": 0.0,
    }
    # Compute the ranking score for each sample
    complex_scores["ranking_score"] = compute_full_complex_ranking(complex_scores)
    summary["complex"] = complex_scores

    # Chain-level metrics
    chain_scores: dict[str, dict] = {}
    m = ref_struct.metadata
    asym_id_to_name = {cm.asym_id: cm.name for cm in m.chains}
    for c in ref_struct.chains:
        aid = c.asym_id
        chain_atom_mask = asym_id_atom == aid
        chain_token_mask = asym_id == aid
        chain_frame_mask = frame_mask[chain_token_mask]

        key = asym_id_to_name[aid]
        chain_scores[key] = {
            "plddt": plddt[chain_atom_mask].mean().item(),
            "pde": pde[chain_token_mask][:, chain_token_mask].mean().item(),
            "ptm": compute_ptm(
                pae_logits[chain_token_mask][:, chain_token_mask],
                pae_bin_centers,
                chain_frame_mask,
            ).item(),
        }
    summary["chains"] = chain_scores

    # Interface-level metrics
    interface_scores: dict[str, dict] = {}
    for ai, aj in itertools.combinations(asym_id_atom.unique().tolist(), 2):
        iface_token_mask = (asym_id == ai) | (asym_id == aj)
        iface_frame_mask = frame_mask[iface_token_mask]
        if iface_frame_mask.sum() == 0:
            continue

        key = f"{asym_id_to_name[ai]}-{asym_id_to_name[aj]}"
        interface_scores[key] = {
            "iptm": compute_iptm(
                pae_logits[iface_token_mask][:, iface_token_mask],
                pae_bin_centers,
                asym_id[iface_token_mask],
                iface_frame_mask,
            ).item()
        }
    summary["interfaces"] = interface_scores

    return summary, confidence_scores


def compute_full_complex_ranking(
    confidence_scores: dict[str, float],
    w_ptm: float = 0.2,
    w_iptm: float = 0.8,
    w_disorder: float = 0.0,
    w_clash: float = 100.0,
) -> float:
    """
    AlphaFold3 sample ranking metric for the full complex.
    See Section 5.9.3 of the AF3 SI for details.

    Score: 0.8·ipTM + 0.2·pTM + 0.5·disorder - 100·has_clash
    """
    iptm = confidence_scores["iptm"]
    ptm = confidence_scores["ptm"]
    has_clash = confidence_scores["has_clash"]
    disorder = confidence_scores["disorder_ratio"]
    return w_iptm * iptm + w_ptm * ptm + w_disorder * disorder - w_clash * has_clash
