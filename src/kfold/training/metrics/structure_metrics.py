"""Model validation metrics for structure prediction tasks."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, TypeVar

import torch

import kfold.constants as C
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure
from kfold.training.dataset.utils.permutation import get_aligned_true_coords
from kfold.utils.geometry.rigid_align import rigid_align

logger = logging.getLogger(__name__)

LDDTWeights: dict[str, float] = {}
for k, v in C.training.LDDTWeights.items():
    if isinstance(k, tuple):
        key_name = f"interface/lddt-{k[0].name.lower()}_{k[1].name.lower()}"
    else:
        key_name = f"chain/lddt-{k.name.lower()}"
    LDDTWeights[key_name] = v

main_metric_names = [
    # complex-level metrics
    "complex/rmsd",
    "complex/lddt",
    # chain-level metrics
    "chain/lddt-protein",
    "chain/lddt-dna",
    "chain/lddt-rna",
    "chain/lddt-ligand",
    # interface-level metrics
    "interface/lddt-protein_protein",
    "interface/lddt-protein_dna",
    "interface/lddt-protein_rna",
    "interface/lddt-protein_ligand",
    "interface/lddt-dna_dna",
    "interface/lddt-dna_rna",
    "interface/lddt-dna_ligand",
    "interface/lddt-rna_rna",
    "interface/lddt-rna_ligand",
    "interface/lddt-ligand_ligand",
    # additional metrics for other subtypes
    "special/chain/lddt-peptide",
    "special/chain/lddt-small_molecule",
    "special/chain/lddt-covalent_ligand",
    "special/interface/lddt-protein_small_molecule",
    "special/interface/lddt-protein_peptide",
    "special/interface/lddt-protein_ion",
    "special/interface/lddt-protein_glycan",
]
monitor_metric_names = [
    "weighted_lddt",
    "top1_rmsd",
    "top5_rmsd",
    "top1_lddt",
    "top5_lddt",
]

_T = TypeVar("_T")


def norm_key(k1: _T, k2: _T) -> tuple[_T, _T]:
    """Return a normalized key tuple (k_min, k_max)."""
    return (k1, k2) if k1 <= k2 else (k2, k1)


# ============================================================
# Symmetry correction for accurate metric computation
# ============================================================
def get_aligned_structure(
    ref_struct: RefStructure,
    pred_coords: torch.Tensor,
    find_best_permutation: bool = True,
    symmetry_dict: dict | None = None,
) -> RefStructure:
    """Get the best matching true coordinates to the predicted coordinates.
    Chain permutation and atom swaps.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure containing ground truth coordinates and masks.
    pred_coords : torch.Tensor
        Predicted atom coordinates, Shape of [Natom, 3]
    find_best_permutation : bool, optional
        Whether to find the best permutation (default: True).
    symmetry_dict : dict (optional)
        The dictionary containing symmetry information:

    Returns
    -------
    ref_struct_aligned: RefStructure
        The reference structure with permuted ground truth coordinates.
    """
    return get_aligned_true_coords(
        ref_struct, pred_coords, find_best_permutation, symmetry_dict
    )  # [Nsample, Natom, 3]


# ============================================================
# Helper functions for metric computations
# ============================================================
def compute_rmsd(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    align: bool = False,
):
    """Compute RMSD between predicted and true coordinates.

    Parameters
    ----------
    a : torch.Tensor
        Coordinates 1, shape (*, Natom, 3)
    b : torch.Tensor
        Coordinates 2, shape (*, Natom, 3)
    mask : torch.Tensor | None
        Optional mask for valid atoms, shape (*, Natom)
    align : bool
        Whether to rigidly align a to b before RMSD computation
    """
    if align:
        a = rigid_align(a, b, mask)
    diff = (a - b).pow(2).sum(-1)  # (*, Natom)
    if mask is not None:
        assert mask.any(), "At least one atom must be valid in the mask."
        return (diff * mask).sum(-1).div(mask.sum(-1)).sqrt()
    else:
        return diff.mean(-1).sqrt()


def compute_pair_lddt(
    pdist_pred: torch.Tensor,
    pdist_true: torch.Tensor,
    thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
):
    """Compute the lddt score from predicted and true distances.

    Parameters
    ----------
    d_predicted : torch.Tensor
        Predicted distances, shape (*, Natom, Natom)
    d_true : torch.Tensor
        Ground truth distances, shape (*, Natom, Natom)
    thresholds : Sequence[float]
        Distance error thresholds for lddt calculation
    """
    dtype = pdist_pred.dtype
    error = torch.abs(pdist_true - pdist_pred)
    scores = torch.zeros_like(error, dtype=dtype)
    for threshold in thresholds:
        scores += (error < threshold).to(dtype)
    return scores / len(thresholds)


# ============================================================
# Metric computations
# ============================================================
def compute_validation_metric(
    ref_struct: RefStructure,
    pred_coords: torch.Tensor,
) -> dict:
    """Get structure prediction metrics.

    Parameters
    ----------
    ref_struct : RefStructure
        Reference structure containing metadata
    pred_coords : torch.Tensor
        Predicted atom coordinates, Shape of [Natom, 3]

    Returns
    -------
    metrics: dict
        The computed metrics

    Notes
    -----
    The computed metrics include:
    - Complex-level metrics:
        - RMSD
        - LDDT (15Å cutoff)
    - Chain-level metrics:
        - RMSD
        - Aligned RMSD (for chains with >4 atoms)
        - LDDT (15Å cutoff for proteins, 30Å cutoff for DNA/RNA)
    - Interface-level metrics:
        - LDDT (15Å cutoff for protein-protein, 30Å cutoff for DNA/RNA involved)
    """
    # Validate inputs
    num_atoms = ref_struct.num_atoms
    assert pred_coords.shape == (num_atoms, 3), (
        f"Predicted coordinates must have shape ({num_atoms}, 3), get {pred_coords.shape}"
    )

    # ============================================================
    # Prepare target information
    # ============================================================
    # Prepare entity information
    struct_id = ref_struct.id
    entity_info: dict[str, dict] = {}
    for c in ref_struct.chains:
        entity_id = str(c.entity_id)
        if entity_id in entity_info:
            entity_info[entity_id]["asym_ids"].append(c.asym_id)
            continue
        cm = ref_struct.metadata.get_chain_by_asym_id(c.asym_id)
        if c.ctype.is_polymer:
            entity_info[entity_id] = {
                "entity_id": c.entity_id,
                "type": c.ctype.name.lower(),
                "sequence": c.get_sequence(),
                "asym_ids": [c.asym_id],
                "is_low_homology": cm.is_low_homology,
            }
        else:
            entity_info[entity_id] = {
                "entity_id": c.entity_id,
                "type": c.ctype.name.lower(),
                "ccd": "-".join(c.get_ccd_sequence()),
                "asym_ids": [c.asym_id],
                "is_low_homology": cm.is_low_homology,
            }

    # Extract metadata (chains and interfaces)
    metadata: Metadata = ref_struct.metadata
    chain_asym_ids: list[int] = metadata.asym_ids
    iface_asym_ids: list[tuple[int, int]] = [im.asym_ids for im in metadata.interfaces]

    # ============================================================
    # Compute structure metrics
    # ============================================================
    # Prepare atom asym_id tensor for masking
    atom_asym_ids = torch.zeros(num_atoms, dtype=torch.int32, device=pred_coords.device)
    st = 0
    for c_i in range(metadata.num_chains):
        c = ref_struct.chains[c_i]
        asym_id = c.asym_id
        n_atoms_chain = c.num_atoms
        atom_asym_ids[st : st + n_atoms_chain] = asym_id
        st += n_atoms_chain
    assert st == num_atoms, "Mismatch in number of atoms when preparing asym_id tensor."
    assert atom_asym_ids.min().item() >= 1, "Asym IDs should be positive integers."

    # Get true coordinates
    dev = pred_coords.device
    true_coords = torch.from_numpy(ref_struct.get_atom_coords()).to(dev)
    atom_mask = torch.isfinite(true_coords).all(-1)  # [Natom]

    # Rigid-align
    true_coords = rigid_align(true_coords, pred_coords, atom_mask)  # [Natom, 3]

    # Remove unresolved atoms
    true_coords = true_coords[atom_mask]  # [Natom_resolved, 3]
    pred_coords = pred_coords[atom_mask]  # [Natom_resolved, 3]
    atom_asym_ids = atom_asym_ids[atom_mask]  # [Natom_resolved]
    del atom_mask

    # For lddt computations
    pdist_true = torch.norm(true_coords[:, None, :] - true_coords[None, :, :], dim=-1)
    pdist_pred = torch.norm(pred_coords[:, None, :] - pred_coords[None, :, :], dim=-1)
    lddt_score = compute_pair_lddt(pdist_pred, pdist_true)  # [natom, natom]
    # Use 30Å cutoff for DNA/RNA, 15Å cutoff for protein/ligand
    cutoff_mask_15 = pdist_true < 15.0
    cutoff_mask_30 = pdist_true < 30.0
    # Exclude self-pairs
    cutoff_mask_15.fill_diagonal_(False)
    cutoff_mask_30.fill_diagonal_(False)
    assert cutoff_mask_15.any(), "No valid atom pairs found for complex-level LDDT."

    # Compute complex-level metrics
    complex_metrics = {
        "rmsd": compute_rmsd(true_coords, pred_coords, mask=None, align=False).item(),
        "lddt": lddt_score[cutoff_mask_15].mean().item(),
    }

    # Compute chain-level metrics
    chain_summaries: dict[str, dict] = {}
    for asym_id in chain_asym_ids:
        c = ref_struct.get_chain_by_asym_id(asym_id)
        cm = metadata.get_chain_by_asym_id(asym_id)

        chain_mask = atom_asym_ids == asym_id  # [Natom_resolved]
        num_chain_atoms = chain_mask.sum().item()
        if num_chain_atoms == 0:
            logger.warning(f"No resolved atoms found for chain: {struct_id} {asym_id}")
            continue

        # Compute chain-level metrics
        metrics: dict[str, float] = {}

        # Compute RMSD and aligned RMSD (n_atoms > 4)
        chain_true_coords = true_coords[chain_mask]
        chain_pred_coords = pred_coords[chain_mask]
        metrics["rmsd"] = compute_rmsd(
            chain_true_coords, chain_pred_coords, align=False
        ).item()
        if num_chain_atoms > 4:
            metrics["rmsd_aligned"] = compute_rmsd(
                chain_true_coords, chain_pred_coords, align=True
            ).item()

        if num_chain_atoms > 1:
            # Compute LDDT
            intra_mask = chain_mask[:, None] & chain_mask[None, :]
            if c.is_nucleic_acid:
                # Use 30Å cutoff for DNA/RNA intra-chains
                cutoff_mask = cutoff_mask_30
            else:
                # Default to 15Å cutoff
                cutoff_mask = cutoff_mask_15
            lddt_mask = cutoff_mask & intra_mask
            if not lddt_mask.any():
                logger.warning(
                    f"No valid atom pairs found for chain: {struct_id} {asym_id}"
                )
                continue

            # Compute chain LDDT
            metrics["lddt"] = lddt_score[lddt_mask].mean().item()

        # Determine chain subtype for ligands
        summary: dict[str, Any] = {
            "name": cm.name,
            "type": c.ctype.name.lower(),
            "subtype": c.subtype.name.lower(),
            "entity_id": c.entity_id,
            "asym_id": c.asym_id,
            "is_low_homology": cm.is_low_homology,
            "num_valid_atoms": num_chain_atoms,
            "metrics": metrics,
        }
        # Store chain summary
        chain_summaries[cm.name] = summary

    # Compute interface-level metrics
    interface_summaries: dict[str, dict] = {}
    for aid1, aid2 in iface_asym_ids:
        assert aid1 != aid2, "Interface cannot be intra-chain."
        c1 = ref_struct.get_chain_by_asym_id(aid1)
        c2 = ref_struct.get_chain_by_asym_id(aid2)
        cm1 = metadata.get_chain_by_asym_id(aid1)
        cm2 = metadata.get_chain_by_asym_id(aid2)
        im = metadata.get_interface_by_asym_ids(aid1, aid2)

        # Check if interface atoms are present
        chain1_mask = atom_asym_ids == aid1  # [Natom_resolved]
        chain2_mask = atom_asym_ids == aid2  # [Natom_resolved]
        interface_mask = chain1_mask[:, None] & chain2_mask[None, :]

        if c1.is_nucleic_acid or c2.is_nucleic_acid:
            # Use 30Å cutoff for DNA/RNA involved interfaces
            cutoff_mask = cutoff_mask_30
        else:
            # Default to 15Å cutoff
            cutoff_mask = cutoff_mask_15
        lddt_mask = cutoff_mask & interface_mask
        num_interface_pairs = lddt_mask.sum().item()

        if num_interface_pairs == 0:
            continue

        # Compute interface LDDT
        interface_lddt = lddt_score[lddt_mask].mean().item()
        metrics = {"lddt": interface_lddt}

        interface_summary = {
            "name_1": cm1.name,
            "name_2": cm2.name,
            "type_1": c1.ctype.name.lower(),
            "type_2": c2.ctype.name.lower(),
            "subtype_1": c1.subtype.name.lower(),
            "subtype_2": c2.subtype.name.lower(),
            "entity_id_1": c1.entity_id,
            "entity_id_2": c2.entity_id,
            "asym_id_1": c1.asym_id,
            "asym_id_2": c2.asym_id,
            "is_low_homology": im.is_low_homology,
            "num_valid_atom_pairs": num_interface_pairs,
            "metrics": metrics,
        }
        interface_summaries[f"{cm1.name}:{cm2.name}"] = interface_summary

    # TODO: add confidence metrics if available

    return {
        "id": metadata.id,
        "entity": entity_info,
        "metrics": complex_metrics,
        "chains": chain_summaries,
        "interfaces": interface_summaries,
    }


def extract_validation_metrics(summary: dict[str, Any]) -> dict[str, float]:
    # Collect chain lddt metrics
    chain_metrics = defaultdict(list)
    special_chain_metrics = defaultdict(list)
    for v in summary["chains"].values():
        if not v["is_low_homology"]:
            continue
        c_metrics = v["metrics"]
        if "lddt" in c_metrics:
            lddt = c_metrics["lddt"]

            chain_metrics[f"lddt-{v['type']}"].append(lddt)
            # Additional metrics
            special_chain_metrics[f"lddt-{v['subtype']}"].append(lddt)

    # Collect interface lddt metrics
    iface_metrics = defaultdict(list)
    special_iface_metrics = defaultdict(list)
    for v in summary["interfaces"].values():
        if not v["is_low_homology"]:
            continue
        i_metrics = v["metrics"]
        lddt = i_metrics["lddt"]

        ctype1 = C.ChainType[v["type_1"].upper()]
        ctype2 = C.ChainType[v["type_2"].upper()]
        ctypes = norm_key(ctype1, ctype2)
        key = f"lddt-{ctypes[0].name.lower()}_{ctypes[1].name.lower()}"
        iface_metrics[key].append(lddt)

        # Additional metrics
        subtype1 = C.SubChainType[v["subtype_1"].upper()]
        subtype2 = C.SubChainType[v["subtype_2"].upper()]
        subtypes = norm_key(subtype1, subtype2)
        key = f"lddt-{subtypes[0].name.lower()}_{subtypes[1].name.lower()}"
        special_iface_metrics[key].append(lddt)

    # Aggregate chain-level and interface-level LDDTs
    extracted_metrics: dict[str, float] = {}
    for k, v in summary["metrics"].items():
        extracted_metrics[f"complex/{k}"] = v
    for k, vs in chain_metrics.items():
        extracted_metrics[f"chain/{k}"] = sum(vs) / len(vs)
    for k, vs in iface_metrics.items():
        extracted_metrics[f"interface/{k}"] = sum(vs) / len(vs)
    for k, vs in special_chain_metrics.items():
        extracted_metrics[f"special/chain/{k}"] = sum(vs) / len(vs)
    for k, vs in special_iface_metrics.items():
        extracted_metrics[f"special/interface/{k}"] = sum(vs) / len(vs)
    return extracted_metrics


def aggregate_validation_metrics(
    sample_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate validation metrics across multiple samples."""
    # All values
    all_metrics: dict[str, list[float]] = {}
    for sample in sample_summaries:
        # Collect complex-level metrics
        sample_metrics = extract_validation_metrics(sample)
        for k, v in sample_metrics.items():
            all_metrics.setdefault(k, []).append(v)

    # Aggregate metrics across samples
    # Average
    avg_metrics = {k: sum(vs) / len(vs) for k, vs in all_metrics.items()}

    # Best complex (by complex LDDT)
    # TODO: change ranking criterion from LDDT to global PDE according to AF3
    top1_idx = max(
        range(len(sample_summaries)),
        key=lambda i: sample_summaries[i]["metrics"]["lddt"],
    )
    top1_metrics = {k: vs[top1_idx] for k, vs in all_metrics.items()}

    # Top of five
    top5_metrics = {}
    for k, vs in all_metrics.items():
        if "rmsd" in k:
            # For RMSD, lower is better
            top5_metrics[k] = min(vs)
        else:
            # For LDDT, higher is better
            top5_metrics[k] = max(vs)

    # Compute weighted LDDT monitor value
    aggr_metrics = {
        k: (top1_metrics[k] + top5_metrics[k]) / 2.0 for k in all_metrics.keys()
    }
    monitor_values: list[float] = []
    monitor_weights: list[float] = []
    for k, w in LDDTWeights.items():
        if k in aggr_metrics:
            monitor_values.append(aggr_metrics[k])
            monitor_weights.append(w)
    weighted_lddt = sum(
        v * w for v, w in zip(monitor_values, monitor_weights, strict=True)
    ) / sum(monitor_weights)
    monitor_metrics = {
        "weighted_lddt": weighted_lddt,
        "top1_rmsd": top1_metrics["complex/rmsd"],
        "top5_rmsd": top5_metrics["complex/rmsd"],
        "top1_lddt": top1_metrics["complex/lddt"],
        "top5_lddt": top5_metrics["complex/lddt"],
    }
    return {
        "avg": avg_metrics,
        "top1": top1_metrics,
        "top5": top5_metrics,
        "monitor": monitor_metrics,
    }
