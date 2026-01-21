"""Summarize structure prediction metrics."""

import logging
from typing import Any

import torch

from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure
from kfold.training.metrics.structure_metrics import compute_pair_lddt, compute_rmsd
from kfold.utils.geometry.rigid_align import rigid_align

logger = logging.getLogger(__name__)


def summarize_prediction(
    ref_struct: RefStructure,
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    align: bool = True,
) -> dict:
    """Get structure prediction metrics.

    Parameters
    ----------
    ref_struct : RefStructure
        Reference structure containing metadata
    pred_coords : torch.Tensor
        Predicted atom coordinates, Shape of [Natom, 3]
    true_coords : torch.Tensor
        Ground truth atom coordinates, Shape of [Natom, 3]
    atom_mask : torch.Tensor
        Mask for resolved atoms, Shape of [Natom]
    align : bool
        Whether to rigidly align true coordinates to predicted coordinates
        before metric computation.

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
    assert true_coords.shape == (num_atoms, 3), (
        f"True coordinates must have shape ({num_atoms}, 3), get {true_coords.shape}"
    )
    assert atom_mask.shape == (num_atoms,), (
        f"Atom mask must have shape ({num_atoms},), get {atom_mask.shape}"
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
        if c.ctype.is_polymer:
            entity_info[entity_id] = {
                "entity_id": c.entity_id,
                "type": c.ctype.name.lower(),
                "sequence": c.get_sequence(),
                "asym_ids": [c.asym_id],
            }
        else:
            entity_info[entity_id] = {
                "entity_id": c.entity_id,
                "type": c.ctype.name.lower(),
                "ccd": "-".join(c.get_ccd_sequence()),
                "asym_ids": [c.asym_id],
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
    for c in ref_struct.chains:
        n_atoms_chain = c.num_atoms
        atom_asym_ids[st : st + n_atoms_chain] = c.asym_id
        st += n_atoms_chain
    assert st == num_atoms, "Mismatch in number of atoms when preparing asym_id tensor."
    assert atom_asym_ids.min().item() >= 1, "Asym IDs should be positive integers."

    # Remove invalid atoms
    true_coords = true_coords[atom_mask]  # [Natom_resolved, 3]
    pred_coords = pred_coords[atom_mask]  # [Natom_resolved, 3]
    atom_asym_ids = atom_asym_ids[atom_mask]  # [Natom_resolved]
    del atom_mask

    if align:
        # Align true coordinates to predicted coordinates if needed
        true_coords = rigid_align(true_coords, pred_coords, mask=None)

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
        key = str(asym_id)
        cm = metadata.get_chain_by_asym_id(asym_id)
        ctype = cm.ctype.name.lower()
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
            if ctype in ("dna", "rna"):
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

        summary: dict[str, Any] = {
            "type": ctype,
            "name": cm.chain_name,
            "entity_id": cm.entity_id,
            "asym_id": cm.asym_id,
            "num_valid_atoms": num_chain_atoms,
            "metrics": metrics,
        }
        # Store chain summary
        chain_summaries[key] = summary

    # Compute interface-level metrics
    interface_summaries: dict[str, dict] = {}
    for aid1, aid2 in iface_asym_ids:
        assert aid1 != aid2, "Interface cannot be intra-chain."
        key = f"{aid1}:{aid2}"
        cm1 = metadata.get_chain_by_asym_id(aid1)
        cm2 = metadata.get_chain_by_asym_id(aid2)
        ctypes = (cm1.ctype.name.lower(), cm2.ctype.name.lower())

        # Check if interface atoms are present
        chain1_mask = atom_asym_ids == aid1  # [Natom_resolved]
        chain2_mask = atom_asym_ids == aid2  # [Natom_resolved]
        interface_mask = chain1_mask[:, None] & chain2_mask[None, :]

        if "dna" in ctypes or "rna" in ctypes:
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

        interface_summary = {
            "type_1": cm1.ctype.name.lower(),
            "type_2": cm2.ctype.name.lower(),
            "name_1": cm1.chain_name,
            "name_2": cm2.chain_name,
            "entity_id_1": cm1.entity_id,
            "entity_id_2": cm2.entity_id,
            "asym_id_1": cm1.asym_id,
            "asym_id_2": cm2.asym_id,
            "num_valid_atom_pairs": num_interface_pairs,
            "metrics": {
                "lddt": interface_lddt,
            },
        }
        interface_summaries[key] = interface_summary

    # TODO: add confidence metrics if available

    return {
        "id": metadata.id,
        "entity": entity_info,
        "metrics": complex_metrics,
        "chains": chain_summaries,
        "interfaces": interface_summaries,
    }
