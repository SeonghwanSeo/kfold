"""Pre-forward apo structure tokenization."""

from typing import Any

import torch

from kfold.data.types.model_input import FoldingInput


@torch.inference_mode()
def apply_apo_structure_tokens(
    f_input: FoldingInput,
    struct_token_records: list[list[dict]],
    structure_encoder: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Tokenize apo structures and return attention-group, position, and chain IDs."""
    if f_input.is_batched:
        if f_input.batch_size != 1:
            raise NotImplementedError(
                "Apo structure tokenization only supports batch size 1."
            )
        bb_token_ids = f_input.sequence.bb_struct_token_id[0]
        fa_token_ids = f_input.sequence.fa_struct_token_id[0]
        asym_ids = f_input.sequence.asym_id[0]
        pos_ids = f_input.sequence.pos_id[0]
    else:
        bb_token_ids = f_input.sequence.bb_struct_token_id
        fa_token_ids = f_input.sequence.fa_struct_token_id
        asym_ids = f_input.sequence.asym_id
        pos_ids = f_input.sequence.pos_id

    if bb_token_ids.ndim != 2:
        raise ValueError(
            "Expected apo structure-token IDs with shape [L, Napo], "
            f"got {tuple(bb_token_ids.shape)}."
        )
    assert fa_token_ids.shape == bb_token_ids.shape

    for records in struct_token_records:
        if len(records) > bb_token_ids.shape[1]:
            raise ValueError(
                "Structure-token input count exceeds the apo axis: "
                f"{len(records)} > {bb_token_ids.shape[1]}."
            )
        for apo_i, record in enumerate(records):
            residue_index = record.get("residue_index")
            if residue_index is None:
                tokens = structure_encoder.tokenize(record["seq"], record["coords"])
            else:
                tokens = structure_encoder.tokenize(
                    record["seq"],
                    record["coords"],
                    residue_index=residue_index,
                )
            source_bb = tokens["bb_token_id"]
            source_fa = tokens["fa_token_id"]
            if record.get("mask_missing_structure", False):
                coords = torch.as_tensor(record["coords"], device=source_bb.device)
                backbone_valid = coords[:, :3].isfinite().all(dim=(-1, -2))
                # Keep sequence/topology; only suppress absent structural evidence.
                source_bb = source_bb.masked_fill(~backbone_valid, -1)
                source_fa = source_fa.masked_fill(~backbone_valid, -1)
            segments = record.get("segments")
            if segments is None:
                segments = [
                    (asym_id, offset_start, offset_end, 0, len(source_bb))
                    for asym_id, offset_start, offset_end in record["targets"]
                ]
            if len(segments) != len(record["targets"]):
                raise ValueError("Structure-token segment/target count differs")
            for asym_id, offset_start, offset_end, source_start, source_end in segments:
                chain_start = int(torch.where(asym_ids == asym_id)[0][0])
                target_start = chain_start + 1 + offset_start
                target_end = chain_start + 1 + offset_end
                target_slice = slice(target_start, target_end)
                source_slice = slice(source_start, source_end)
                if target_end - target_start != source_end - source_start:
                    raise ValueError("Structure-token source/target lengths differ")
                bb_token_ids[target_slice, apo_i] = source_bb[source_slice]
                fa_token_ids[target_slice, apo_i] = source_fa[source_slice]

    structure_groups: dict[int, tuple[int, int]] = {}
    structure_pos_id = pos_ids.clone()
    structure_chain_id = torch.zeros_like(asym_ids)
    for records in struct_token_records:
        for record in records:
            group = record.get("structure_group", [])
            if len(group) < 2:
                continue
            if len(group) > 100 or len(set(group)) != len(group):
                raise ValueError("Invalid TriProRep structure chain group")
            group_id = min(group)
            for chain_id, asym_id in enumerate(group):
                assignment = (group_id, chain_id)
                previous = structure_groups.setdefault(asym_id, assignment)
                if previous != assignment:
                    raise ValueError(
                        "Overlapping or inconsistently ordered structure groups"
                    )
            # The public complex dataset resets positions per chain. These records
            # contain one complete chain and must not reuse the old gap convention.
            if "residue_index" in record or "segments" in record:
                raise ValueError(
                    "Multi-chain encoding requires separate chain token records"
                )
            for asym_id, offset_start, offset_end in record["targets"]:
                if asym_id not in group or offset_start != 0:
                    raise ValueError(
                        "Multi-chain encoding requires complete grouped chains"
                    )
                chain_start = int(torch.where(asym_ids == asym_id)[0][0])
                start, end = chain_start + 1, chain_start + 1 + offset_end
                if end - start != len(record["seq"]):
                    raise ValueError("Structure position source/target lengths differ")
                structure_pos_id[start:end] = torch.arange(
                    len(record["seq"]), dtype=pos_ids.dtype, device=pos_ids.device
                )
    if not structure_groups:
        return None, None, None

    structure_seq_id = asym_ids.clone()
    for asym_id, (group_id, chain_id) in structure_groups.items():
        chain_mask = asym_ids == asym_id
        structure_seq_id[chain_mask] = group_id
        structure_chain_id[chain_mask] = chain_id
    if f_input.is_batched:
        return tuple(
            ids.unsqueeze(0)
            for ids in (structure_seq_id, structure_pos_id, structure_chain_id)
        )
    return structure_seq_id, structure_pos_id, structure_chain_id
