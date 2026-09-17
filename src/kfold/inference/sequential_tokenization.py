"""Pre-forward apo structure tokenization."""

from typing import Any

import torch

from kfold.data.types.model_input import FoldingInput


@torch.inference_mode()
def apply_apo_structure_tokens(
    f_input: FoldingInput,
    struct_token_records: list[list[dict]],
    structure_encoder: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Tokenize apo structures and return structure-only sequence/position IDs."""
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

    structure_groups: dict[int, int] = {}
    structure_pos_id = pos_ids.clone()
    position_assigned = torch.zeros_like(pos_ids, dtype=torch.bool)
    for records in struct_token_records:
        for record in records:
            group = record.get("structure_group", [])
            if len(group) < 2:
                continue
            group_id = min(group)
            for asym_id in group:
                previous = structure_groups.setdefault(asym_id, group_id)
                if previous != group_id:
                    raise ValueError("Overlapping protein structure groups")
            residue_index = record.get("residue_index")
            segments = record.get("segments")
            if residue_index is None or segments is None:
                raise ValueError(
                    "Multi-chain structure records require residue indices and segments"
                )
            residue_index = torch.as_tensor(
                residue_index, dtype=pos_ids.dtype, device=pos_ids.device
            )
            if residue_index.shape != (len(record["seq"]),):
                raise ValueError("Structure position/source lengths differ")
            if bool((residue_index < 0).any()):
                raise ValueError("Structure position IDs must be non-negative")
            for asym_id, offset_start, offset_end, source_start, source_end in segments:
                chain_start = int(torch.where(asym_ids == asym_id)[0][0])
                target_start = chain_start + 1 + offset_start
                target_end = chain_start + 1 + offset_end
                target_slice = slice(target_start, target_end)
                values = residue_index[source_start:source_end]
                if target_end - target_start != len(values):
                    raise ValueError("Structure position source/target lengths differ")
                assigned = position_assigned[target_slice]
                if bool(assigned.any()) and not torch.equal(
                    structure_pos_id[target_slice][assigned], values[assigned]
                ):
                    raise ValueError("Inconsistent multi-apo structure position IDs")
                structure_pos_id[target_slice] = values
                position_assigned[target_slice] = True
    if not structure_groups:
        return None, None

    structure_seq_id = asym_ids.clone()
    for asym_id, group_id in structure_groups.items():
        structure_seq_id[asym_ids == asym_id] = group_id
    if f_input.is_batched:
        return structure_seq_id.unsqueeze(0), structure_pos_id.unsqueeze(0)
    return structure_seq_id, structure_pos_id
