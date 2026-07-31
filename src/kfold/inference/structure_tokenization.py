"""Pre-forward apo structure tokenization."""

from typing import Any

import torch

from kfold.data.types.model_input import FoldingInput


@torch.inference_mode()
def apply_apo_structure_tokens(
    f_input: FoldingInput,
    struct_token_records: list[list[dict]],
    structure_encoder: Any,
) -> None:
    """Tokenize raw apo structures and update ``f_input`` in place."""
    if f_input.is_batched:
        if f_input.batch_size != 1:
            raise NotImplementedError(
                "Apo structure tokenization only supports batch size 1."
            )
        bb_token_ids = f_input.sequence.bb_struct_token_id[0]
        fa_token_ids = f_input.sequence.fa_struct_token_id[0]
        asym_ids = f_input.sequence.asym_id[0]
    else:
        bb_token_ids = f_input.sequence.bb_struct_token_id
        fa_token_ids = f_input.sequence.fa_struct_token_id
        asym_ids = f_input.sequence.asym_id

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
            tokens = structure_encoder.tokenize(record["seq"], record["coords"])
            source_bb = tokens["bb_token_id"]
            source_fa = tokens["fa_token_id"]
            for asym_id, offset_start, offset_end in record["targets"]:
                chain_start = int(torch.where(asym_ids == asym_id)[0][0])
                target_start = chain_start + 1 + offset_start
                target_end = chain_start + 1 + offset_end
                target_slice = slice(target_start, target_end)
                bb_token_ids[target_slice, apo_i] = source_bb
                fa_token_ids[target_slice, apo_i] = source_fa
