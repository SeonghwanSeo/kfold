"""Pre-forward apo structure tokenization."""

from typing import Any

import torch

from kfold.data.types.model_input import FoldingInput


@torch.inference_mode()
def apply_apo_structure_tokens(
    f_input: FoldingInput,
    struct_token_inputs: dict[int, dict],
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
    else:
        bb_token_ids = f_input.sequence.bb_struct_token_id
        fa_token_ids = f_input.sequence.fa_struct_token_id

    for token_input in struct_token_inputs.values():
        tokens = structure_encoder.tokenize(token_input["seq"], token_input["coords"])
        source_bb = tokens["bb_token_id"]
        source_fa = tokens["fa_token_id"]
        for target_start, target_end, source_start, source_end in token_input["mappings"]:
            target_slice = slice(target_start, target_end)
            source_slice = slice(source_start, source_end)
            bb_token_ids[target_slice] = source_bb[source_slice]
            fa_token_ids[target_slice] = source_fa[source_slice]
