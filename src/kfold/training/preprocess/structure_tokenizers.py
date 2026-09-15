"""Load the two structure tokenizers used by offline apo preprocessing."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from kfold.model.layers.struct_enc import BackboneTokenizer, FullAtomTokenizer
from kfold.model.modules.prot_struct_encoder import (
    HF_BB_TOKENIZER_FILENAME,
    HF_FA_TOKENIZER_FILENAME,
    HF_REPO_ID,
)


def load_tokenizers(
    cache_dir: Path | None = None,
    device: torch.device | str = "cuda",
) -> tuple[BackboneTokenizer, FullAtomTokenizer]:
    """Load standalone backbone/full-atom weights from Hugging Face."""
    tokenizers = []
    for tokenizer_cls, filename in (
        (BackboneTokenizer, HF_BB_TOKENIZER_FILENAME),
        (FullAtomTokenizer, HF_FA_TOKENIZER_FILENAME),
    ):
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename, cache_dir=cache_dir)
        state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        tokenizer = tokenizer_cls().to(torch.bfloat16)
        tokenizer.load_state_dict(state_dict, strict=True)
        del state_dict
        tokenizers.append(tokenizer.eval().to(device))
    return tokenizers[0], tokenizers[1]
