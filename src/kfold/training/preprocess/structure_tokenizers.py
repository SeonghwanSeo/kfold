# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
