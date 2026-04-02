# Copyright 2026 KAIST
# Copyright 2025 AlQuraishi Laboratory
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

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.utils.checkpoint

checkpoint_fn = torch.utils.checkpoint.checkpoint

BLOCK_ARG = Any
BLOCK_ARGS = Sequence[BLOCK_ARG]
LAYER_ARG = Any
LAYER_ARGS = dict[str, Sequence[LAYER_ARG]]


def wrap(a: BLOCK_ARG) -> BLOCK_ARGS:
    return (a,) if type(a) is not tuple else a


@torch.jit.ignore
def checkpoint_blocks(
    blocks: list[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: int | None,
    layer_args: LAYER_ARGS | None = None,
    use_reentrant: bool = False,
) -> BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Parameters
    ----------
    blocks: list[Callable]
        List of blocks to execute sequentially.
    args: BLOCK_ARGS
        Tuple of arguments for the first block.
    blocks_per_ckpt: int | None
        Size of each chunk. A higher value corresponds to fewer
        checkpoints, and trades memory for speed. If None, no checkpointing
        is performed.
    layer_args: LAYER_ARGS | None
        Optional dictionary mapping argument names to lists of arguments for
        each block. If provided, the arguments for block i will be passed as
        keyword arguments to block i. If not provided, no keyword arguments
        will be passed to the blocks.
    use_reentrant: bool
        Whether to use reentrant checkpointing.

    Returns:
        The output of the final block
    """
    # Add layer arguments if not provided
    layer_args = layer_args or {}
    for k, v in layer_args.items():
        if len(v) != len(blocks):
            raise ValueError(
                f"Length of layer arguments for {k} must match number of blocks"
            )

    # Add block indices
    blocks: list[tuple[int, Callable]] = list(enumerate(blocks))

    def exec(b: list[tuple[int, Callable]], a: BLOCK_ARGS) -> BLOCK_ARGS:
        for i, _b in b:
            la = {k: v[i] for k, v in layer_args.items()}
            a = wrap(_b(*a, **la))
        return a

    if not torch.is_grad_enabled() or blocks_per_ckpt is None:
        return exec(blocks, args)

    if blocks_per_ckpt < 1:
        raise ValueError("blocks_per_ckpt must be at least 1")

    def chunker(s, e):
        return lambda *a: exec(blocks[s:e], a)

    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        args = checkpoint_fn(chunker(s, e), *args, use_reentrant=use_reentrant)
        args = wrap(args)

    return args
