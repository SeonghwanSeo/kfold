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

"""Weight-layout helpers for per-head concatenated projections.

Interleaving Q/K/V (or K/V) per head into one contiguous weight lets a kernel
issue a single matmul + single weight load instead of N separate ones, and
keeps each head's slice contiguous for L2 locality.
"""

import torch


def interleave_qkv(
    WQ: torch.Tensor, WK: torch.Tensor, WV: torch.Tensor, H: int, D: int
) -> torch.Tensor:
    """Concat WQ/WK/WV (each (N, H*D)) → (N, 3*H*D), per-head block-interleaved:
    columns [3*h*D : 3*(h+1)*D] hold [Q_h | K_h | V_h].  (matmul convention: x @ W)
    """
    N = WQ.shape[0]
    assert WQ.shape == WK.shape == WV.shape == (N, H * D)
    WQ_r, WK_r, WV_r = WQ.view(N, H, D), WK.view(N, H, D), WV.view(N, H, D)
    W = torch.stack([WQ_r, WK_r, WV_r], dim=2)  # (N, H, 3, D)
    return W.contiguous().view(N, 3 * H * D)


def interleave_kv(WK: torch.Tensor, WV: torch.Tensor, H: int, D: int) -> torch.Tensor:
    """Concat WK/WV (each (C_in, H*D)) → (C_in, 2*H*D), FEATURE-interleaved per head:
    per-head 2*D cols = [K[0], V[0], K[1], V[1], ...] so a (·, D, 2) reshape +
    `tl.split` recovers K, V.  (matmul convention: x @ W)
    """
    C_in = WK.shape[0]
    assert WK.shape == WV.shape == (C_in, H * D)
    WK_hv, WV_hv = WK.view(C_in, H, D), WV.view(C_in, H, D)
    W = torch.stack([WK_hv, WV_hv], dim=-1)  # (C_in, H, D, 2) — K=0, V=1 in last
    return W.contiguous().view(C_in, 2 * H * D)
