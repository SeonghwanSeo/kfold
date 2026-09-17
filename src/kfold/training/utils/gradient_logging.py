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

import torch


@torch.no_grad()
def gradient_norm(module: torch.nn.Module) -> float:
    # Only compute over parameters that are being trained
    grads = [
        p.grad for p in module.parameters() if p.requires_grad and p.grad is not None
    ]
    if not grads:
        return 0.0
    norms = torch._foreach_norm(grads, ord=2)
    total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    return total_norm.item()


@torch.no_grad()
def parameter_norm(module: torch.nn.Module) -> float:
    # Get all parameters that are being trained
    parameters = [p for p in module.parameters() if p.requires_grad]
    if not parameters:
        return 0.0
    norms = torch._foreach_norm(parameters, ord=2)
    total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    return total_norm.item()
