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

from . import (  # noqa: F401
    atom,
    bond,
    ccd,
    chain,
    residue,
    sequence,
    training,
)

NUM_RES_TYPES: int = len(residue.ResidueName)
NUM_ATOM_NAME_CHARS: int = 64  # AlphaFold3
NUM_ATOM_ELEMENTS: int = 128  # AlphaFold3

MAX_NUM_ATOMS_PER_TOKEN: int = 24

# See Section 2.7.3 of the AlphaFold3 paper
INTERFACE_CUTOFF: float = 15.0  # Angstroms

ChainType = chain.ChainType
SubChainType = chain.SubChainType
ResidueName = residue.ResidueName
AtomName = atom.AtomName
ConnectionType = bond.ConnectionType
