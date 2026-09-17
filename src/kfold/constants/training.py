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

from .chain import ChainType

Protein = ChainType.PROTEIN
DNA = ChainType.DNA
RNA = ChainType.RNA
Ligand = ChainType.LIGAND

# For mmCIF parsing
# See mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_exptl.method.html
CRYSTALLIZATION_METHODS = {
    "ELECTRON CRYSTALLOGRAPHY",
    "FIBER DIFFRACTION",
    "NEUTRON DIFFRACTION",
    "POWDER DIFFRACTION",
    "X-RAY DIFFRACTION",
}
NMR_METHODS = {
    "SOLUTION NMR",
    "SOLID-STATE NMR",
}
EM_METHODS = {
    "ELECTRON MICROSCOPY",
}
OTHER_METHODS = {
    "FLUORESCENCE TRANSFER",
    "INFRARED SPECTROSCOPY",
    "SOLUTION SCATTERING",
}
ALL_EXPERIMENT_METHODS = (
    CRYSTALLIZATION_METHODS | NMR_METHODS | EM_METHODS | OTHER_METHODS
)


LDDTWeightsAF3: dict[ChainType | tuple[ChainType, ChainType], float] = {
    # intra-chain modalities
    Protein: 20.0,
    DNA: 4.0,
    RNA: 16.0,
    Ligand: 20.0,
    # interface modalities
    (Protein, Protein): 20.0,
    (Protein, DNA): 10.0,
    (Protein, RNA): 10.0,
    (Protein, Ligand): 10.0,
    (DNA, DNA): 0.0,
    (DNA, RNA): 0.0,
    (DNA, Ligand): 5.0,
    (RNA, RNA): 0.0,
    (RNA, Ligand): 5.0,
    (Ligand, Ligand): 0.0,
}
assert all(list(k) == sorted(k) for k in LDDTWeightsAF3 if isinstance(k, tuple)), (
    "LDDTWeightsAF3 keys must be ordered tuples"
)

LDDTWeights: dict[ChainType | tuple[ChainType, ChainType], float] = {
    # intra-chain modalities
    Protein: 20.0,
    DNA: 4.0,
    RNA: 8.0,  # adjusted from AF3
    Ligand: 20.0,
    # interface modalities
    (Protein, Protein): 20.0,
    (Protein, DNA): 10.0,
    (Protein, RNA): 10.0,
    (Protein, Ligand): 10.0,
    (DNA, DNA): 0.0,
    (DNA, RNA): 0.0,
    (DNA, Ligand): 5.0,
    (RNA, RNA): 0.0,
    (RNA, Ligand): 5.0,
    (Ligand, Ligand): 0.0,
}
assert all(list(k) == sorted(k) for k in LDDTWeights if isinstance(k, tuple)), (
    "LDDTWeights keys must be ordered tuples"
)
