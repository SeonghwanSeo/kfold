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
