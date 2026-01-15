import enum

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


# For model training
# TODO: add modified and resolved
class LDDTType(enum.Enum):
    # interface modalities
    INTER_PROTEIN_PROTEIN = "inter_protein_protein"
    INTER_DNA_DNA = "inter_dna_dna"
    INTER_RNA_RNA = "inter_rna_rna"
    INTER_DNA_PROTEIN = "inter_dna_protein"
    INTER_RNA_PROTEIN = "inter_rna_protein"
    INTER_LIGAND_PROTEIN = "inter_ligand_protein"
    INTER_DNA_LIGAND = "inter_dna_ligand"
    INTER_RNA_LIGAND = "inter_rna_ligand"
    # intra-chain modalities
    INTRA_PROTEIN = "intra_protein"
    INTRA_DNA = "intra_dna"
    INTRA_RNA = "intra_rna"
    INTRA_LIGAND = "intra_ligand"


LDDTWeightsAF3 = {
    # interface modalities
    LDDTType.INTER_PROTEIN_PROTEIN: 20.0,
    LDDTType.INTER_DNA_DNA: 0.0,
    LDDTType.INTER_RNA_RNA: 0.0,
    LDDTType.INTER_DNA_PROTEIN: 10.0,
    LDDTType.INTER_RNA_PROTEIN: 10.0,
    LDDTType.INTER_LIGAND_PROTEIN: 10.0,
    LDDTType.INTER_DNA_LIGAND: 5.0,
    LDDTType.INTER_RNA_LIGAND: 5.0,
    # intra-chain modalities
    LDDTType.INTRA_PROTEIN: 20.0,
    LDDTType.INTRA_DNA: 4.0,
    LDDTType.INTRA_RNA: 16.0,
    LDDTType.INTRA_LIGAND: 20.0,
}
LDDTWeightsBoltz = {
    # interface modalities
    LDDTType.INTER_PROTEIN_PROTEIN: 20.0,
    LDDTType.INTER_DNA_DNA: 0.0,
    LDDTType.INTER_RNA_RNA: 0.0,
    LDDTType.INTER_DNA_PROTEIN: 5.0,
    LDDTType.INTER_RNA_PROTEIN: 5.0,
    LDDTType.INTER_LIGAND_PROTEIN: 20.0,
    LDDTType.INTER_DNA_LIGAND: 2.0,
    LDDTType.INTER_RNA_LIGAND: 2.0,
    # intra-chain modalities
    LDDTType.INTRA_PROTEIN: 20.0,
    LDDTType.INTRA_DNA: 2.0,
    LDDTType.INTRA_RNA: 8.0,
    LDDTType.INTRA_LIGAND: 20.0,
}
