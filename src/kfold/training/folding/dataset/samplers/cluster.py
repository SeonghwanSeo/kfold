# Started from https://github.com/jwohlwend/boltz
from collections import defaultdict

import numpy as np

import kfold.constants as C
from kfold.data.metadata import ChainInfo, InterfaceInfo, Metadata
from kfold.utils.registry import DATA_SAMPLER

from .base import BaseSampler, Sample


# === Helpers to compute weights === #
def get_chain_cluster(chain: ChainInfo) -> str:
    """Get the cluster id for a chain.

    Parameters
    ----------
    chain : ChainInfo
        The chain id to get the cluster id for.

    Returns
    -------
    str
        The cluster id of the chain.
    """
    return chain.cluster_id


def get_interface_cluster(
    interface: InterfaceInfo, chain_dict: dict[int, ChainInfo]
) -> str:
    """Get the cluster id for an interface.

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the cluster id for.
    chain_dict : dict[int, ChainInfo]
        The dictionary of chains in the complex. {asym_id: ChainInfo}

    Returns
    -------
    str
        The cluster id of the interface.
    """
    assert len(interface.asym_ids) == 2, "Only support interfaces between two chains."
    asym_id1, asym_id2 = interface.asym_ids
    chain1 = chain_dict[asym_id1]
    chain2 = chain_dict[asym_id2]

    cluster_1 = chain1.cluster_id
    cluster_2 = chain2.cluster_id

    cluster_id = (cluster_1, cluster_2)
    cluster_id = tuple(sorted(cluster_id))

    return ":".join(cluster_id)


def get_chain_weight(
    chain: ChainInfo,
    cluster_sizes: dict[str, int],
    beta_chain: float = 0.5,
    alpha_prot: float = 3.0,
    alpha_nuc: float = 3.0,
    alpha_ligand: float = 1.0,
    ensure_protein: bool = False,
) -> float:
    """Get the weight of a chain.

    Parameters
    ----------
    chain : ChainInfo
        The chain to get the weight for.
    cluster_sizes : dict[str, int]
        The cluster sizes.
    beta_chain : float
        The beta value for chains.
    alpha_prot : float
        The alpha value for proteins.
    alpha_nuc : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.
    ensure_protein : bool
        Whether to ensure at least one protein chain in the interface.

    Returns
    -------
    float
        The weight of the chain.
    """
    n_prot, n_nuc, n_ligand = 0, 0, 0
    if chain.chain_type is C.chain.ChainType.Protein:
        n_prot += 1
    elif chain.chain_type in (C.chain.ChainType.DNA, C.chain.ChainType.RNA):
        n_nuc += 1
    else:
        n_ligand += 1

    if ensure_protein and n_prot == 0:
        return 0.0

    cluster_id = get_chain_cluster(chain)
    n_cluster = cluster_sizes[cluster_id]

    # See Section 2.5.1 Equation 1
    weight = (beta_chain / n_cluster) * (
        alpha_prot * n_prot + alpha_nuc * n_nuc + alpha_ligand * n_ligand
    )
    return weight


def get_interface_weight(
    interface: InterfaceInfo,
    chain_dict: dict[int, ChainInfo],
    cluster_sizes: dict[str, int],
    beta_interface: float = 1.0,
    alpha_prot: float = 3.0,
    alpha_nuc: float = 3.0,
    alpha_ligand: float = 1.0,
    ensure_protein: bool = False,
) -> float:
    """Get the weight of an interface.

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the weight for.
    chain_dict : dict[int, ChainInfo]
        The dictionary of chains in the complex. {asym_id: ChainInfo}
    cluster_sizes : dict[str, int]
        The cluster sizes.
    beta_interface : float
        The beta value for interfaces.
    alpha_prot : float
        The alpha value for proteins.
    alpha_nuc : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.
    ensure_protein : bool
        Whether to ensure at least one protein chain in the interface.

    Returns
    -------
    float
        The weight of the interface.

    """
    weight = 0.0
    n_prot, n_nuc, n_ligand = 0, 0, 0
    for asym_id in interface.asym_ids:
        chain = chain_dict[asym_id]
        if chain.chain_type is C.chain.ChainType.Protein:
            n_prot += 1
        elif chain.chain_type in (C.chain.ChainType.DNA, C.chain.ChainType.RNA):
            n_nuc += 1
        else:
            n_ligand += 1

    if ensure_protein and n_prot == 0:
        return 0.0

    cluster_id = get_interface_cluster(interface, chain_dict)
    n_cluster = cluster_sizes[cluster_id]

    # See Section 2.5.1 Equation 1
    weight = (beta_interface / n_cluster) * (
        alpha_prot * n_prot + alpha_nuc * n_nuc + alpha_ligand * n_ligand
    )
    return weight


@DATA_SAMPLER.register()
class ClusterSampler(BaseSampler):
    """The weighted sampling approach, as described in AF3.
    See section 2.5.1 Weighted PDB dataset in the AF3 paper.

    Each chain / interface is given a weight according
    to the following formula, and sampled accordingly:

    w ∝ (b / N_clust) * (a_prot * N_prot + a_nuc * N_nuc + a_ligand * N_ligand)

    NOTE: Compare to Boltz, I change the cluster size estimation.
    # Boltz: consider all valid chains / interfaces for each complex:
        e.g.) If there are 3 identical chains in a complex, chain counts is 3.
    # KFold: consider unique valid chains / interfaces for each complex:
        e.g.) If there are 3 identical chains in a complex, chain counts is 1.
    where
    """

    class Config:
        """Initialize the sampler.

        Parameters
        ----------
        alpha_prot : float, optional
            The alpha value for proteins.
        alpha_nuc : float, optional
            The alpha value for nucleic acids.
        alpha_ligand : float, optional
            The alpha value for ligands.
        beta_chain : float, optional
            The beta value for chains.
        beta_interface : float, optional
            The beta value for interfaces.
        allow_redundant : bool, optional
            Whether to allow redundant chains / interfaces when
            estimating cluster sizes. Default to True (Boltz behavior).
        ensure_protein : bool, optional
            Whether to ensure at least one protein chain in
            each sampled item. Default to False.
        """

        alpha_prot: float = 3.0
        alpha_nuc: float = 3.0
        alpha_ligand: float = 1.0
        beta_chain: float = 0.5
        beta_interface: float = 1.0
        allow_redundant: bool = True
        ensure_protein: bool = False

    def __init__(self, config: Config, records: list[Metadata]) -> None:
        self.config = config
        # weights
        self.alpha_prot = config.alpha_prot
        self.alpha_nuc = config.alpha_nuc
        self.alpha_ligand = config.alpha_ligand

        self.beta_chain = config.beta_chain
        self.beta_interface = config.beta_interface

        self.allow_redundant = config.allow_redundant
        self.ensure_protein = config.ensure_protein

    def setup(self, records: list[Metadata]):
        self.estimate_cluster_sizes(records)
        self.compute_sampling_weights(records)

    def estimate_cluster_sizes(self, records: list[Metadata]):
        # Estimate cluster sizes of chains and interfaces
        self.chain_cluster_sizes: dict[str, int] = defaultdict(int)
        self.interface_cluster_sizes: dict[str, int] = defaultdict(int)

        for record in records:
            chain_dict: dict[int, ChainInfo] = {
                chain.asym_id: chain for chain in record.chains
            }
            chain_clusters_in_record = [
                get_chain_cluster(chain) for chain in record.chains if chain.valid
            ]
            interface_clusters_in_record = [
                get_interface_cluster(interface, chain_dict)
                for interface in record.interfaces
                if interface.valid
            ]

            if not self.allow_redundant:
                # Remove redundant clusters in the record
                chain_clusters_in_record = set(chain_clusters_in_record)
                interface_clusters_in_record = set(interface_clusters_in_record)

            for cluster_id in chain_clusters_in_record:
                self.chain_cluster_sizes[cluster_id] += 1
            for cluster_id in interface_clusters_in_record:
                self.interface_cluster_sizes[cluster_id] += 1

    def compute_sampling_weights(self, records: list[Metadata]):
        """Compute sampling weights for chains and interfaces."""

        # Compute weights
        items: list[Sample] = []
        weights: list[float] = []

        for record in records:
            chain_dict: dict[int, ChainInfo] = {
                chain.asym_id: chain for chain in record.chains
            }
            for chain in record.chains:
                if not chain.valid:
                    continue
                weight = get_chain_weight(
                    chain,
                    self.chain_cluster_sizes,
                    self.beta_chain,
                    self.alpha_prot,
                    self.alpha_nuc,
                    self.alpha_ligand,
                    self.ensure_protein,
                )
                items.append(Sample(record, (chain.asym_id,)))
                weights.append(weight)

            for interface in record.interfaces:
                if not interface.valid:
                    continue
                weight = get_interface_weight(
                    interface,
                    chain_dict,
                    self.interface_cluster_sizes,
                    self.beta_interface,
                    self.alpha_prot,
                    self.alpha_nuc,
                    self.alpha_ligand,
                    self.ensure_protein,
                )
                items.append(Sample(record, interface.asym_ids))
                weights.append(weight)

        self.items = items
        self.weights = np.array(weights) / np.sum(weights)
