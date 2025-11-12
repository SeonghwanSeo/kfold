# Started from https://github.com/jwohlwend/boltz
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import kfold.constants as C
from kfold.data.metadata import ChainInfo, InterfaceInfo, Metadata
from kfold.utils.registry import DATA_SAMPLER, BaseConfig

from .base import BaseSampler, Sample

# FIXME: (SeonghwanSeo) Currently, I restrict that only protein and ligand
# are considered. Need to remove this restriction in the future.


# === Helpers to compute weights === #
def get_chain_cluster(chain: ChainInfo) -> str:
    """Get the cluster ID of a chain."""
    return chain.cluster_id


def get_interface_cluster(
    interface: InterfaceInfo, chain_dict: dict[int, ChainInfo]
) -> str:
    """Get the cluster ID of an interface."""
    chains = [chain_dict[asym_id] for asym_id in interface.asym_ids]
    cluster_ids = [get_chain_cluster(chain) for chain in chains]
    return ":".join(sorted(cluster_ids))


def get_chain_weight(
    chain: ChainInfo,
    cluster_sizes: dict[str, int],
    beta_chain: float = 0.5,
    alpha_prot: float = 3.0,
    alpha_nuc: float = 3.0,
    alpha_ligand: float = 1.0,
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

    # FIXME: remove following lines.
    if n_nuc > 0:
        return 0

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

    # FIXME: remove following lines.
    if n_nuc > 0:
        return 0

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

    Equation 1 in Section 2.5.1 of AF3 paper:
    w ∝ (β_r / N_clust) * (α_prot * n_prot + α_nuc * n_nuc + α_ligand * n_ligand),
    where β_r is the beta value for the chain / interface.

    NOTE: (SeonghwanSeo) Compare to Boltz, I changed a logic of cluster size estimation.

    In Boltz, all valid chains / interfaces in the record are considered when
    estimating the cluster sizes.
    However, I introduced `allow_redundant` parameter to control whether to allow
    redundant chains / interfaces when estimating cluster sizes.

    e.g.)
    If a record has 3 chains, all of which belong to the same cluster,
    Boltz will estimate the cluster size as 3, while this will estimate it as 1
    if `allow_redundant` is False.
    """

    @dataclass
    class Config(BaseConfig):
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
        """

        alpha_prot: float = 3.0
        alpha_nuc: float = 3.0
        alpha_ligand: float = 1.0
        beta_chain: float = 0.5
        beta_interface: float = 1.0
        allow_redundant: bool = True

    def __init__(self, config: Config, records: list[Metadata]) -> None:
        self.config = config
        # weights
        self.alpha_prot = config.alpha_prot
        self.alpha_nuc = config.alpha_nuc
        self.alpha_ligand = config.alpha_ligand

        self.beta_chain = config.beta_chain
        self.beta_interface = config.beta_interface

        self.allow_redundant = config.allow_redundant

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
                )
                items.append(Sample(record, interface.asym_ids))
                weights.append(weight)

        self.items = items
        self.weights = np.array(weights) / np.sum(weights)
