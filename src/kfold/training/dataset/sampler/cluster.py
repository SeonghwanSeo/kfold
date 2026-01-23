from collections import defaultdict

import numpy as np

from kfold.data.types.metadata import ChainInfo, InterfaceInfo, Metadata
from kfold.utils.registry import DATA_SAMPLER

from .base import BaseSampler, Sample


# === Helpers to compute weights === #
def get_chain_cluster_id(chain_m: ChainInfo) -> str:
    """Get the cluster ID of a chain."""
    assert chain_m.cluster_id is not None
    return chain_m.cluster_id


def get_interface_cluster_id(iface_m: InterfaceInfo) -> str:
    """Get the cluster ID of an interface."""
    assert iface_m.cluster_id is not None
    return iface_m.cluster_id


@DATA_SAMPLER.register()
class ClusterSampler(BaseSampler):
    """The weighted sampling approach, as described in AF3.

    See section 2.5.1 Weighted PDB dataset in the AF3 paper.

    Each chain / interface is given a weight according
    to the following formula, and sampled accordingly:

    Equation 1 in Section 2.5.1 of AF3 paper:
    w ∝ (β_r / N_clust) * (α_prot * n_prot + α_nuc * n_nuc + α_ligand * n_ligand),
    where β_r is the beta value for the chain / interface.

    NOTE: (SeonghwanSeo) Compare to Boltz1/AlphaFold3, I changed a logic of cluster
    size estimation. (`allow_redundant=False`) This is because Boltz1's data processing
    pipeline includes all chains in a complex while AF3's pipeline crops the complex up
    to 20 chains, which may lead to overestimation of cluster sizes and underestimation
    the chains/interfaces of small complexes. When `allow_redundant` is False, the
    cluster size is estimated by counting unique clusters in a complex, rather than
    counting all chains/interfaces. This way, small complexes are less penalized during
    sampling. For example, consider a complex with 3 chains, all belonging to the same
    cluster.
    e.g.)
    If a metadata has 3 chains, all of which belong to the same cluster,
    Boltz will estimate the cluster size as 3, while this will estimate it as 1
    if `allow_redundant` is False.
    """

    class Config(BaseSampler.Config):
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

    def __init__(self, config: Config) -> None:
        self.config = config
        self.is_initialized = False

        # weights
        self.alpha_prot = config.alpha_prot
        self.alpha_nuc = config.alpha_nuc
        self.alpha_ligand = config.alpha_ligand

        self.beta_chain = config.beta_chain
        self.beta_interface = config.beta_interface

        self.allow_redundant = config.allow_redundant

        # Cluster sizes
        self.chain_cluster_sizes: dict[str, int] = defaultdict(int)
        self.interface_cluster_sizes: dict[str, int] = defaultdict(int)
        self.num_clusters_in_complex: dict[str, dict[str, int]] = {}

    def get_samples(self, metadatas: list[Metadata]) -> tuple[list[Sample], np.ndarray]:
        assert self.is_initialized is False, "ClusterSampler can be used only once."
        self.is_initialized = True

        # Estimate cluster sizes
        self.estimate_cluster_sizes(metadatas)

        # Get samples and its weights
        samples: list[Sample] = []
        weights: list[float] = []

        for m in metadatas:
            chain_dict: dict[int, ChainInfo] = {
                chain.asym_id: chain for chain in m.chains
            }
            num_clusters_in_complex = self.num_clusters_in_complex[m.id]
            for chain in m.chains:
                weight = self._get_chain_weight(chain)
                if not self.allow_redundant:
                    # Adjust weight by number of clusters in the metadata
                    weight /= num_clusters_in_complex[get_chain_cluster_id(chain)]
                samples.append(Sample(m, chain.asym_id))
                weights.append(weight)

            for interface in m.interfaces:
                weight = self._get_interface_weight(interface, chain_dict)
                if not self.allow_redundant:
                    # Adjust weight by number of clusters in the metadata
                    weight /= num_clusters_in_complex[get_interface_cluster_id(interface)]
                samples.append(Sample(m, interface.asym_ids))
                weights.append(weight)

        # Normalize weights
        weights_arr = np.array(weights) / np.sum(weights)
        return samples, weights_arr

    def estimate_cluster_sizes(self, metadatas: list[Metadata]):
        """Estimate cluster sizes of chains and interfaces"""
        for m in metadatas:
            chain_clusters_in_entry: list[str] = [
                get_chain_cluster_id(chain) for chain in m.chains
            ]
            interface_clusters_in_entry: list[str] = [
                get_interface_cluster_id(interface) for interface in m.interfaces
            ]

            if not self.allow_redundant:
                # Store number of each cluster for each entry
                num_clusters = defaultdict(int)
                for cluster_id in chain_clusters_in_entry:
                    num_clusters[cluster_id] += 1
                for cluster_id in interface_clusters_in_entry:
                    num_clusters[cluster_id] += 1
                self.num_clusters_in_complex[m.id] = dict(num_clusters)

                # Remove redundant clusters in the metadata
                chain_clusters_in_entry = list(set(chain_clusters_in_entry))
                interface_clusters_in_entry = list(set(interface_clusters_in_entry))

            for cluster_id in chain_clusters_in_entry:
                self.chain_cluster_sizes[cluster_id] += 1
            for cluster_id in interface_clusters_in_entry:
                self.interface_cluster_sizes[cluster_id] += 1

    def _get_chain_weight(self, chain_m: ChainInfo) -> float:
        """Get the weight of a chain.

        Parameters
        ----------
        chain_m : ChainInfo
            The chain to get the weight for.

        Returns
        -------
        float
            The weight of the chain.
        """
        n_prot, n_nuc, n_ligand = 0, 0, 0
        ctype = chain_m.ctype
        if ctype.is_protein:
            n_prot += 1
        elif ctype.is_nucleic_acid:
            n_nuc += 1
        else:
            n_ligand += 1

        cluster_id = get_chain_cluster_id(chain_m)
        n_cluster = self.chain_cluster_sizes[cluster_id]

        # See Section 2.5.1 Equation 1
        weight = (self.beta_chain / n_cluster) * (
            self.alpha_prot * n_prot
            + self.alpha_nuc * n_nuc
            + self.alpha_ligand * n_ligand
        )
        return weight

    def _get_interface_weight(
        self, interface: InterfaceInfo, chain_dict: dict[int, ChainInfo]
    ) -> float:
        """Get the weight of an interface.

        Parameters
        ----------
        interface : InterfaceInfo
            The interface to get the weight for.
        chain_dict : dict[int, ChainInfo]
            The dictionary of chains in the complex. {asym_id: ChainInfo}

        Returns
        -------
        float
            The weight of the interface.
        """
        n_prot, n_nuc, n_ligand = 0, 0, 0
        for asym_id in interface.asym_ids:
            chain = chain_dict[asym_id]
            ctype = chain.ctype
            if ctype.is_protein:
                n_prot += 1
            elif ctype.is_nucleic_acid:
                n_nuc += 1
            else:
                n_ligand += 1

        cluster_id = get_interface_cluster_id(interface)
        n_cluster = self.interface_cluster_sizes[cluster_id]

        # See Section 2.5.1 Equation 1
        weight = (self.beta_interface / n_cluster) * (
            self.alpha_prot * n_prot
            + self.alpha_nuc * n_nuc
            + self.alpha_ligand * n_ligand
        )
        return weight
