import numpy as np
import scipy.spatial

import kfold.constants as C
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import Chain, RefStructure


def get_chain_ref_atom_coordinates(chain: Chain) -> np.ndarray:
    """Get reference atom coordinates for a chain."""
    if chain.ctype.is_nonpolymer:
        # Return all atom coordinates for non-polymer chains
        return chain.atom.coords
    else:
        match chain.ctype:
            case C.ChainType.PROTEIN:
                ref_atom = "CA"  # Alpha carbon
            case C.ChainType.RNA:
                ref_atom = "C1'"
            case C.ChainType.DNA:
                ref_atom = "C1'"
        ref_atom_mask = chain.atom.name == ref_atom
        ref_coords = chain.atom.coords[ref_atom_mask]
        return ref_coords


def extract_substructure(
    ref_struct: RefStructure,
    max_chains: int = 20,
    bias_asym_id: int | tuple[int, int] | None = None,
    rng: np.random.Generator | None = None,
) -> RefStructure:
    """Pre-Cropper that selects up to `max_chains` chains.
    See AlphaFold 3 SI Section 2.5.4.

    NOTE: The output of this cropper is treated as the "original" structure
    rather than a training crop. It effectively samples neighboring chains to
    reduce the complex to a manageable bioassembly size before further processing.

    NOTE: Unlike the original AF3 description which always samples the anchor token
    from an interface, this cropper falls back to sampling any resolved token
    from the specified chain (=bias) if no valid interfaces are found.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure.
    max_chains : int, optional
        The maximum number of chains to keep.
    bias_asym_ids : tuple[int, ...] | None, optional
        If specified, bias the cropping to include these asym_ids.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    cropped_struct: TokenizedStructure
        The sub-complex structure with limited number of chains.
    """
    rng = rng or np.random.default_rng()

    # Validate metadata
    metadata: Metadata = ref_struct.metadata

    if ref_struct.num_chains <= max_chains:
        # No cropping needed
        return ref_struct

    # Collect reference coordinates
    ref_coords_dict: dict[int, np.ndarray] = {}

    for chain in ref_struct.chains:
        """Get reference atom indices for a chain."""
        ref_coords = get_chain_ref_atom_coordinates(chain)
        # Only keep finite coordinates
        ref_coords = ref_coords[np.isfinite(ref_coords).all(axis=-1)]
        ref_coords_dict[chain.asym_id] = ref_coords

    # Sample random interfaces
    if isinstance(bias_asym_id, tuple):
        candidate_ifaces = [bias_asym_id]
    elif isinstance(bias_asym_id, int):
        # Sample an interface involving the bias chain
        candidate_ifaces = [
            iface.asym_ids
            for iface in metadata.interfaces
            if (bias_asym_id in iface.asym_ids)
        ]
    else:
        candidate_ifaces = [iface.asym_ids for iface in metadata.interfaces]

    if len(candidate_ifaces) > 0:
        asym_id1, asym_id2 = candidate_ifaces[rng.integers(len(candidate_ifaces))]

        # Get interface atom
        ref_coords1 = ref_coords_dict[asym_id1]  # [N1, 3]
        ref_coords2 = ref_coords_dict[asym_id2]  # [N2, 3]
        dists = scipy.spatial.distance.cdist(ref_coords1, ref_coords2)
        is_contact = dists < 15.0

        if np.any(is_contact):
            if rng.random() < 0.5:
                # Start from chain 1
                is_contact_1 = np.any(is_contact, axis=1)
                contact_indices_1 = np.where(is_contact_1)[0]
                seed_atom = rng.choice(contact_indices_1)
                seed_coords = ref_coords1[seed_atom, :].reshape(1, 3)
            else:
                # Start from chain 2
                is_contact_2 = np.any(is_contact, axis=0)
                contact_indices_2 = np.where(is_contact_2)[0]
                seed_atom = rng.choice(contact_indices_2)
                seed_coords = ref_coords2[seed_atom, :].reshape(1, 3)
        else:
            # Fallback: random atom from either chain
            if rng.random() < 0.5:
                seed_atom = rng.integers(ref_coords1.shape[0])
                seed_coords = ref_coords1[seed_atom, :].reshape(1, 3)
            else:
                seed_atom = rng.integers(ref_coords2.shape[0])
                seed_coords = ref_coords2[seed_atom, :].reshape(1, 3)

    else:
        # Fallback: random token
        if isinstance(bias_asym_id, tuple):
            candidate_asym_ids = list(bias_asym_id)
        elif isinstance(bias_asym_id, int):
            candidate_asym_ids = [bias_asym_id]
        else:
            candidate_asym_ids = list(ref_coords_dict.keys())
        random_asym_id = rng.choice(candidate_asym_ids)
        ref_coords = ref_coords_dict[random_asym_id]
        seed_atom = rng.integers(ref_coords.shape[0])
        seed_coords = ref_coords[seed_atom, :].reshape(1, 3)

    # Collect closest chains until reaching max_chains
    chain_dists: dict[int, float] = {}
    for chain in ref_struct.chains:
        coords = ref_coords_dict[chain.asym_id]
        dists = np.linalg.norm(coords - seed_coords, axis=-1)
        min_dist = np.min(dists)
        chain_dists[chain.asym_id] = min_dist

    sorted_chains = sorted(
        chain_dists.items(),
        key=lambda x: x[1],
    )  # list of (asym_id, dist)
    selected_asym_ids = set(asym_id for asym_id, _ in sorted_chains[:max_chains])

    # Filter chains
    new_chains = [
        chain for chain in ref_struct.chains if chain.asym_id in selected_asym_ids
    ]
    # Filter connections
    new_connections = [
        c
        for c in ref_struct.connections
        if c.asym_id[0] in selected_asym_ids and c.asym_id[1] in selected_asym_ids
    ]

    # Filter metadata
    new_chain_metas = [cm for cm in metadata.chains if cm.asym_id in selected_asym_ids]
    new_interfaces_meta = [
        iface
        for iface in metadata.interfaces
        if (
            iface.asym_ids[0] in selected_asym_ids
            and iface.asym_ids[1] in selected_asym_ids
        )
    ]
    new_metadata = Metadata(
        id=metadata.id,
        source=metadata.source,
        exp=metadata.exp,
        prediction=metadata.prediction,
        chains=new_chain_metas,
        interfaces=new_interfaces_meta,
    )
    new_ref_struct = RefStructure(
        chains=new_chains,
        connections=new_connections,
        metadata=new_metadata,
    )
    return new_ref_struct
