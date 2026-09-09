"""Clean RCSB structure bonds before serializing training and validation data."""

import numpy as np

from kfold.data.types.structure import BondLayout, Chain, CovalentConnection, RefStructure

MAX_BOND_LENGTH = 2.4


def clean_up_ref_structure(
    struct: RefStructure,
) -> RefStructure:
    """Prepare the reference structure from chains.

    Parameters
    ----------
    struct : RefStructure
        The reference structure to be cleaned.

    Returns
    -------
    clean_struct : RefStructure
        The cleaned reference structure.
    """
    # clean up each chain
    struct = RefStructure(
        chains=[clean_up_chain(chain) for chain in struct.chains],
        connections=struct.connections,
        metadata=struct.metadata,
    )
    # clean up connections
    struct = clean_up_connections(struct)
    return struct


def clean_up_chain(
    chain: Chain,
) -> Chain:
    """Clean up the reference chain.

    Parameters
    ----------
    chain : RefStructure.Chain
        The reference chain to be cleaned.

    Returns
    -------
    clean_chain : Chain
        The cleaned reference chain.
    """

    # For now, we only remove bonds with unrealistic bond lengths.
    bonds = chain.bond
    is_valid = np.zeros(len(bonds), dtype=bool)
    for bond_i in range(len(bonds)):
        res_idx1, res_idx2 = bonds.residue_index[bond_i]
        atom1, atom2 = bonds.atom_name[bond_i]
        aidx1 = chain.find_atom_index(res_idx1, atom1)
        aidx2 = chain.find_atom_index(res_idx2, atom2)

        coord1 = chain.atom.coords[aidx1]
        coord2 = chain.atom.coords[aidx2]
        dsq = ((coord1 - coord2) ** 2).sum()
        if dsq < MAX_BOND_LENGTH**2 or np.isnan(dsq):
            # Keep the bond if the distance is less than the threshold
            # or if the distance is NaN since we can't ensure this is
            # an unrealistic bond
            is_valid[bond_i] = True

    clean_bond = BondLayout(
        residue_index=bonds.residue_index[is_valid],
        atom_name=bonds.atom_name[is_valid],
        bond_type=bonds.bond_type[is_valid],
    )

    return chain.copy_with(bond=clean_bond)


def clean_up_connections(
    struct: RefStructure,
) -> RefStructure:
    """Clean up the covalent connections by removing unrealistic connections.

    Parameters
    ----------
    struct : RefStructure
        The reference structure containing the connections to be cleaned.

    Returns
    -------
    clean_struct : RefStructure
        The reference structure with cleaned connections.
    """
    # For training, remove connections with unrealistic bond lengths.
    connections: list[CovalentConnection] = struct.connections
    if len(connections) == 0:
        return struct

    asym_id_to_chain = {c.asym_id: c for c in struct.chains}
    clean_connections: list[CovalentConnection] = []
    for conn in connections:
        asym_id1, asym_id2 = conn.asym_id
        res_idx1, res_idx2 = conn.residue_index
        atom1, atom2 = conn.atom_names
        chain1 = asym_id_to_chain[asym_id1]
        chain2 = asym_id_to_chain[asym_id2]

        # Get the coordinates of the connected atoms
        aidx1 = chain1.find_atom_index(res_idx1, atom1)
        aidx2 = chain2.find_atom_index(res_idx2, atom2)

        coord1 = chain1.atom.coords[aidx1]
        coord2 = chain2.atom.coords[aidx2]
        dsq = ((coord1 - coord2) ** 2).sum()

        if dsq < MAX_BOND_LENGTH**2 or np.isnan(dsq):
            # Keep the bond if the distance is less than the threshold
            # or if the distance is NaN since we can't ensure this is
            # an unrealistic bond
            clean_connections.append(conn)

    return RefStructure(
        chains=struct.chains,
        connections=clean_connections,
        metadata=struct.metadata,
    )
