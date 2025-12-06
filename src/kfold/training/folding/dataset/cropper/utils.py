# started from code from https://github.com/jwohlwend/boltz, MIT License
import numbers

import numpy as np
from scipy.spatial.distance import cdist

import kfold.constants as C
from kfold.data.structure import TokenizedStructure


def pick_token(
    struct: TokenizedStructure,
    asym_id: int | tuple[int, int] | None = None,
    mask: np.ndarray | None = None,
) -> int:
    """Pick a random token from a chain.

    Parameters
    ----------
    struct : TokenizedStructure
        The tokenized structure.
    asym_id : int | tuple[int, int] | None, optional
        The chain asymmetric ID or interface chain IDs.
        If None, pick from all tokens.
    mask : np.ndarray | None, optional
        An optional mask of valid tokens.

    Returns
    -------
    token_index : int
        The selected token index.
    """
    if asym_id is None:
        # Pick from entire complex
        return pick_complex_token(struct, mask)
    if isinstance(asym_id, int | numbers.Integral):
        # Pick from specific chain
        return pick_chain_token(struct, asym_id, mask)
    else:
        # Pick from interface
        assert len(asym_id) == 2, "asym_id tuple must have length 2"
        return pick_interface_token(struct, asym_id, mask)


def pick_complex_token(
    struct: TokenizedStructure,
    mask: np.ndarray | None = None,
) -> int:
    """Pick a random token from the entire complex.

    Parameters
    ----------
    struct : TokenizedStructure
        The tokenized structure.
    mask : np.ndarray | None, optional
        An optional mask of valid tokens.

    Returns
    -------
    token_index : int
        The selected token index.
    """
    token_indices = struct.token.token_index
    if mask is not None:
        token_indices = token_indices[mask]
    return np.random.choice(token_indices)


def pick_chain_token(
    struct: TokenizedStructure,
    asym_id: int,
    mask: np.ndarray | None = None,
) -> int:
    """Pick a random token from a chain.

    Parameters
    ----------
    struct : TokenizedStructure
        The tokenized structure.
    asym_id : int
        The chain asymmetric ID.
    resolved_mask : np.ndarray | None, optional
        An optional mask of valid tokens.

    Returns
    -------
    token_index : int
        The selected token index.
    """
    # Get chain mask
    chain_mask = struct.token.asym_id == asym_id
    if mask is not None:
        chain_mask &= mask

    if not np.any(chain_mask):
        # Fallback to all tokens
        return pick_token(struct, asym_id=None, mask=mask)

    # Pick from chain, fallback to all tokens
    token_indices = struct.token.token_index  # =np.arange(num_tokens)
    chain_tokens = token_indices[chain_mask]
    return np.random.choice(chain_tokens)


def pick_interface_token(
    struct: TokenizedStructure,
    asym_ids: tuple[int, int],
    mask: np.ndarray | None = None,
) -> int:
    """Pick a random token from an interface.

    Parameters
    ----------
    struct : TokenizedStructure
        The tokenized data.
    asym_ids : tuple[int, int]
        The chain IDs defining the interface.
    mask : np.ndarray | None, optional
        An optional mask of valid tokens.

    Returns
    -------
    token_index : int
        The selected token index.
    """
    assert len(asym_ids) == 2, "asym_ids must be a tuple of length 2"

    if mask is None:
        # Interface can be determined only on resolved residues
        mask = struct.atom.resolved_mask[
            struct.token.token_index, struct.token.center_index
        ]  # (num_tokens,)

    # Get chain masks
    chain_1, chain_2 = asym_ids
    chain_mask_1 = (struct.token.asym_id == chain_1) & mask
    chain_mask_2 = (struct.token.asym_id == chain_2) & mask
    is_empty_1 = not np.any(chain_mask_1)
    is_empty_2 = not np.any(chain_mask_2)

    if is_empty_1 and is_empty_2:
        # Fallback to all tokens with resolved residues
        return pick_complex_token(struct, mask)

    if (not is_empty_1) and is_empty_2:
        # Fallback to chain 1 with resolved residues
        return pick_chain_token(struct, chain_1, mask)

    if is_empty_1 and (not is_empty_2):
        # Fallback to chain 2 with resolved residues
        return pick_chain_token(struct, chain_2, mask)

    # Get interface tokens
    tokens = struct.token.token_index  # =np.arange(num_tokens)
    center_index = struct.token.center_index
    tokens_1 = tokens[chain_mask_1]
    tokens_2 = tokens[chain_mask_2]

    # Compute distances between tokens in the two chains to determine interface
    holo_coords = struct.atom.coords[..., 0, :]  # (num_tokens, 24, 3)
    tokens_1_coords = holo_coords[tokens_1, center_index[tokens_1]]
    tokens_2_coords = holo_coords[tokens_2, center_index[tokens_2]]

    dists = cdist(tokens_1_coords, tokens_2_coords)
    cutoff = dists < C.INTERFACE_CUTOFF

    # In rare cases, the interface cuttoff is slightly to small,
    # then we slightly expand it if it happens
    if not np.any(cutoff):
        cutoff = dists < (C.INTERFACE_CUTOFF + 5.0)

    if not np.any(cutoff):
        # Fallback to all tokens with resolved residues
        return pick_complex_token(struct, mask)

    tokens_1 = tokens_1[cutoff.any(axis=1)]
    tokens_2 = tokens_2[cutoff.any(axis=0)]

    # Select random token
    candidates = np.concatenate([tokens_1, tokens_2])

    return np.random.choice(candidates)
