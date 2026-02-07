import torch

import kfold.constants as C


def compute_pair_interactions(interaction_type: torch.Tensor) -> torch.Tensor:
    """Project primitive interaction types to 5 symmetric pair features.

    Parameters
    ----------
    interaction_type : torch.Tensor
        Multi-hot primitive interaction types of shape [*, T].

    Returns
    -------
    torch.Tensor
        Pair interaction features of shape [*, T, T, 5].
        The last dimension follows ``C.PairInteractionType`` order.
    """
    interaction_type = interaction_type
    hi = interaction_type[..., C.InteractionType.HI]
    hbd = interaction_type[..., C.InteractionType.HBD]
    hba = interaction_type[..., C.InteractionType.HBA]
    sbc = interaction_type[..., C.InteractionType.SBC]
    sba = interaction_type[..., C.InteractionType.SBA]
    pp = interaction_type[..., C.InteractionType.PP]
    pc = interaction_type[..., C.InteractionType.PC]

    def pair_and(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x[..., None] & y[..., None, :]

    hydrophobic = pair_and(hi, hi)
    hydrogen_bond = pair_and(hbd, hba) | pair_and(hba, hbd)
    salt_bridge = pair_and(sbc, sba) | pair_and(sba, sbc)
    pi_pi = pair_and(pp, pp)
    pi_cation = pair_and(pp, pc) | pair_and(pc, pp)

    pair_interactions = torch.stack(
        [hydrophobic, hydrogen_bond, salt_bridge, pi_pi, pi_cation], dim=-1
    )
    return pair_interactions
