"""
Code adopted from La-Proteina (https://github.com/NVIDIA-Digital-Bio/la-proteina).
"""

from math import prod

import torch
from scipy.spatial.transform import Rotation as Scipy_Rotation


def sample_uniform_rotation(shape=tuple(), dtype=None, device=None):
    """
    Samples rotations distributed uniformly. Adapted from FrameFlow's code.
    https://github.com/microsoft/protein-frame-flow/blob/main/data/so3_utils.py

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Scipy_Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)
