# started from code from https://github.com/jwohlwend/boltz, MIT License,

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device

from . import initialize as init

LinearNoBias = partial(nn.Linear, bias=False)


class Transition(nn.Module):
    """Perform a two-layer MLP.
    See Section 3.3 Algorithm 11 Transition layer
    """

    def __init__(
        self,
        channel: int,
        expansion_factor: int,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        channel: int
            The dimension of the input
        expansion_factor: int
            The expansion factor for the hidden dimension

        """
        super().__init__()

        model_dim = channel * expansion_factor
        self.model_dim = model_dim
        self.layernorm = nn.LayerNorm(channel, eps=1e-5)
        self.linear_no_bias_a = LinearNoBias(channel, model_dim)
        self.linear_no_bias_b = LinearNoBias(channel, model_dim)
        self.linear_no_bias_out = LinearNoBias(model_dim, channel)

        init.bias_init_one_(self.layernorm.weight)
        init.bias_init_zero_(self.layernorm.bias)

        init.lecun_normal_init_(self.linear_no_bias_a.weight)
        init.lecun_normal_init_(self.linear_no_bias_b.weight)
        init.final_init_(self.linear_no_bias_out.weight)

    def forward(self, x: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        """Perform a forward pass.
        See Section 3.3 Algorithm 11 Transition layer

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)
        chunk_size: Optional[int]
            The chunk size for memory-efficient computation, default None

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        # Line 1
        x = self.layernorm(x)

        if chunk_size is None or self.training:
            # Line 2
            a = self.linear_no_bias_a(x)

            # Line 3
            b = self.linear_no_bias_b(x)

            # Line 4
            x = self.linear_no_bias_out(F.silu(a) * b)
            return x
        else:
            # Compute in chunks
            for i in range(0, self.model_dim, chunk_size):
                lin_a_slice = self.linear_no_bias_a.weight[i : i + chunk_size, :]
                lin_b_slice = self.linear_no_bias_b.weight[i : i + chunk_size, :]
                lin_o_slice = self.linear_no_bias_out.weight[:, i : i + chunk_size]
                x_chunk = F.silu(x @ lin_a_slice.T) * (x @ lin_b_slice.T)
                if i == 0:
                    x_out = x_chunk @ lin_o_slice.T
                else:
                    x_out = x_out + x_chunk @ lin_o_slice.T  # type: ignore
            return x_out  # type: ignore


class AdaLN(nn.Module):
    """Adaptive Layer Normalization
    See Section 3.7 Algorithm 26 Adaptive LayerNorm
    """

    def __init__(self, channel_a: int, channel_s: int):
        """Initialize the adaptive layer normalization.

        Parameters
        ----------
        channel_a : int
            The input dimension.
        channel_s : int
            The single condition dimension.

        """
        super().__init__()
        self.a_norm = nn.LayerNorm(channel_a, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(channel_s, bias=False)
        self.s_scale = nn.Linear(channel_s, channel_a)
        self.s_bias = LinearNoBias(channel_s, channel_a)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """see Section 3.7 Algorithm 26 Adaptive LayerNorm"""
        # Line 1
        a = self.a_norm(a)
        # Line 2
        s = self.s_norm(s)
        # Line 3
        a = torch.sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a


class CenterRandomAugmentation(nn.Module):
    """Centering and Random Augmentation Module
    See Section 3.7 Algorithm 19 CentreRandomAugmentation
    """

    def __init__(
        self, s_trans: float = 1.0, centering: bool = True, random_rotate: bool = True
    ):
        super().__init__()
        self.s_trans: float = s_trans
        self.centering: bool = centering
        self.random_rotate: bool = random_rotate

    def forward(
        self,
        *coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        """See Section 3.7 Algorithm 19 CentreRandomAugmentation

        Parameters
        ----------
        coords : torch.Tensor
            One or more tensors of shape (B, N, 3) representing atomic coordinates.
        atom_mask : torch.Tensor
            A tensor of shape (B, N) representing the atom mask.

        examples)
        ```python

        random_aug = CenterRandomAugmentation(...)
        x = random_aug(x, atom_mask=mask)
        x, y = random_aug(x, y, atom_mask=mask)

        ```
        """

        coords_list: list[torch.Tensor] = list(coords)
        ref_coords = coords_list[0]
        B, N = atom_mask.shape

        # Check all input coords have the same batch size and number of atoms
        for c in coords_list:
            assert c.shape[0] == B and c.shape[1] == N, (
                "All input coordinate tensors must have the same batch size and length."
            )

        # Line 1
        if self.centering:
            center = torch.sum(
                ref_coords * atom_mask[:, :, None], dim=1, keepdim=True
            ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)

            coords_list = [x - center for x in coords_list]

        # Line 2,4
        if self.random_rotate:
            R = random_rotations(N, ref_coords.dtype, ref_coords.device)
            rotate = lambda x: torch.einsum("bmd,bds->bms", x, R)  # noqa
            coords_list = [rotate(x) for x in coords_list]

        # Line 3,4
        if self.s_trans > 0.0:
            random_trans = torch.randn_like(ref_coords[:, 0:1, :]) * self.s_trans
            coords_list = [x + random_trans for x in coords_list]

        if len(coords) == 1:
            # Single tensor input, return tensor
            return coords_list[0]
        else:
            # Multiple tensor input, return list of tensors
            return coords_list


def center(atom_coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    atom_mean = torch.sum(
        atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
    ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
    atom_coords = atom_coords - atom_mean
    return atom_coords


def compute_random_augmentation(
    num_diffusion_samples: int,
    s_trans: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
):
    R = random_rotations(num_diffusion_samples, dtype=dtype, device=device)
    random_trans = (
        torch.randn((num_diffusion_samples, 1, 3), dtype=dtype, device=device) * s_trans
    )
    return R, random_trans


# the following is copied from Torch3D, BSD License,
# Copyright (c) Meta Platforms, Inc. and affiliates.


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int, dtype: torch.dtype | None = None, device: Device | None = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: torch.dtype | None = None, device: Device | None = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)
