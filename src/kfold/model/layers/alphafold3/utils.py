# started from code from https://github.com/jwohlwend/boltz, MIT License,

from typing import TypeVar, overload

import torch
from torch.types import Device

_T = TypeVar("_T")


def expand_batch(x: torch.Tensor, b: int) -> torch.Tensor:
    """Expands a tensor x to have batch size b by unsqueezing and expanding."""
    if b == 1:
        return x.unsqueeze(0)
    return x.unsqueeze(0).expand(b, *x.shape)


def repeat_batch(x: torch.Tensor, b: int) -> torch.Tensor:
    """Repeats a tensor x to have batch size b by unsqueezing and repeating."""
    if b == 1:
        return x.unsqueeze(0)
    return x.unsqueeze(0).repeat(b, *(1,) * x.ndim)


def exists(v) -> bool:
    return v is not None


def default(v: _T | None, d: _T) -> _T:
    return v if exists(v) else d  # type: ignore[return-value]


def log(t: torch.Tensor, eps=1e-20) -> torch.Tensor:
    return torch.log(t.clamp(min=eps))


class ExponentialMovingAverage:
    """from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py,
    Apache-2.0 license
    Maintains (exponential) moving average of a set of parameters."""

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters, strict=True):
                s_param.sub_(one_minus_decay * (s_param - param))

    def compatible(self, parameters):
        if len(self.shadow_params) != len(parameters):
            print(
                f"Model has {len(self.shadow_params)} parameter tensors, the incoming ema"
                f"{len(parameters)}"
            )
            return False

        for s_param, param in zip(self.shadow_params, parameters, strict=True):
            if param.data.shape != s_param.data.shape:
                print(
                    f"Model has parameter tensor of shape {s_param.data.shape},"
                    f" the incoming ema {param.data.shape}"
                )
                return False
        return True

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters, strict=True):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters, strict=True):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [tensor.to(device) for tensor in state_dict["shadow_params"]]

    def to(self, device):
        self.shadow_params = [tensor.to(device) for tensor in self.shadow_params]


class CenterRandomAugmentation:
    """Centering and Random Augmentation Module
    See Section 3.7 Algorithm 19 CentreRandomAugmentation

    Usage)
    ```python
    augment = CenterRandomAugmentation(...)
    x = augment(x, atom_mask=mask)
    x, y = augment(x, y, atom_mask=mask)
    ```
    """

    def __init__(
        self, s_trans: float = 1.0, centering: bool = True, random_rotate: bool = True
    ):
        self.s_trans: float = s_trans
        self.centering: bool = centering
        self.random_rotate: bool = random_rotate

    @overload
    def __call__(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
        *others: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]: ...

    def __call__(  # type: ignore[override]
        self,
        *coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return self.augment(*coords, atom_mask=atom_mask)

    @overload
    def augment(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def augment(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
        *others: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]: ...

    def augment(  # type: ignore[override]
        self,
        *coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """See Section 3.7 Algorithm 19 CentreRandomAugmentation

        Parameters
        ----------
        coords : torch.Tensor
            One or more tensors of shape (B, N, 3) representing atomic coordinates.
        atom_mask : torch.Tensor
            A tensor of shape (B, N) representing the atom mask.
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
            return tuple(coords_list)


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
