"""Utility functions for initializing weights and biases.
Modified from OpenFold-3 initialize.py
"""

# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import torch
from scipy.stats import truncnorm


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


@torch.no_grad()
def trunc_normal_init_openfold_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    weights.copy_(torch.as_tensor(samples, device=weights.device))


@torch.no_grad()
def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    """New truncated normal initialization consistent with
    pure PyTorch implementation.
    """
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)

    # Same to truncnorm.std with a=-2, b=2, loc=0, scale=1
    correction_factor = 0.87962566103423978

    std = math.sqrt(scale) / correction_factor

    torch.nn.init.trunc_normal_(
        weights,
        mean=0.0,
        std=std,
        a=-2.0 * std,
        b=2.0 * std,
    )


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


@torch.no_grad()
def glorot_uniform_init_(weights):
    torch.nn.init.xavier_uniform_(weights, gain=1)


@torch.no_grad()
def zero_init_(weights):
    weights.fill_(0.0)


@torch.no_grad()
def final_init_(weights):
    weights.fill_(0.0)


@torch.no_grad()
def gating_init_(weights):
    weights.fill_(0.0)


@torch.no_grad()
def bias_init_zero_(bias):
    bias.fill_(0.0)


@torch.no_grad()
def bias_init_one_(bias):
    bias.fill_(1.0)


@torch.no_grad()
def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


@torch.no_grad()
def ipa_point_weights_init_(weights):
    softplus_inverse_1 = 0.541324854612918
    weights.fill_(softplus_inverse_1)
