#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from typing import *


class Gamma(nn.Module):
    # γ(\mathbb{R}) -> (sin(2^{0..L-1}*\pi*p), cos(2^{0..L-1}*\pi*p))
    def __init__(self, L: int):
        self.L = L

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # lst: (2*L, batch_size, x_dim)
        lst = []
        for i in range(L):
            lst.append(torch.sin(2**i * x))
            lst.append(torch.cos(2**i * x))
        return torch.cat(lst, dim=-1)  # (batch_size, 2*L*x_dim)


class NeRF(nn.Module):
    # F_Θ
    def __init__(
        self,
        dim_position: int,
        dim_direction: int,
        num_linear_layers: int,
        dim_fully_connected: int,
        cat_position_index: List[int]
    ):
        # init basic variables
        super(NeRF, self).__init__()
        self.dim_in: int = dim_position
        self.num_linear_layers: int = num_linear_layers
        self.dim_fully_connected: int = dim_fully_connected

        # I'll imitate other's implementation this part
        self.linears = nn.ModuleList(
            [nn.Linear(dim_position, dim_fully_connected)]  # entry
            + [nn.Linear(dim_position + dim_fully_connected, dim_fully_connected) if i in cat_position_index
               else nn.Linear(dim_fully_connected, dim_fully_connected)
               for i in num_linear_layers])

        # implement tail layers
        self.alpha_output = nn.Linear(dim_fully_connected, 1)
        self.cat_direction = nn.Linear(
            dim_direction + dim_fully_connected, dim_fully_connected // 2)
        self.output = nn.Linear(dim_fully_connected // 2, 3)  # RGB

    def forward(
        self,
        x: torch.Tensor,
        d: torch.Tensor
    ) -> torch.Tensor:
        original_x = x
        for index, linear in enumerate(self.linears):
            x = linear(x)
            x = nn.functional.relu(x)

            # index is shifted because of the `entry`
            if index in cat_position_index:
                # dim=-1: vector is presented in column
                x = torch.cat([original_x, x], dim=-1)

        # tail layers
        # first, extract alpha from the layer
        alpha = self.alpha_output(x)
        # second, concat the direction vector
        x = torch.cat([x, d], dim=-1)
        x = self.cat_direction(x)
        x = nn.functional.relu(x)
        x = self.output(x)

        # concat alpha and direction vector
        x = torch.cat([x, alpha], dim=-1)

        return x


def get_rays(
        height: int,
        width: int,
        focal: float,
        transform_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(transform_matrix),
        torch.arange(height, dtype=torch.float32).to(transform_matrix),
        indexing='ij')
    # to height, width
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    directions = torch.stack(
        [i - width / 2, -(j - height / 2), -torch.ones_like(i), torch.zeros_like(i)], dim=-1)
    directions[:, :, :2] /= focal

    rays_d = directions @ transform_matrix.mT
    rays_o = transform_matrix @ torch.tensor(
        [0, 0, 0, 1], dtype=torch.float32).mT

    # normalize homogeneous coordinate
    rays_d = rays_d[:, :, :3]
    rays_o = (rays_o[:3] / rays_o[3]).expand(rays_d.shape)

    return rays_o, rays_d


# sample rays along their directions
# use delta-tracking instead?
def sample_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int
):
    t = torch.linspace(0, 1, num_samples + 1, device=rays_o.device)
    t = near + (far - near) * t  # evenly spread in (near, far)
    t_delta = t[1:] - t[:-1]  # len(t_delta) = len(t) - 1
    t = t[:-1]  # neglect the last term(the starting point)
    t_rand = torch.rand(num_samples, device=rays_o.device)
    t = t + t_delta * t_rand
    t = t.expand(list(rays_o.shape[:-1]) + [num_samples])
    sample_positions = rays_o[..., None, :] + \
        t[..., :, None] * rays_d[..., None, :]
    return sample_positions, t  # (width, height, num_samples, 3)


if __name__ == '__main__':
    # get_rays(10, 12, 10, torch.from_numpy(np.identity(4, dtype=np.float32)))
    print(sample_rays(torch.tensor([0, 1, 0]),
          torch.tensor([1, 0, 1]), 0, 10, 10))
