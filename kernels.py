#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from typing import *
import matplotlib.pyplot as plt


class Gamma(nn.Module):
    # γ(\mathbb{R}) -> (sin(2^{0..L-1}*\pi*p), cos(2^{0..L-1}*\pi*p))
    def __init__(self, L: int):
        super(Gamma, self).__init__()
        self.L = L

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # lst: (2*L, batch_size, x_dim)
        lst = []
        for i in range(self.L):
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
               for i in range(num_linear_layers)])

        # implement tail layers
        self.alpha_output = nn.Linear(dim_fully_connected, 1)
        self.cat_direction = nn.Linear(
            dim_direction + dim_fully_connected, dim_fully_connected // 2)
        self.output = nn.Linear(dim_fully_connected // 2, 3)  # RGB

        self.cat_position_index = cat_position_index

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
            if index in self.cat_position_index:
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
        transform_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(transform_matrix),
        torch.arange(height, dtype=torch.float32).to(transform_matrix),
        indexing='ij')
    # to height, width
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    directions = torch.stack(
        [(i - width / 2) / focal, -(j - height / 2) / focal, -torch.ones_like(i), torch.zeros_like(i)], dim=-1)
    rays_d = directions @ transform_matrix.mT
    rays_o = torch.tensor(
        [0, 0, 0, 1], dtype=torch.float32).to(transform_matrix) @ transform_matrix.mT

    # normalize homogeneous coordinate
    rays_o = rays_o[:3] / rays_o[3]
    rays_d = rays_d[:, :, :3]
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = rays_o.expand(rays_d.shape)

    # flatten the results
    rays_o = rays_o.reshape((-1, 3))
    rays_d = rays_d.reshape((-1, 3))

    return rays_o, rays_d


def rand_sorted(near: float, far: float, num_samples: int, device) -> torch.Tensor:
    t = torch.linspace(0, 1, num_samples + 1, device=device)
    t = near + (far - near) * t  # evenly spread in (near, far)
    t_delta = t[1:] - t[:-1]  # len(t_delta) = len(t) - 1
    t = t[:-1]  # neglect the last term(the starting point)
    t_rand = torch.rand(num_samples, device=device)
    t = t + t_delta * t_rand
    return t


def sample_rays(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # sample rays along their directions
    # use delta-tracking instead?
    t = rand_sorted(near, far, num_samples, device=rays_o.device)
    t = t.expand(list(rays_o.shape[:-1]) + [num_samples])
    sample_positions = rays_o[..., None, :] + \
        t[..., :, None] * rays_d[..., None, :]
    return sample_positions, t  # (num_rays, num_samples, 3)


def visualize_samples(samples: torch.Tensor) -> None:
    y = torch.zeros_like(samples)
    plt.plot(samples.cpu().numpy(),
             1 + y.cpu().numpy(), 'b-o')
    plt.ylim([0, 2])
    plt.grid(True)
    plt.show()


def weighted_sample_rays(
        rays_o: torch.Tensor,  # (num_rays, 3)
        rays_d: torch.Tensor,  # (num_rays, 3)
        weights: torch.Tensor,  # (num_rays, num_previous_samples)
        original_t: torch.Tensor,  # (num_rays, num_previous_samples)
        num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # the last term of weights is zero
    weights = nn.functional.relu(weights) + 1e-2
    pdf = (weights) / (torch.sum(weights, dim=-1, keepdim=True))
    cdf = torch.cumsum(pdf, dim=-1)  # (num_rays, num_previous_samples)

    rand = rand_sorted(0, 1, num_samples, device=cdf.device)
    rand = rand.expand(list(cdf.shape[:-1]) + [num_samples])
    cdf = cdf.contiguous()
    rand = rand.contiguous()
    indices = torch.searchsorted(cdf, rand)  # (num_rays, num_samples)

    with torch.no_grad():
        indices_max, _ = torch.max(indices, dim=-1)
        indices_max = torch.max(indices_max)
        assert indices_max < weights.shape[-1] - 1

    # sample the corresponding t
    delta_original_t = original_t[..., 1:] - original_t[..., :-1]
    start_t = torch.gather(original_t, dim=-1, index=indices)
    delta_t = torch.gather(delta_original_t, dim=-1, index=indices)
    t = start_t + delta_t * torch.rand(delta_t.shape, device=delta_t.device)

    if torch.isnan(t).any():
        print('!!! nan in t')
    if torch.isinf(t).any():
        print('!!! inf in t')

    weighted_sample_positions = rays_o[..., None, :] + \
        t[..., :, None] * rays_d[..., None, :]
    return weighted_sample_positions, t


def generate_any_batch(
    unbatched: torch.Tensor,
    batch_size: int
) -> torch.Tensor:
    return [unbatched[i:i + batch_size] for i in range(0, unbatched.shape[0], batch_size)]


def generate_batches(
        sample_positions: torch.Tensor,
        rays_d: torch.Tensor,
        position_encoding: Callable[[torch.Tensor], torch.Tensor],
        direction_encoding: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int
) -> torch.Tensor:
    # normalize direction
    rays_d = rays_d[:, None, ...].expand(sample_positions.shape)
    rays_d = rays_d.reshape((-1, 3))  # flatten the first two dim
    rays_d = direction_encoding(rays_d)
    rays_d = generate_any_batch(rays_d, batch_size)

    sample_positions = sample_positions.reshape((-1, 3))
    sample_positions = position_encoding(sample_positions)
    sample_positions = generate_any_batch(sample_positions, batch_size)
    return sample_positions, rays_d


def add_zero(tensor: torch.Tensor, front=True) -> torch.Tensor:
    tensor_zero = torch.zeros(list(tensor.shape[:-1]) + [1]).to(tensor)
    if front:
        tensor = torch.cat((tensor_zero, tensor), dim=-1)
    else:
        tensor = torch.cat((tensor, tensor_zero), dim=-1)
    return tensor


def render(
    # output from network
    results: torch.Tensor,  # (num_rays, num_samples, 4)
    rays_d: torch.Tensor,  # direction of the ray: (num_rays, 3)
    t: torch.Tensor,  # t on the ray's direction (num_rays, num_samples)
) -> Tuple[torch.Tensor]:
    # delta_{i} = t_{i+1} - t_{i}
    delta = t[..., 1:] - t[..., :-1]  # len(t_delta) = len(t) - 1
    T = results[..., 1:, 3] * delta
    # add zero before
    T = add_zero(T, front=True)
    T = torch.cumsum(T, dim=-1)
    T = torch.exp(-T)  # (width, height)

    delta = add_zero(delta, front=False)
    weights = T * (1 - torch.exp(-results[..., 3] * delta))
    rgb = results[..., :3] * weights[..., None]
    rgb = torch.sum(rgb, dim=-2)

    return rgb, weights


def render_with_samples(
    # Basic information
    rays_o: torch.Tensor,  # (num_rays, 3)
    rays_d: torch.Tensor,  # (num_rays, 3)
    batch_size: int,
    # Sampled information
    sample_positions: torch.Tensor,  # (num_rays, num_samples, 3)
    t: torch.Tensor,  # (num_rays, num_samples)
    # Encoding model
    network: nn.Module,
    position_encoding_network: nn.Module,
    direction_encoding_network: nn.Module,
):
    sample_positions_batches, rays_d_batches = generate_batches(
        sample_positions, rays_d,
        position_encoding=position_encoding_network, direction_encoding=direction_encoding_network,
        batch_size=batch_size)

    # pass through the network
    results = []
    for sample_positions_batch, rays_d_batch in zip(sample_positions_batches, rays_d_batches):
        result = network(sample_positions_batch, rays_d_batch)
        results.append(result)

    # results in coarse_results: [(batch_size, 4) ...] # RGB,
    results = torch.cat(results, dim=0)  # (all_samples, 4)
    results = results.reshape(
        list(sample_positions.shape[:2]) + [results.shape[-1]])

    rgb, weights = render(results, rays_d, t)

    return rgb, weights


def nerf_forward(
    # Basic information
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    batch_size: int,
    # Hierarchical volume sampling
    num_samples_coarse: int,
    num_samples_fine: int,
    coarse_network: nn.Module,
    fine_network: nn.Module,
    # encoding model
    position_encoding_network: nn.Module,
    direction_encoding_network: nn.Module,
) -> torch.Tensor:
    # First, sample some points along the ray for coarse model
    sample_positions, t = sample_rays(
        rays_o, rays_d, near, far, num_samples_coarse)

    coarse_rgb, weights = render_with_samples(
        rays_o, rays_d, batch_size, sample_positions, t, coarse_network, position_encoding_network, direction_encoding_network)

    # Then, perform weighted sample
    weighted_sample_positions, weighted_t = weighted_sample_rays(
        rays_o, rays_d, weights, t, num_samples_fine)

    fine_rgb, weights = render_with_samples(
        rays_o, rays_d, batch_size, weighted_sample_positions, weighted_t, fine_network, position_encoding_network, direction_encoding_network)

    return fine_rgb


if __name__ == '__main__':
    # get_rays(10, 12, 10, torch.from_numpy(np.identity(4, dtype=np.float32)))
    # print(sample_rays(torch.tensor([0, 1, 0]),
    #       torch.tensor([1, 0, 1]), 2, 10, 10))
    weighted_sample_rays(None, None, torch.tensor(
        [1, 5, 2, 0], dtype=torch.float32), torch.tensor([0, 1, 2, 3]), 80)
