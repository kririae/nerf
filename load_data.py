#!/usr/bin/env python3

import os
import json
import torch
import imageio
import numpy as np
from pathlib import Path
from progress.bar import Bar
import matplotlib.pyplot as plt

import kernels


def load_file(filename: Path):
    # extract basedir
    file = Path(filename)
    basedir = file.parent

    # load transforms from file
    transforms = None
    with file.open() as f:
        transforms = json.load(f)

    # load basic info
    camera_angle_x = transforms['camera_angle_x']
    frames = transforms['frames']
    num_images = len(frames)

    # load info from every frames
    images = []
    transforms = []
    bar = Bar('loading frames', max=num_images)
    for frame in frames:
        file_path = frame['file_path']
        rotation = frame['rotation']

        image = imageio.imread(basedir / Path(f'{file_path}.png'))
        # transform_matrix is a camera-to-world matrix
        transform_matrix = np.array(
            frame['transform_matrix']).astype(np.float32)

        images.append(image)
        transforms.append(transform_matrix)
        bar.next()
    bar.finish()

    # convert to numpy array
    images = np.array(images).astype(np.float32)
    transforms = np.array(transforms).astype(np.float32)
    height, width, _ = images[0].shape
    focal = width / 2 / np.tan(0.5 * camera_angle_x)
    return images, transforms, focal


def load_npz(filename: Path):
    data = np.load(filename)
    images = data['images']
    transforms = data['poses']
    focal = float(data['focal'])

    return images, transforms, focal


def display_rays(
    rays_o: np.ndarray,
    rays_d: np.ndarray
):
    # visualization code
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(
        rays_o[..., 0].flatten(),
        rays_o[..., 1].flatten(),
        rays_o[..., 2].flatten(),
        rays_d[..., 0].flatten(),
        rays_d[..., 1].flatten(),
        rays_d[..., 2].flatten(), length=0.5, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.show()


def display_cameras(filename: Path):
    images, transforms, _ = load_file(filename)

    # through the definition of homogeneous coordinate
    origins = transforms @ np.array([0, 0, 0, 1]).T
    dirs = transforms @ np.array([0, 0, -1, 0]).T

    display_rays(origins, dirs)


def display_cameras_npz():
    images, transforms, _ = load_npz('data/tiny_nerf_data.npz')
    origins = transforms @ np.array([0, 0, 0, 1]).T
    dirs = transforms @ np.array([0, 0, -1, 0]).T
    display_rays(origins, dirs)


def test_get_rays():
    images, transforms, focal = load_npz('data/tiny_nerf_data.npz')
    assert len(images) > 0

    image = images[101]
    transform_matrix = torch.from_numpy(transforms[101])
    height, width, _ = image.shape
    with torch.no_grad():
        rays_o, rays_d = kernels.get_rays(
            height, width, focal, transform_matrix)
    print(rays_o[height // 2, width // 2, :])
    print(rays_d[height // 2, width // 2, :])

    display_rays(rays_o, rays_d)


if __name__ == '__main__':
    # load file test
    # display_cameras('data/nerf_synthetic/lego/transforms_train.json')
    # test_get_rays()
    display_cameras_npz()
