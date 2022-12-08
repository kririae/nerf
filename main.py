#!/usr/bin/env python3

import torch
import torch.optim
from pathlib import *
from tqdm import trange

# project files
from kernels import *
from load_data import *

# device init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f'torch using device: {device}')


# NeRF parameters
position_L = 10
direction_L = 4
dim_position = position_L * 6
dim_direction = direction_L * 6
num_linear_layers = 8
dim_fully_connected = 256
cat_position_index = [4]
near = 2
far = 6
num_samples_coarse = 8
num_samples_fine = 8

# training parameters
num_iters = 10000
batch_size = 2**12
lr = 5e-4

# networks
coarse_NeRF: nn.Module = NeRF(
    dim_position=dim_position,
    dim_direction=dim_direction,
    num_linear_layers=num_linear_layers,
    dim_fully_connected=dim_fully_connected,
    cat_position_index=cat_position_index).to(device)
fine_NeRF: nn.Module = NeRF(
    dim_position=dim_position,
    dim_direction=dim_direction,
    num_linear_layers=num_linear_layers,
    dim_fully_connected=dim_fully_connected,
    cat_position_index=cat_position_index).to(device)
position_encoding_network = Gamma(L=position_L).to(device)
direction_encoding_network = Gamma(L=direction_L).to(device)

# optimizing parameters
coarse_NeRF_params = coarse_NeRF.parameters()
fine_NeRF_params = fine_NeRF.parameters()
parameters = list(coarse_NeRF_params) + list(fine_NeRF_params)

optimizer = torch.optim.Adam(parameters, lr=lr)

train_transform_filename = Path(
    'data/nerf_synthetic/lego/transforms_train.json')
test_transform_filename = Path(
    'data/nerf_synthetic/lego/transforms_test.json')
val_transform_filename = Path(
    'data/nerf_synthetic/lego/transforms_val.json')
coarse_NeRF_model_filename = Path('data/coarse_NeRF.pt')
fine_NeRF_model_filename = Path('data/fine_NeRF.pt')

display_steps = 50
save_steps = 1000


def train():
    if coarse_NeRF_model_filename.exists():
        coarse_NeRF.load_state_dict(torch.load(coarse_NeRF_model_filename))
        coarse_NeRF.eval()

    if fine_NeRF_model_filename.exists():
        fine_NeRF.load_state_dict(torch.load(fine_NeRF_model_filename))
        fine_NeRF.eval()

    # images, transforms, focal = load_file(train_transform_filename)
    images, transforms, focal = load_npz('data/tiny_nerf_data.npz')
    images = torch.from_numpy(images)
    transforms = torch.from_numpy(transforms)
    test_image = images[101].to(device)
    test_transform_matrix = transforms[101].to(device)
    images = images[:100]
    transforms = transforms[:100]
    focal = float(focal)

    for i in trange(num_iters):
        coarse_NeRF.train()
        fine_NeRF.train()

        # randomly select any image
        images_index = np.random.randint(images.shape[0])
        image = images[images_index].to(device)
        transform_matrix = transforms[images_index].to(device)
        height, width = image.shape[:2]

        # spawn ray to image-plane
        rays_o, rays_d = get_rays(height, width, focal, transform_matrix)

        # predicted rgb
        fine_rgb, coarse_rgb, _ = nerf_forward(
            rays_o=rays_o,
            rays_d=rays_d,
            near=near,
            far=far,
            batch_size=batch_size,
            num_samples_coarse=num_samples_coarse,
            num_samples_fine=num_samples_fine,
            coarse_network=coarse_NeRF,
            fine_network=fine_NeRF,
            position_encoding_network=position_encoding_network,
            direction_encoding_network=direction_encoding_network)

        loss_fine = nn.functional.mse_loss(fine_rgb, image.reshape((-1, 3)))
        loss_coarse = nn.functional.mse_loss(
            coarse_rgb, image.reshape((-1, 3)))
        loss = loss_fine + loss_coarse
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del image
        del transform_matrix
        del rays_o
        del rays_d

        if i % display_steps == 0:
            coarse_NeRF.eval()
            fine_NeRF.eval()

            test_rays_o, test_rays_d = get_rays(
                height, width, focal, test_transform_matrix)

            test_fine_rgb, test_coarse_rgb, test_weighted_t = nerf_forward(
                rays_o=test_rays_o,
                rays_d=test_rays_d,
                near=near,
                far=far,
                batch_size=batch_size,
                num_samples_coarse=num_samples_coarse,
                num_samples_fine=num_samples_fine,
                coarse_network=coarse_NeRF,
                fine_network=fine_NeRF,
                position_encoding_network=position_encoding_network,
                direction_encoding_network=direction_encoding_network)

            fig = plt.figure()
            gs = fig.add_gridspec(2, 3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[1, :])

            ax1.imshow(test_fine_rgb.reshape(
                (height, width, 3)).detach().cpu().numpy())
            ax1.set_title('fine_rgb')
            ax2.imshow(test_coarse_rgb.reshape(
                (height, width, 3)).detach().cpu().numpy())
            ax2.set_title('coarse_rgb')
            ax3.imshow(test_image.detach().cpu().numpy())
            ax3.set_title('ref_rgb')

            y = torch.zeros_like(test_weighted_t)
            ax4.plot(test_weighted_t.detach().cpu().numpy(),
                     1 + y.cpu().numpy(), 'b-o')
            ax4.set_ylim([0, 2])
            ax4.grid(True)
            plt.savefig('data/output.png')

        if i % 1000 == 0:
            print('model saved')
            torch.save(coarse_NeRF.state_dict(), coarse_NeRF_model_filename)
            torch.save(fine_NeRF.state_dict(), fine_NeRF_model_filename)

    print('model saved')
    torch.save(coarse_NeRF.state_dict(), coarse_NeRF_model_filename)
    torch.save(fine_NeRF.state_dict(), fine_NeRF_model_filename)


def main():
    train()


if __name__ == '__main__':
    main()
