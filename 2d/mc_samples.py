import torch
from typing import Iterable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_coordinates(image_shape: Iterable[int], batch, upsampling_factor, downsampling_factor) -> torch.Tensor:
    if upsampling_factor == 1 and downsampling_factor == 1:
        individual_voxel_ids = [torch.arange(num_elements) for num_elements in image_shape]

    elif upsampling_factor > 1 and downsampling_factor == 1:
        image_shape = [num_elements * upsampling_factor for num_elements in image_shape]
        individual_voxel_ids = [torch.arange(num_elements - 1) for num_elements in image_shape]
        individual_voxel_ids = [el / upsampling_factor for el in individual_voxel_ids]
    elif upsampling_factor == 1 and downsampling_factor > 1:
        num_elements = [int(num_elements / downsampling_factor) for num_elements in image_shape]
        individual_voxel_ids = [torch.linspace(0, x - 1, num_elements[(image_shape.index(x))]) for x in image_shape]
    else:
        print("either upsampling or downsampling factor is invalid!")
    individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')
    voxel_ids = torch.stack(individual_voxel_ids_meshed, -1)
    voxel_ids = voxel_ids.reshape(-1, voxel_ids.shape[-1])
    voxel_ids = voxel_ids.repeat(batch, 1, 1)
    voxel_ids = voxel_ids.to(torch.float32)
    return voxel_ids


#Not true!
def mc_sample_mean(mean, number_of_sample_per_gt):
    batch = mean.size(0)
    number_of_pixels = mean.size(2)
    mc_samples = torch.rand(batch, number_of_sample_per_gt, number_of_pixels)
    for b in range(batch):
        for i in range(number_of_sample_per_gt):
            mc_samples[b][i] = mean[b].view(-1)
    return mc_samples


def mc_sample_cov_is_low_rank(mean, cov_factor, number_of_sample_per_gt):
    number_of_pixels = mean.size(2)
    batch = mean.size(0)
    rank = cov_factor.size(1)
    mc_samples = torch.rand(number_of_sample_per_gt, batch, number_of_pixels)
    for i in range(number_of_sample_per_gt):
        sample_p = torch.normal(0, 1, size=(rank,)).to(torch.float32).cuda()
        for j in range(batch):
            mc_samples[i][j] = (cov_factor[j].T @ sample_p) + mean[j].view(-1)
    mc_samples = mc_samples.permute((1, 0, 2))
    return mc_samples
