import torch
from typing import Iterable
from distribution_components import distribution_components


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
        raise NotImplementedError
    individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')
    voxel_ids = torch.stack(individual_voxel_ids_meshed, -1)
    voxel_ids = voxel_ids.reshape(-1, voxel_ids.shape[-1])
    voxel_ids = voxel_ids.repeat(batch, 1, 1)
    voxel_ids = voxel_ids.to(torch.float32)
    return voxel_ids


def mc_sample_type_m(mean, number_of_sampl):
    number_of_pixels = mean.size(2)
    batch = mean.size(0)  # mean shape : batch_size, 1, number_of_pixels
    mean = mean.repeat(1, 1, number_of_sampl)   # batch, 1 , number_of_sampl* number_of_pixels ,
    # mean is of the form [[[m1],[m1],[m1]]... [[m_b],[ m_b],[m_b]]]
    mean = mean.reshape(batch, number_of_sampl, number_of_pixels)
    mc_samples = mean
    return mc_samples


def mc_sample_type_m_p(mean, cov_factor, number_of_sampl):
    number_of_pixels = mean.size(2)
    batch = mean.size(0)  # mean shape : batch_size, 1, number_of_pixels
    rank = cov_factor.size(2)  # cov_factor_shape torch.Size([batch_size, number_of_pixels, rank])
    samples_p = torch.normal(0, 1, size=(batch, rank, number_of_sampl)).to(torch.float32).cuda()
    mc_samples = cov_factor @ samples_p  # mc_samples shape (batch, number_of_pixels, number_of_samples (not per gt)
    mc_samples = mc_samples.transpose(1, -1)  # batch_size, number_of_samples, number_of_pixels
    mean = mean.repeat(1, 1, number_of_sampl)  # batch, 1 , number_of_sampl* number_of_pixels ,
    # mean is of the form [[[m1],[m1],[m1]]... [[m_b],[ m_b],[m_b]]]
    mean = mean.reshape(batch, number_of_sampl, number_of_pixels)
    mc_samples = mc_samples + mean
    return mc_samples


def mc_sample_type_m_p_d(mean, cov_factor, diagonal, number_of_sampl):
    number_of_pixels = mean.size(2)
    batch = mean.size(0)  # mean shape : batch_size, 1, number_of_pixels
    rank = cov_factor.size(2)  # cov_factor_shape torch.Size([batch_size, number_of_pixels, rank])
    samples_d = torch.normal(0, 1, size=(batch, number_of_pixels, number_of_sampl)).to(torch.float32).cuda()
    samples_p = torch.normal(0, 1, size=(batch, rank, number_of_sampl)).to(torch.float32).cuda()
    mc_samples = cov_factor @ samples_p + torch.sqrt(diagonal) * samples_d  # mc_samples shape (batch, number_of_pixels, number_of_samples) (not per gt)
    mc_samples = mc_samples.transpose(1, -1)  # batch_size, number_of_samples, number_of_pixels
    mean = mean.repeat(1, 1, number_of_sampl)  # batch, 1 , number_of_sampl* number_of_pixels ,
    # mean is of the form [[[m1],[m1],[m1]]... [[m_b],[ m_b],[m_b]]]
    mean = mean.reshape(batch, number_of_sampl, number_of_pixels)
    mc_samples = mc_samples + mean
    return mc_samples


def mc_sample_function(model_type):
    dispatcher = {"m_p_d": mc_sample_type_m_p_d, "m_p": mc_sample_type_m_p, "m": mc_sample_type_m}
    try:
        mc_samples_function = dispatcher[model_type]
    except KeyError:
        raise ValueError('invalid input')
    return mc_samples_function


def sampling(model_outputs, model_type, num_samples):
    vectors = distribution_components(model_outputs, model_type)
    sampling_func = mc_sample_function(model_type)
    if model_type == 'm':
        mc_samples = sampling_func(vectors, num_samples)
    else:
        mc_samples = sampling_func(*vectors, num_samples)
    return mc_samples
