import torch
from lidcdata import LIDC_IDRI
from typing import Iterable
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse

# data_folder = '/srv/public/esirin/data/'
data_folder = '/scratch/visual/esirin/data/'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location=data_folder)


def ground_truths(n, data):
    gts = data[n][1]
    return gts


def get_coordinates(image_shape: Iterable[int], upsampling_factor, downsampling_factor) -> torch.Tensor:
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
    voxel_ids = voxel_ids.to(torch.float32)

    return voxel_ids


def function_eval(mean_func, log_diagonal_func, cov_factor_func, coords):
    mu = 63.5
    sigma = 37.09
    coords = (coords - mu) / sigma
    mean_vector = mean_func(coords)
    diagonal = log_diagonal_func(coords)
    #print(torch.any(torch.isnan(diagonal)))
    diagonal_matrix = torch.exp(diagonal.view(-1))
    cov_factor_matrix = cov_factor_func(coords)
    return mean_vector, diagonal_matrix, cov_factor_matrix


def mc_sample_mean(mean, number_of_sample):
    mc_samples = torch.rand(number_of_sample, mean.size(0))
    for i in range(number_of_sample):
        mc_samples[i] = mean.view(-1)
    return mc_samples


def mc_sample_cov_is_low_rank(mean, diagonal, cov_factor, number_of_sample):
    mc_samples = torch.rand(number_of_sample, diagonal.size(0))
    diagonal = torch.sqrt(diagonal)
    for i in range(number_of_sample):
        sample_d = torch.normal(0, 1, size=(diagonal.size(0),)).to(torch.float32).to(device=DEVICE)
        sample_p = torch.normal(0, 1, size=(cov_factor.size(1),)).to(torch.float32).to(device=DEVICE)
        mc_samples[i] = (cov_factor @ sample_p) + (diagonal * sample_d) + mean.view(-1)
    return mc_samples


class DeepSDF(nn.Module):
    def __init__(self, in_ch, latent_size, out_ch):
        super().__init__()
        self.latent_parameter = torch.nn.Parameter(torch.normal(0.0, 1e-4, [1, latent_size]), requires_grad=True)
        self.fc1 = nn.Linear(in_ch + latent_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256 - (in_ch + latent_size))
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, out_ch)

    def forward(self, x):
        latent_vec = self.latent_parameter.expand(len(x), -1)
        x1 = torch.cat((x, latent_vec), 1)
        x2 = F.relu(self.fc1(x1))
        x3 = F.relu(self.fc2(x2))
        x4 = F.relu(self.fc3(x3))
        x5 = F.relu(self.fc4(x4))
        y = torch.cat((x1, x5), 1)
        x6 = F.relu(self.fc5(y))
        x7 = F.relu(self.fc6(x6))
        x8 = F.relu(self.fc7(x7))
        x9 = self.fc8(x8)
        return x9


class Disc(nn.Module):
    def __init__(self, coords, out_ch):
        super().__init__()
        self.latent_parameter = torch.nn.Parameter(torch.normal(0.0, 1e-4, [coords ** 2, out_ch]), requires_grad=True)

    def forward(self, x):
        x = self.latent_parameter
        return x


def train_low_rank(t, number_pre_epochs, gts, mean, log_diagonal, cov_factor, loss_function, optimizer_m, optimizer_a,
                   number_of_samples, coords, writer):
    mean_vec, diagonal_vec, cov_factor_matrix = function_eval(mean, log_diagonal, cov_factor, coords)
    log_prob = torch.zeros(4, number_of_samples)
    if t <= number_pre_epochs:
        optimizer = optimizer_m
        for i in range(4):
            mc_samples = mc_sample_mean(mean_vec, number_of_samples).to(device=DEVICE)
            gt = gts[i].view(-1).to(device=DEVICE)
            for j in range(number_of_samples):
                log_prob[i][j] = -loss_function(mc_samples[j], gt).to(device=DEVICE)

    else:
        optimizer = optimizer_a
        for i in range(4):
            mc_samples = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples).to(
                device=DEVICE)
            gt = gts[i].view(-1).to(device=DEVICE)
            for j in range(number_of_samples):
                log_prob[i][j] = -loss_function(mc_samples[j], gt).to(device=DEVICE)
    loss = torch.mean(-torch.logsumexp(log_prob, dim=1) + math.log(number_of_samples))
    print("epoch :", t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 50 == 0:
        cross, gts_div, sample_div = gen_energy_distance(t, number_pre_epochs, mean, log_diagonal, cov_factor, coords,
                                                         100, gts)
        ged = 2 * cross - gts_div - sample_div
        writer.add_scalar('cross', cross.item(), global_step=t)
        writer.add_scalar('gts_diversity', gts_div.item(), global_step=t)
        writer.add_scalar('sample_diversity', sample_div.item(), global_step=t)
        writer.add_scalar('GED', ged.item(), global_step=t)
        writer.add_scalar('Loss', loss.item(), global_step=t)
    del mc_samples
    del gt


def results_low_rank(mean_func, diagonal_func, cov_factor_func, img, coords, path):
    with torch.no_grad():
        mean, diagonal, cov_factor = function_eval(mean_func, diagonal_func, cov_factor_func, coords)
        columns, rows = 4, 3
        figsize = [40, 40]
        fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)
        reshape_size = int(math.sqrt(len(coords)))

        mc_samples = mc_sample_cov_is_low_rank(mean, diagonal, cov_factor, 15)
        for i, axi in enumerate(ax.flat):
            sample_i = torch.round(torch.sigmoid(mc_samples[i]))
            sample_i = sample_i.reshape(reshape_size, reshape_size)
            sample_i = sample_i.squeeze()
            img = img.squeeze()
            axi.imshow(img, cmap="gray")
            axi.imshow(sample_i, alpha=0.4)
            rowid = i // rows
            colid = i % columns
            axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
        plt.tight_layout()
        plt.plot()
        plt.savefig(path + "samples")


def results_low_rank_upsampled(mean_func, diagonal_func, cov_factor_func, coords, path):
    with torch.no_grad():
        mean, diagonal, cov_factor = function_eval(mean_func, diagonal_func, cov_factor_func, coords)
        columns, rows = 4, 3
        figsize = [40, 40]
        fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)
        reshape_size = int(math.sqrt(len(coords)))
        mc_samples = mc_sample_cov_is_low_rank(mean, diagonal, cov_factor, 15)
        for i, axi in enumerate(ax.flat):
            sample_i = torch.round(torch.sigmoid(mc_samples[i]))
            sample_i = sample_i.reshape(reshape_size, reshape_size)
            sample_i = sample_i.squeeze()
            axi.imshow(sample_i, alpha=0.4)
        plt.tight_layout()
        plt.plot()
        plt.savefig(path + "samples_upsampled")


def results_mean(mean_func, diagonal_func, cov_factor_func, coords, path):
    with torch.no_grad():
        mean, diagonal, cov_factor = function_eval(mean_func, diagonal_func, cov_factor_func, coords)
        figx = plt.figure(figsize=(40, 40))
        mean = mean.cpu().detach()
        reshape_size = int(math.sqrt(len(coords)))
        mean_prob = torch.round(torch.sigmoid(mean)).numpy()
        mean_prob = mean_prob.reshape(reshape_size, reshape_size)
        mean = mean.numpy().reshape(reshape_size, reshape_size)
        lim = np.max(np.abs(mean))
        figx.add_subplot(1, 2, 1)
        plt.imshow(mean_prob, cmap='seismic', clim=(-1, 1))
        cbar = plt.colorbar()
        cbar.minorticks_on()
        figx.add_subplot(1, 2, 2)
        plt.imshow(mean, cmap='seismic', clim=(-lim, lim))
        cbar = plt.colorbar()
        cbar.minorticks_on()
        plt.plot()
        plt.savefig(path + "mean")


def iou(x, y, axis=-1):
    iou_ = (x & y).sum(axis) / (x | y).sum(axis)
    iou_[torch.isnan(iou_)] = 1.
    return iou_


def distance(x, y):
    try:
        per_class_iou = iou(x[:, None], y[None, :], axis=-2)
    except MemoryError:
        per_class_iou = []
        for x_ in x:
            per_class_iou.append(iou(torch.unsqueeze(x_, dim=0), y[None, :], axis=-2))
        per_class_iou = torch.cat(per_class_iou)
    return 1 - per_class_iou[..., 1:].mean(-1)


def gen_energy_distance(t, number_pre_epochs, mean_func, diagonal_func, cov_factor_func, coords, sample_num, gts_list):
    with torch.no_grad():
        mean, diagonal, cov_factor = function_eval(mean_func, diagonal_func, cov_factor_func, coords)
        if t <= number_pre_epochs:
            samples = mc_sample_mean(mean, sample_num).to(device=DEVICE)
            samples = torch.round(torch.sigmoid(samples)).to(dtype=torch.long)
            samples = samples.reshape((len(samples), -1))
        else:
            samples = mc_sample_cov_is_low_rank(mean, diagonal, cov_factor, sample_num).to(device=DEVICE)
            samples = torch.round(torch.sigmoid(samples)).to(dtype=torch.long)
            samples = samples.reshape((len(samples), -1))
        gts = torch.stack(gts_list).to(dtype=torch.long)
        gts = gts.reshape((len(gts), -1))
        eye = torch.eye(2)
        gt_dist = eye[gts].to(dtype=bool)
        sample_dist = eye[samples].to(dtype=bool)
        gts_diversity = torch.mean(distance(gt_dist, gt_dist))
        sample_diversity = torch.mean(distance(sample_dist, sample_dist))
        cross = torch.mean(distance(sample_dist, gt_dist))
        del samples
        del gts
        return cross, gts_diversity, sample_diversity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str, help="model type")
    ap.add_argument("--lr", required=True, type=float, help="learning rate")
    model = ap.parse_args().model
    lr = ap.parse_args().lr
    latent_size = 64
    in_channel = 2
    rank = 10
    out_channel = 1
    gts_index = 355
    #image = dataset[gts_index][0]
    gt = ground_truths(gts_index, dataset)
    dimension = ground_truths(gts_index, dataset)[0].shape
    coordinates = get_coordinates(dimension, 1, 1).to(device=DEVICE)
    #coordinates_2 = get_coordinates(dimension, 2, 1).to(device=DEVICE)
    number_of_mc = 20
    pre_epochs = 20000
    number_epochs = 20000
    if model == "discrete":
        if lr == 1e-3:
            learning_rate = "_original"
        elif lr == 1e-4:
            learning_rate = "_slower"
        else:
            learning_rate = str(lr)
        writer = SummaryWriter('runs/lcdi/'+model+learning_rate)
        mean_function = Disc(128, 1).to(device=DEVICE)
        log_diagonal_function = Disc(128, 1).to(device=DEVICE)
        optimizer_mean = torch.optim.Adam(mean_function.parameters(), lr=lr)
        low_rank_factor_function = Disc(128, rank).to(device=DEVICE)
        parameters_all = [mean_function.parameters(), log_diagonal_function.parameters(),
                          low_rank_factor_function.parameters()]
        optimizer_all = torch.optim.Adam(itertools.chain(*parameters_all), lr=lr)
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
        for t in range(pre_epochs + number_epochs):
            train_low_rank(t, pre_epochs, gt, mean_function, log_diagonal_function, low_rank_factor_function, criterion,
                           optimizer_mean, optimizer_all, number_of_mc, coordinates, writer)
        writer.close()
        model_path = model+learning_rate+".pt"
        torch.save({
            'mean_function_state_dict': mean_function.state_dict(),
            'log_diagonal_function_state_dict': log_diagonal_function.state_dict(),
            'cov_factor_function_dict': low_rank_factor_function.state_dict(),
            'optimizer_mean_state_dict': optimizer_mean.state_dict(),
            'optimizer_all_state_dict': optimizer_all.state_dict(),
        }, model_path)

    elif model == "deepSDF":
        if lr == 1e-3:
            learning_rate = "_original"
        elif lr == 1e-4:
            learning_rate = "_slower"
        else:
            learning_rate = str(lr)
        writer = SummaryWriter('runs/lcdi/'+model+learning_rate)
        mean_function = DeepSDF(in_channel, latent_size, out_channel).to(device=DEVICE)
        log_diagonal_function = DeepSDF(in_channel, latent_size, out_channel).to(device=DEVICE)
        low_rank_factor_function = DeepSDF(in_channel, latent_size, rank).to(device=DEVICE)
        optimizer_mean = torch.optim.Adam(mean_function.parameters(), lr=lr)
        parameters_all = [mean_function.parameters(), log_diagonal_function.parameters(),
                          low_rank_factor_function.parameters()]
        optimizer_all = torch.optim.Adam(itertools.chain(*parameters_all), lr=lr)
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
        for t in range(pre_epochs + number_epochs):
            train_low_rank(t, pre_epochs, gt, mean_function, log_diagonal_function, low_rank_factor_function, criterion,
                           optimizer_mean, optimizer_all, number_of_mc, coordinates, writer)
        writer.close()
        model_path = model+learning_rate+".pt"
        torch.save({
            'mean_function_state_dict': mean_function.state_dict(),
            'log_diagonal_function_state_dict': log_diagonal_function.state_dict(),
            'cov_factor_function_dict': low_rank_factor_function.state_dict(),
            'optimizer_mean_state_dict': optimizer_mean.state_dict(),
            'optimizer_all_state_dict': optimizer_all.state_dict(),
        }, model_path)

    else:
        print("wrong argument")


if __name__ == '__main__':
    main()
