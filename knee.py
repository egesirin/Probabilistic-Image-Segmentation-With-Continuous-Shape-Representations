import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
import itertools
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class KneeDataSet(Dataset):
    def __init__(self, annotations_dir, transform=None):
        self.label_dir = annotations_dir
        self.labels = os.listdir(self.label_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.labels[idx])
        label = np.load(label_path)
        if self.transform:
            label = self.transform(label)
        label = torch.from_numpy(label)
        return label.float()


def ground_truths(dataset, n):
    gt_list = []
    for i in range(n):
        gt_list.append(dataset[i].to(device=DEVICE))
    return gt_list


def get_coordinates(image_shape: Iterable[int]) -> torch.Tensor:
    individual_voxel_ids = [torch.arange(num_elements) for num_elements in image_shape]
    individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')
    voxel_ids = torch.stack(individual_voxel_ids_meshed, -1)
    voxel_ids = voxel_ids.reshape(-1, voxel_ids.shape[-1])
    voxel_ids = voxel_ids.to(torch.float32)
    return voxel_ids


def function_eval(mean_func, log_diagonal_func, cov_factor_func, coords):
    mu = 159.5
    sigma = 92.52
    coords = (coords - mu) / sigma
    mean_vector = mean_func(coords)
    diagonal_matrix = torch.exp(log_diagonal_func(coords).view(-1))
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
        self.fc4 = nn.Linear(256, 256-(in_ch+latent_size))
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

    def forward(self):
        return self.latent_parameter



def train_low_rank(t, number_pre_epochs, gts, mean, log_diagonal, cov_factor, loss_function, optimizer_m, optimizer_a,
                   number_of_samples, coords, writer):
    mean_vec, diagonal_vec, cov_factor_matrix = function_eval(mean, log_diagonal, cov_factor, coords)
    log_prob = torch.zeros(len(gts), number_of_samples)
    if t <= number_pre_epochs:
        optimizer = optimizer_m
        for i in range(len(gts)):
            mc_samples = mc_sample_mean(mean_vec, number_of_samples).to(device=DEVICE)
            for j in range(number_of_samples):
                log_prob[i][j] = -loss_function(mc_samples[j], gts[i].view(-1)).to(device=DEVICE)
        del mc_samples
        loss = torch.mean(-torch.logsumexp(log_prob, dim=1) + math.log(number_of_samples))
        print("epoch :", t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Training Losses/Epoch', loss.item(), global_step=t)
    else:
        optimizer = optimizer_a
        for i in range(len(gts)):
            mc_samples = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix,
                                                   number_of_samples).to(device=DEVICE)
            for j in range(number_of_samples):
                log_prob[i][j] = -loss_function(mc_samples[j], gts[i].view(-1)).to(device=DEVICE)
        del mc_samples
        loss = torch.mean(-torch.logsumexp(log_prob, dim=1) + math.log(number_of_samples))
        print("epoch :", t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Training Losses/Epoch', loss.item(), global_step=t)


def results_low_rank(mean_func, diagonal_func, cov_factor_func, coords, path, res):
    with torch.no_grad():
        mean, diagonal, cov_factor = function_eval(mean_func, diagonal_func, cov_factor_func, coords)
        resize_num = int(math.sqrt(len(coords)))
        columns, rows = 2, 3
        figsize = [40, 40]
        fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)
        mc_samples = mc_sample_cov_is_low_rank(mean, diagonal, cov_factor, columns*rows)
        for i, axi in enumerate(ax.flat):
            sample_i = torch.round(torch.sigmoid(mc_samples[i]))
            sample_i = sample_i.reshape(resize_num, resize_num)
            sample_i = sample_i.squeeze()
            axi.imshow(sample_i)
            rowid = i // rows
            colid = i % columns
            axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
        plt.tight_layout()
        plt.plot()
        plt.savefig(path + res)


def results_mean(mean_func, diagonal_func, cov_factor_func, coords, path):
    with torch.no_grad():
        mean, diagonal, cov_factor = function_eval(mean_func, diagonal_func, cov_factor_func, coords)
        figx = plt.figure(figsize=(40, 40))
        mean = mean.cpu().detach()
        mean_prob = torch.round(torch.sigmoid(mean)).numpy()
        mean_prob = mean_prob.reshape(320, 320)
        mean = mean.numpy().reshape(320, 320)
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


def gt_show(ground_truth, path ,res):
    with torch.no_grad():
        columns, rows = 2, 2
        figsize = [40, 40]
        fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)

        for i, axi in enumerate(ax.flat):
            axi.imshow(ground_truth[i].cpu().detach())
            rowid = i // rows
            colid = i % columns
            axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
        plt.tight_layout()
        plt.plot()
        plt.savefig(path + res)


def main():
    writer = SummaryWriter('runs/kneeloss')
    knee_data_folder = '/scratch/visual/esirin/data/label_slices/'
    knee_dataset = KneeDataSet(knee_data_folder)
    gt = ground_truths(knee_dataset, 4)
    coordinates = get_coordinates(gt[0].size()).to(device=DEVICE)
    gt_show(gt, '/srv/public/esirin/results/model_deepsdf/knee/', 'gts')
    latent_size = 64
    in_channel = 2
    rank = 10
    out_channel = 1
    learning_rate = 1e-4
    number_of_mc = 20
    pre_epochs = 20000
    number_epochs = 20000
    mean_function = DeepSDF(in_channel, latent_size, out_channel).to(device=DEVICE)
    log_diagonal_function = DeepSDF(in_channel, latent_size, out_channel).to(device=DEVICE)
    low_rank_factor_function = DeepSDF(in_channel, latent_size,  rank).to(device=DEVICE)
    optimizer_mean = torch.optim.Adam(mean_function.parameters(), lr=learning_rate)
    parameters_all = [mean_function.parameters(), log_diagonal_function.parameters(),
                      low_rank_factor_function.parameters()]
    optimizer_all = torch.optim.Adam(itertools.chain(*parameters_all), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    PATH = '/srv/public/esirin/results/model_deepsdf/knee/'
    for t in range(pre_epochs + number_epochs):
        train_low_rank(t, pre_epochs, gt, mean_function, log_diagonal_function, low_rank_factor_function,
                       criterion, optimizer_mean, optimizer_all, number_of_mc, coordinates, writer)
    writer.close()
    torch.cuda.empty_cache()
    results_mean(mean_function, log_diagonal_function, low_rank_factor_function, coordinates, PATH)
    results_low_rank(mean_function, log_diagonal_function, low_rank_factor_function, coordinates, PATH, "samples1")
    results_low_rank(mean_function, log_diagonal_function, low_rank_factor_function, coordinates, PATH, "samples2")


if __name__ == '__main__':
    main()
