import torch
from lidcdata import LIDC_IDRI
from typing import Iterable
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt



data_folder = '/scratch/visual/esirin/data/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location=data_folder)

def ground_truths(n,data):
	gts = data[n][1]
	return gts

def get_coordinates(image_shape: Iterable[int]) -> torch.Tensor:
	individual_voxel_ids = [torch.arange(num_elements) for num_elements in image_shape]
	individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')
	voxel_ids = torch.stack(individual_voxel_ids_meshed, -1)
	voxel_ids = voxel_ids.reshape(-1, voxel_ids.shape[-1])
	voxel_ids = voxel_ids.to(torch.float32)
	return voxel_ids


def from_funct_to_matrix_cov_is_diagonal(mean_func, log_diagonal_func, coords):
	mu = 63.5
	sigma = 37.09
	coords = (coords- mu)/sigma
	mean_vector = mean_func(coords)
	diagonal_matrix = torch.diag(torch.exp(log_diagonal_func(coords).view(-1)))
	return mean_vector, diagonal_matrix


def from_funct_to_matrix_cov_is_low_rank(mean_func, log_diagonal_func, cov_factor_func, coords):
	mu = 63.5
	sigma = 37.09
	coords = (coords - mu) / sigma
	mean_vector = mean_func(coords)
	diagonal_matrix = torch.diag(torch.exp(log_diagonal_func(coords).view(-1)))
	cov_factor_matrix = cov_factor_func(coords)

	return mean_vector, diagonal_matrix, cov_factor_matrix


def mc_sample_mean(mean, number_of_sample):
	mc_samples = torch.rand(number_of_sample, mean.size(0))
	for i in range(number_of_sample):
		mc_samples[i] = mean.view(-1)
	return mc_samples


def mc_sample_cov_is_diagonal(mean, diagonal, number_of_sample):
	mc_samples = torch.zeros(number_of_sample, diagonal.size(0))
	for i in range(number_of_sample):
		sample_d = torch.normal(0, 1, size=(diagonal.size(0),))
		mc_samples[i] = torch.sqrt(diagonal) @ sample_d + mean.view(-1)
	return mc_samples


def mc_sample_cov_is_low_rank(mean, diagonal, cov_factor, number_of_sample):
	mc_samples = torch.rand(number_of_sample, diagonal.size(0))
	for i in range(number_of_sample):
		sample_d = torch.normal(0, 1, size=(diagonal.size(0),)).to(device=DEVICE)
		sample_p = torch.normal(0, 1, size=(cov_factor.size(1),)).to(device=DEVICE)
		mc_samples[i] = (cov_factor @ sample_p + torch.sqrt(diagonal) @ sample_d + mean.view(-1))
	return mc_samples


class DeepSDF(nn.Module):
	def __init__(self, in_ch, latent_size, dim, out_ch):
		super().__init__()
		self.latent_parameter = torch.nn.Parameter(torch.normal(0.0, 1e-4, [1, latent_size]).repeat(dim**in_ch, 1),
												   requires_grad=True)
		self.fc1 = nn.Linear(in_ch + latent_size, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 256)
		self.fc4 = nn.Linear(256, 256-(in_ch+latent_size))
		self.fc5 = nn.Linear(256, 256)
		self.fc6 = nn.Linear(256, 256)
		self.fc7 = nn.Linear(256, 256)
		self.fc8 = nn.Linear(256, out_ch)

	def forward(self, x):
		x1 = torch.cat((x, self.latent_parameter), 1)
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



def train_low_rank(t, number_pre_epochs, gts, mean, log_diagonal, cov_factor, loss_function, optimizer_m, optimizer_a,
				   number_of_samples, coords):
	gts0 = gts[0].to(device=DEVICE)
	gts1 = gts[1].to(device=DEVICE)
	gts2 = gts[2].to(device=DEVICE)
	gts3 = gts[3].to(device=DEVICE)
	log_prob = torch.zeros(4, number_of_samples)
	if t <= number_pre_epochs:
		optimizer = optimizer_m
		mean_vec, diagonal_vec, cov_factor_matrix = from_funct_to_matrix_cov_is_low_rank(mean, log_diagonal,
																						 cov_factor, coords)

		mc_samples0 = mc_sample_mean(mean_vec, number_of_samples).to(device=DEVICE)
		mc_samples1 = mc_sample_mean(mean_vec, number_of_samples).to(device=DEVICE)
		mc_samples2 = mc_sample_mean(mean_vec, number_of_samples).to(device=DEVICE)
		mc_samples3 = mc_sample_mean(mean_vec, number_of_samples).to(device=DEVICE)
	else:
		optimizer = optimizer_a
		mean_vec, diagonal_vec, cov_factor_matrix = from_funct_to_matrix_cov_is_low_rank(mean, log_diagonal,
																						 cov_factor, coords)
		mc_samples0 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples).to(
			device=DEVICE)
		mc_samples1 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples).to(
			device=DEVICE)
		mc_samples2 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples).to(
			device=DEVICE)
		mc_samples3 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples).to(
			device=DEVICE)


	for j in range(number_of_samples):
		log_prob[0][j] = -loss_function(mc_samples0[j], gts0.view(-1)).to(device=DEVICE)
		log_prob[1][j] = -loss_function(mc_samples1[j], gts1.view(-1)).to(device=DEVICE)
		log_prob[2][j] = -loss_function(mc_samples2[j], gts2.view(-1)).to(device=DEVICE)
		log_prob[3][j] = -loss_function(mc_samples3[j], gts3.view(-1)).to(device=DEVICE)
	loss = torch.mean(-torch.logsumexp(log_prob, dim=1) + math.log(number_of_samples)).to(device=DEVICE)
	print("epoch :",t, loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()




def results_low_rank(mean_func, diagonal_func, cov_factor_func, img, coords, path):
	with torch.no_grad():
		mean, diagonal, cov_factor = from_funct_to_matrix_cov_is_low_rank(mean_func, diagonal_func, cov_factor_func,
																		  coords)
		columns, rows = 4,3
		figsize = [40,40]
		fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)

		mc_samples = mc_sample_cov_is_low_rank(mean, diagonal, cov_factor, 80)
		for i, axi in enumerate(ax.flat):
			sample_i = torch.round(torch.sigmoid(mc_samples[i]))
			sample_i = sample_i.reshape(128, 128)
			sample_i = sample_i.squeeze()
			img = img.squeeze()
			axi.imshow(img,  cmap="gray")
			axi.imshow(sample_i, alpha=0.4)
			rowid = i // rows
			colid = i % columns
			axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
		plt.tight_layout()
		plt.plot()
		plt.savefig(path + "samples")


def results_mean(mean_func, diagonal_func, cov_factor_func, coords, path):
	with torch.no_grad():
		mean, diagonal, cov_factor = from_funct_to_matrix_cov_is_low_rank(mean_func, diagonal_func, cov_factor_func,
																		  coords)
		mean = mean.reshape(128, 128)
		mean = mean.cpu().detach().numpy()
		lim = np.max(np.abs(mean))
		plt.imshow(mean, cmap='seismic', clim=(-lim, lim))
		cbar = plt.colorbar()
		cbar.minorticks_on()
		plt.plot()
		plt.savefig(path + "mean")



latent_size = 64
in_channel = 2
rank= 10
out_channel = 1
gts_index = 69
image = dataset[gts_index][0]
gt = ground_truths(gts_index, dataset)
dimension = ground_truths(gts_index, dataset)[0].shape
coordinates = get_coordinates(dimension).to(device=DEVICE)
learning_rate = 1e-3
number_of_mc = 20
pre_epochs = 10
number_epochs = 3
#print(coordinates.shape)
#print(coordinates[0])
#print(coordinates[16383])
mean_function = DeepSDF(in_channel,latent_size, 128, out_channel).to(device=DEVICE)
log_diagonal_function = DeepSDF(in_channel,latent_size, 128, out_channel).to(device=DEVICE)
low_rank_factor_function = DeepSDF(in_channel,latent_size, 128, rank).to(device=DEVICE)
optimizer_mean = torch.optim.Adam(mean_function.parameters(), lr=learning_rate)
parameters_all = [mean_function.parameters(), log_diagonal_function.parameters(), low_rank_factor_function.parameters()]
optimizer_all = torch.optim.Adam(itertools.chain(*parameters_all), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss(reduction="sum")
PATH =  '/scratch/visual/esirin/toy_problem/results/2d_lowrank_deepsdf/'

def main():

	for t in range(pre_epochs+number_epochs):
		train_low_rank(t, pre_epochs,gt, mean_function, log_diagonal_function, low_rank_factor_function, criterion,
					   optimizer_mean, optimizer_all, number_of_mc, coordinates)

	results_mean(mean_function, log_diagonal_function, low_rank_factor_function, coordinates, PATH)
	results_low_rank(mean_function, log_diagonal_function, low_rank_factor_function, image, coordinates, PATH)



if __name__ == '__main__':
	main()


