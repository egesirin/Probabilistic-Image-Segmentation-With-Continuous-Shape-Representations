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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location=data_folder)

def ground_truths(n,data):
	gts = data[n][1]
	return gts

def get_coordinates(image_shape: Iterable[int]) -> torch.Tensor:
	epsilon = 1e-4
	individual_voxel_ids = [torch.arange(num_elements) for num_elements in image_shape]
	individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')
	voxel_ids = torch.stack(individual_voxel_ids_meshed, -1)
	voxel_ids = voxel_ids.reshape(-1, voxel_ids.shape[-1])
	voxel_ids = voxel_ids.to(torch.float32)*epsilon
	return voxel_ids



def from_funct_to_matrix_mean(mean_func, coords):
	mean_vector = mean_func(coords)
	return mean_vector


def from_funct_to_matrix_cov_is_diagonal(mean_func, log_diagonal_func, coords):
	mean_vector = mean_func(coords)
	diagonal_matrix = torch.diag(torch.exp(log_diagonal_func(coords).view(-1)))
	return mean_vector, diagonal_matrix


def from_funct_to_matrix_cov_is_low_rank(mean_func, log_diagonal_func, cov_factor_func, coords):
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
		sample_d = torch.normal(0, 1, size=(diagonal.size(0),))
		sample_p = torch.normal(0, 1, size=(cov_factor.size(1),))
		mc_samples[i] = cov_factor @ sample_p + torch.sqrt(diagonal) @ sample_d + mean.view(-1)
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

	log_prob = torch.zeros(4, number_of_samples)
	if t <= number_pre_epochs:
		optimizer = optimizer_m
		mean_vec, diagonal_vec, cov_factor_matrix = from_funct_to_matrix_cov_is_low_rank(mean, log_diagonal,
																						 cov_factor, coords)
		mc_samples0 = mc_sample_mean(mean_vec, number_of_samples)
		mc_samples1 = mc_sample_mean(mean_vec, number_of_samples)
		mc_samples2 = mc_sample_mean(mean_vec, number_of_samples)
		mc_samples3 = mc_sample_mean(mean_vec, number_of_samples)

	else:
		optimizer = optimizer_a
		mean_vec, diagonal_vec, cov_factor_matrix = from_funct_to_matrix_cov_is_low_rank(mean, log_diagonal,
																						 cov_factor, coords)
		mc_samples0 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples)
		mc_samples1 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples)
		mc_samples2 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples)
		mc_samples3 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples)

	for j in range(number_of_samples):
		log_prob[0][j] = -loss_function(mc_samples0[j], gts[0].view(-1))
		log_prob[1][j] = -loss_function(mc_samples1[j], gts[1].view(-1))
		log_prob[2][j] = -loss_function(mc_samples2[j], gts[2].view(-1))
		log_prob[3][j] = -loss_function(mc_samples3[j], gts[3].view(-1))
	loss = torch.mean(-torch.logsumexp(log_prob, dim=1) + math.log(number_of_samples))
	print("epoch :",t, loss.item())

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

def show_cov_matrix(cov, path, str):
	cov = cov.detach().numpy()
	lim = np.max(np.abs(cov))
	plt.imshow(cov, cmap='seismic', clim=(-lim, lim))
	cbar = plt.colorbar()
	cbar.minorticks_on()
	plt.show()
	plt.savefig(path + str)


def my_plot(t, loss):
	plt.plot(t, loss)
	plt.show()


def results_low_rank(mean_func, diagonal_func, cov_factor_func, img, coords, path):
	with torch.no_grad():
		fig = plt.figure(figsize=(40, 40), constrained_layout=False)
		mean, diagonal, cov_factor = from_funct_to_matrix_cov_is_low_rank(mean_func, diagonal_func, cov_factor_func,
																		  coords)
		mc_samples = mc_sample_cov_is_low_rank(mean, diagonal, cov_factor, 14)
		for i in range(4):
			sample_i = torch.round(torch.sigmoid(mc_samples[i]))
			sample_i = sample_i.reshape(128,128)
			sample_i = sample_i.squeeze()
			img = img.squeeze()
			fig.add_subplot(1, 4, i+1)
			plt.imshow(img, cmap="gray")
			plt.imshow(sample_i, alpha=0.4)
	plt.plot()
	plt.savefig(path+"samples")

def results_mean(mean_func, diagonal_func, cov_factor_func, coords, path, str):
	with torch.no_grad():
		mean, diagonal, cov_factor = from_funct_to_matrix_cov_is_low_rank(mean_func, diagonal_func, cov_factor_func,
																		  coords)
		show_cov_matrix(mean,path, str)






#latent_size = 2
latent_size = 64
in_channel = 2
rank= 10
out_channel = 1
gts_index = 69
image = dataset[gts_index][0]
gt = ground_truths(gts_index, dataset)
dimension = ground_truths(gts_index, dataset)[0].shape
coordinates = get_coordinates(dimension)
learning_rate = 1e-3
number_of_mc = 10
pre_epochs = 5000
number_epochs = 5000

#print(coordinates.shape)
#print(coordinates[0])
#print(coordinates[16383])
mean_function = DeepSDF(in_channel,latent_size, 128, out_channel)
log_diagonal_function = DeepSDF(in_channel,latent_size, 128, out_channel)
low_rank_factor_function = DeepSDF(in_channel,latent_size, 128, rank)
optimizer_mean = torch.optim.Adam(mean_function.parameters(), lr=learning_rate)
parameters_all = [mean_function.parameters(), log_diagonal_function.parameters(), low_rank_factor_function.parameters()]
optimizer_all = torch.optim.Adam(itertools.chain(*parameters_all), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss(reduction="sum")
PATH =  '/scratch/visual/esirin/toy_problem/results/2d_lowrank_deepsdf/'

def main():

	for t in range(pre_epochs+number_epochs):
		train_low_rank(t, pre_epochs,gt, mean_function, log_diagonal_function, low_rank_factor_function, criterion,
					   optimizer_mean, optimizer_all, number_of_mc, coordinates)

	results_low_rank(mean_function, log_diagonal_function, low_rank_factor_function, image, coordinates, PATH)


if __name__ == '__main__':
	main()


