import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools


def my_plot(t, loss):
    plt.plot(t, loss)
    plt.show()


def show_cov_matrix(cov):
    cov = cov.detach().numpy()
    lim = np.max(np.abs(cov))
    plt.imshow(cov, cmap='seismic', clim=(-lim, lim))
    cbar = plt.colorbar()
    cbar.minorticks_on()
    plt.show()


'''
def gt(dim):
    gt_type_a = torch.ones(1, dim)
    gt_type_a[0][7:14] = 0
    gt_type_b = torch.zeros(1, dim)
    gt_type_b[0][:7] = 1
    ground_truth = torch.cat((gt_type_a, gt_type_b), dim=0)
    return ground_truth


'''


def gt(dim):
    gt_type_a = torch.zeros(1, dim)
    gt_type_a[0][:7] = 1
    gt_type_b = torch.zeros(1, dim)
    gt_type_b[0][:14] = 1
    ground_truth = torch.cat((gt_type_a, gt_type_b), dim=0)
    return ground_truth


def from_funct_to_matrix_cov_is_diagonal(mean_func, log_diagonal_func, dim):
    #epsilon = 1e-4
    #index = torch.arange(-dim//2, dim//2).to(torch.float32)*epsilon
    index = torch.arange(dim).to(torch.float32)
    index = index.unsqueeze(1)
    mean_vec = mean_func(index)
    diagonal_matrix = torch.diag(torch.exp(log_diagonal_func(index).view(-1)))
    return mean_vec, diagonal_matrix


def from_funct_to_matrix_cov_is_low_rank(mean_func, log_diagonal_func, cov_factor_func, dim):
    #epsilon = 1e-4
    #index = torch.arange(-dim//2, dim//2).to(torch.float32)*epsilon
    index = torch.arange(dim).to(torch.float32)
    index = index.unsqueeze(1)
    mean_vector = mean_func(index)
    diagonal_matrix = torch.diag(torch.exp(log_diagonal_func(index).view(-1)))
    cov_factor_matrix = cov_factor_func(index)
    return mean_vector, diagonal_matrix, cov_factor_matrix


def mc_sample_mean(mean, dim, number_of_sample):
    mc_samples = torch.rand(number_of_sample, dim)
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


class Disc(nn.Module):
    def __init__(self, dim, out_ch):
        super().__init__()
        self.latent_parameter = torch.nn.Parameter(torch.normal(0.0, 1e-4, [dim, out_ch]), requires_grad=True)

    def forward(self, x):
        x = self.latent_parameter
        return x


class DeepSDF(nn.Module):
    def __init__(self, in_ch, latent_size, dim, out_ch):
        super().__init__()
        self.latent_parameter = torch.nn.Parameter(torch.normal(0.0, 1e-4, [in_ch, latent_size]).repeat(dim, 1),
                                                   requires_grad=True)
        self.fc1 = nn.Linear(in_ch + latent_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32 - latent_size)
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, 32)
        self.fc8 = nn.Linear(32, out_ch)

    def forward(self, x):
        x1 = torch.cat((x, self.latent_parameter), 1)
        x2 = F.relu(self.fc1(x1))
        x3 = F.relu(self.fc2(x2))
        x4 = F.relu(self.fc3(x3))
        x5 = F.relu(self.fc4(x4))
        y = torch.cat((self.latent_parameter, x5), 1)
        x6 = F.relu(self.fc5(y))
        x7 = F.relu(self.fc6(x6))
        x8 = F.relu(self.fc7(x7))
        x9 = self.fc8(x8)
        return x9


def train_diagonal(t, number_pre_epochs, mean, log_diagonal, loss_function, optimizer_m, optimizer_d_m,
                   number_of_samples, dim, loss_list):
    log_prob = torch.zeros(2, number_of_samples)
    gts = gt(dim)
    if t <= number_pre_epochs:
        optimizer = optimizer_m
        mean_vec, diagonal_vec = from_funct_to_matrix_cov_is_diagonal(mean, log_diagonal, dim)
        mc_samples0 = mc_sample_mean(mean_vec, dim, number_of_samples)
        mc_samples1 = mc_sample_mean(mean_vec, dim, number_of_samples)
    else:
        optimizer = optimizer_d_m
        mean_vec, diagonal_vec = from_funct_to_matrix_cov_is_diagonal(mean, log_diagonal, dim)
        mc_samples0 = mc_sample_cov_is_diagonal(mean_vec, diagonal_vec, number_of_samples)
        mc_samples1 = mc_sample_cov_is_diagonal(mean_vec, diagonal_vec, number_of_samples)

    for j in range(number_of_samples):
        log_prob[0][j] = -loss_function(mc_samples0[j], gts[0])
        log_prob[1][j] = -loss_function(mc_samples1[j], gts[1])
    loss = torch.mean(-torch.logsumexp(log_prob, dim=1)) + math.log(number_of_samples)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_low_rank(t, number_pre_epochs, mean, log_diagonal, cov_factor, loss_function, optimizer_m, optimizer_a,
                   number_of_samples, dim, loss_list):

    log_prob = torch.zeros(2, number_of_samples)
    gts = gt(dim)
    if t <= number_pre_epochs:
        optimizer = optimizer_m
        mean_vec, diagonal_vec, cov_factor_matrix = from_funct_to_matrix_cov_is_low_rank(mean, log_diagonal,
                                                                                         cov_factor, dim)
        mc_samples0 = mc_sample_mean(mean_vec, dim, number_of_samples)
        mc_samples1 = mc_sample_mean(mean_vec, dim, number_of_samples)

    else:
        optimizer = optimizer_a
        mean_vec, diagonal_vec, cov_factor_matrix = from_funct_to_matrix_cov_is_low_rank(mean, log_diagonal,
                                                                                         cov_factor, dim)
        mc_samples0 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples)
        mc_samples1 = mc_sample_cov_is_low_rank(mean_vec, diagonal_vec, cov_factor_matrix, number_of_samples)
    for j in range(number_of_samples):
        log_prob[0][j] = -loss_function(mc_samples0[j], gts[0])
        log_prob[1][j] = -loss_function(mc_samples1[j], gts[1])
    loss = torch.mean(-torch.logsumexp(log_prob, dim=1) + math.log(number_of_samples))
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def intermediate_results_diagonal(mean_func, diagonal_func, dim):
    with torch.no_grad():
        mean, diagonal = from_funct_to_matrix_cov_is_diagonal(mean_func, diagonal_func, dim)
        print("mean : ", mean.view(-1))
        print("diagonal : ", torch.diagonal(diagonal, 0))
        mean_prob = torch.sigmoid(mean)
        print("mean_prob : ", mean_prob.view(-1))


def results_diagonal(mean_func, diagonal_func, dim):
    with torch.no_grad():
        mean, diagonal = from_funct_to_matrix_cov_is_diagonal(mean_func, diagonal_func, dim)
        print("mean : ", mean.view(-1))
        print("diagonal : ", torch.diagonal(diagonal, 0))
        show_cov_matrix(diagonal)
        show_cov_matrix(mean.unsqueeze(1))
        mean_prob = torch.sigmoid(mean)
        print("mean_prob : ", mean_prob.view(-1))
        mc_samples = mc_sample_cov_is_diagonal(mean, diagonal, 14)
        for i in range(14):
            sample_i = torch.round(torch.sigmoid(mc_samples[i])).unsqueeze(0)
            show_cov_matrix(sample_i)


def intermediate_results_low_rank(mean_func, diagonal_func, cov_factor_func, dim):
    with torch.no_grad():
        mean, diagonal, cov_factor = from_funct_to_matrix_cov_is_low_rank(mean_func, diagonal_func, cov_factor_func,
                                                                          dim)
        print("mean : ", mean.view(-1))
        #cov = (cov_factor @ cov_factor_func.T) + diagonal
        #print("covariance: ", cov)
        mean_prob = torch.sigmoid(mean)
        print("mean_prob : ", mean_prob.view(-1))


def results_low_rank(mean_func, diagonal_func, cov_factor_func, dim):
    with torch.no_grad():
        mean, diagonal, cov_factor = from_funct_to_matrix_cov_is_low_rank(mean_func, diagonal_func, cov_factor_func,
                                                                          dim)
        print("mean : ", mean.view(-1))
        show_cov_matrix(mean.unsqueeze(1))
        cov = (cov_factor @ cov_factor.T) + diagonal
        print("covariance: ", cov)
        show_cov_matrix(cov)
        mean_prob = torch.sigmoid(mean)
        print("mean_prob : ", mean_prob.view(-1))
        mc_samples = mc_sample_cov_is_low_rank(mean, diagonal, cov_factor, 14)
        for i in range(14):
            sample_i = torch.round(torch.sigmoid(mc_samples[i])).unsqueeze(0)
            show_cov_matrix(sample_i)


def main(covariance, model):
    dimension = 21
    rank_of_cov_factor = 2
    latent = 4
    learning_rate = 1e-3
    loss_values = []
    number_of_mc = 200
    pre_epochs = 5000
    number_epochs = 5000
    #continuous + diagonal
    mean_function = DeepSDF(1, latent, dimension, 1)
    log_diagonal_function = DeepSDF(1, latent, dimension, 1)
    optimizer_mean = torch.optim.Adam(mean_function.parameters(), lr=1e-4)
    parameters_mean_diagonal = [mean_function.parameters(), log_diagonal_function.parameters()]
    optimizer_mean_diagonal = torch.optim.Adam(itertools.chain(*parameters_mean_diagonal), lr=1e-4)
    #continuous + low rank
    cov_factor_function = DeepSDF(1, latent, dimension, rank_of_cov_factor)
    parameters_all = [mean_function.parameters(), log_diagonal_function.parameters(), cov_factor_function.parameters()]
    optimizer_all = torch.optim.Adam(itertools.chain(*parameters_all), lr=1e-4)
    #discrete + diagonal
    mean_disc = Disc(dimension, 1)
    log_diagonal_disc = Disc(dimension, 1)
    optimizer_mean_disc = torch.optim.Adam(mean_disc.parameters(), lr=learning_rate)
    parameters_mean_diagonal_disc = [mean_disc.parameters(), log_diagonal_disc.parameters()]
    optimizer_mean_diagonal_disc = torch.optim.Adam(itertools.chain(*parameters_mean_diagonal_disc), lr=learning_rate)
    #discrete + lowrank
    cov_factor_disc = Disc(dimension, rank_of_cov_factor)
    parameters_all_disc = [mean_disc.parameters(), log_diagonal_disc.parameters(), cov_factor_disc.parameters()]
    optimizer_all_disc = torch.optim.Adam(itertools.chain(*parameters_all_disc), lr=learning_rate)

    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    gts = gt(dimension)
    print("gt : ", gts)
    if (covariance == "diagonal") and (model == "DeepSDF"):
        print(covariance)
        print(model)
        for t in range(pre_epochs+number_epochs):
            train_diagonal(t, pre_epochs, mean_function, log_diagonal_function, criterion, optimizer_mean,
                           optimizer_mean_diagonal, number_of_mc, dimension, loss_values)
            #print("epoch :", t,"loss: ", loss_values[-1])
            if (t % 500) == 1:
                print(t)
                print(covariance)
                print(model)
                print("int_res_di")
                intermediate_results_diagonal(mean_function, log_diagonal_function, dimension)
        results_diagonal(mean_function, log_diagonal_function, dimension)

    elif (covariance == "diagonal") and (model == "Disc"):
        print(covariance)
        print(model)
        for t in range(pre_epochs+number_epochs):
            train_diagonal(t, pre_epochs, mean_disc, log_diagonal_disc, criterion, optimizer_mean_disc,
                           optimizer_mean_diagonal_disc, number_of_mc, dimension, loss_values)

            if (t % 500) == 1:
                print(t)
                print(covariance)
                print(model)
                print("int_res_di")
                intermediate_results_diagonal(mean_disc, log_diagonal_disc, dimension)
        results_diagonal(mean_disc, log_diagonal_disc, dimension)
    elif (covariance == "low_rank") and (model == "DeepSDF"):
        print(covariance)
        print(model)
        for t in range(pre_epochs+number_epochs):
            train_low_rank(t, pre_epochs, mean_function, log_diagonal_function, cov_factor_function, criterion,
                           optimizer_mean, optimizer_all, number_of_mc, dimension, loss_values)
            if (t % 500) == 1:
                print(t)
                print(covariance)
                print(model)
                intermediate_results_low_rank(mean_function, log_diagonal_function, cov_factor_function, dimension)
        results_low_rank(mean_function, log_diagonal_function, cov_factor_function, dimension)

    elif (covariance == "low_rank") and (model == "Disc"):
        print(covariance)
        print(model)
        for t in range(pre_epochs+number_epochs):
            train_low_rank(t, pre_epochs, mean_disc, log_diagonal_disc, cov_factor_disc, criterion,
                           optimizer_mean_disc, optimizer_all_disc, number_of_mc, dimension, loss_values)
            if (t % 250) == 1:
                print(t)
                intermediate_results_low_rank(mean_disc, log_diagonal_disc, cov_factor_disc, dimension)
        results_low_rank(mean_disc, log_diagonal_disc, cov_factor_disc, dimension)

    else:
        raise NotImplementedError
    my_plot(np.linspace(1, pre_epochs+number_epochs, pre_epochs+number_epochs).astype(int), loss_values)


if __name__ == '__main__':
    #main("diagonal", "DeepSDF")
    main("low_rank", "DeepSDF")
