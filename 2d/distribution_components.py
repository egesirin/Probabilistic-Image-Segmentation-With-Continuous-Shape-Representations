import torch.nn as nn
import torch


class DistributionComponents(nn.Module):
    def __init__(self, feature_size, rank, hidden_dim=256):
        super(DistributionComponents, self).__init__()
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, rank, 1)
        self.actvn = nn.ReLU()

    def forward(self, x):
        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))
        x = self.actvn(self.fc_2(x))
        x = self.fc_out(x)
        return x


def distribution_components(model_outputs, model_type):
    if model_type == "m_p_d":
        mean_vector, low_rank_factor, log_diagonal_vector = model_outputs
        low_rank_factor = low_rank_factor.transpose(1, -1)  # torch.Size([batch,  #pixels, rank])
        diagonal_vector = torch.exp(log_diagonal_vector).transpose(1, -1)
        return mean_vector, low_rank_factor, diagonal_vector
    elif model_type == "m_p":
        mean_vector, low_rank_factor = model_outputs
        low_rank_factor = low_rank_factor.transpose(1, -1)  # torch.Size([batch,  #pixels, rank])
        return mean_vector, low_rank_factor
    elif model_type == "m":
        mean_vector = model_outputs[0]  # I have to add [0],No idea why I should add it only here
        return mean_vector
    else:
        raise NotImplementedError


'''   
shape_of_functions
mean = torch.Size(batch_size, 1, number_of_pixels]) = torch.Size([12, 1, 16384])
low_rank_factor = torch.Size(batch_size, rank, number_of_pixels]) = torch.Size([12, 10, 16384])
diagonal = torch.Size(batch_size, 1, number_of_pixels]) = torch.Size([12, 1, 16384])
'''