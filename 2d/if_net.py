import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ShapeNet128Vox_Low_Rank(nn.Module):

    def __init__(self,rank, hidden_dim=256):
        super(ShapeNet128Vox_Low_Rank, self).__init__()
        # accepts 128**3 res input
        self.conv_in = nn.Conv2d(1, 16, 3, padding=1)  # out: 128
        self.conv_0 = nn.Conv2d(16, 32, 3, padding=1)  # out: 64
        self.conv_0_1 = nn.Conv2d(32, 32, 3, padding=1)  # out: 64
        self.conv_1 = nn.Conv2d(32, 64, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv2d(64, 64, 3, padding=1)  # out: 32
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)  # out: 16
        self.conv_2_1 = nn.Conv2d(128, 128, 3, padding=1)  # out: 16
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)  # out: 8
        self.conv_3_1 = nn.Conv2d(128, 128, 3, padding=1)  # out: 8

        feature_size = (1 + 16 + 32 + 64 + 128 + 128) * 5
        self.d_fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.d_fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.d_fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.d_fc_out = nn.Conv1d(hidden_dim, 1, 1)

        self.m_fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.m_fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.m_fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.m_fc_out = nn.Conv1d(hidden_dim, 1, 1)

        self.p_fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.p_fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.p_fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.p_fc_out = nn.Conv1d(hidden_dim, rank, 1)

        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2)

        self.conv_in_bn = nn.BatchNorm2d(16)
        self.conv0_1_bn = nn.BatchNorm2d(32)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv3_1_bn = nn.BatchNorm2d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0])
        for x in range(2):
            for y in [-1, 1]:
                input = [0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
       # x = x.unsqueeze(1)
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=1)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, align_corners = True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, align_corners=True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, align_corners=True)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[2], shape[3]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        mean_func = self.actvn(self.m_fc_0(features))
        mean_func = self.actvn(self.m_fc_1(mean_func))
        mean_func = self.actvn(self.m_fc_2(mean_func))
        mean_func = self.m_fc_out(mean_func)
        #mean_func = mean_func.squeeze(1)

        low_rank_factor_func = self.actvn(self.p_fc_0(features))
        low_rank_factor_func = self.actvn(self.p_fc_1(low_rank_factor_func))
        low_rank_factor_func = self.actvn(self.p_fc_2(low_rank_factor_func))
        low_rank_factor_func = self.p_fc_out(low_rank_factor_func)
        #low_rank_factor_func = low_rank_factor_func.squeeze(1)

        diagonal_func = self.actvn(self.d_fc_0(features))
        diagonal_func = self.actvn(self.d_fc_1(diagonal_func))
        diagonal_func = self.actvn(self.d_fc_2(diagonal_func))
        diagonal_func = self.d_fc_out(diagonal_func)
        #diagonal_func = net.squeeze(1)

        return mean_func, low_rank_factor_func, diagonal_func



class ShapeNet128Vox(nn.Module):
    def __init__(self, rank, hidden_dim=256):
        super(ShapeNet128Vox, self).__init__()
        # accepts 128**3 res input
        self.conv_in = nn.Conv2d(1, 16, 3, padding=1)  # out: 128
        self.conv_0 = nn.Conv2d(16, 32, 3, padding=1)  # out: 64
        self.conv_0_1 = nn.Conv2d(32, 32, 3, padding=1)  # out: 64
        self.conv_1 = nn.Conv2d(32, 64, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv2d(64, 64, 3, padding=1)  # out: 32
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)  # out: 16
        self.conv_2_1 = nn.Conv2d(128, 128, 3, padding=1)  # out: 16
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)  # out: 8
        self.conv_3_1 = nn.Conv2d(128, 128, 3, padding=1)  # out: 8

        feature_size = (1 + 16 + 32 + 64 + 128 + 128) * 5

        self.m_fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.m_fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.m_fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.m_fc_out = nn.Conv1d(hidden_dim, 1, 1)

        self.p_fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.p_fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.p_fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.p_fc_out = nn.Conv1d(hidden_dim, rank, 1)

        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2)

        self.conv_in_bn = nn.BatchNorm2d(16)
        self.conv0_1_bn = nn.BatchNorm2d(32)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv3_1_bn = nn.BatchNorm2d(128)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0])
        for x in range(2):
            for y in [-1, 1]:
                input = [0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
        #x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=1)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p, align_corners = True)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, align_corners = True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, align_corners = True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, align_corners = True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, align_corners = True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, align_corners = True)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[2], shape[3]))  # (B, featues_per_sample, samples_num)
        # features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
        mean_func = self.actvn(self.m_fc_0(features))
        mean_func = self.actvn(self.m_fc_1(mean_func))
        mean_func = self.actvn(self.m_fc_2(mean_func))
        mean_func = self.m_fc_out(mean_func)
        #mean_func = mean_func.squeeze(1)

        low_rank_factor_func = self.actvn(self.p_fc_0(features))
        low_rank_factor_func = self.actvn(self.p_fc_1(low_rank_factor_func))
        low_rank_factor_func = self.actvn(self.p_fc_2(low_rank_factor_func))
        low_rank_factor_func = self.p_fc_out(low_rank_factor_func)
        #low_rank_factor_func = low_rank_factor_func.squeeze(1)
        return mean_func, low_rank_factor_func



