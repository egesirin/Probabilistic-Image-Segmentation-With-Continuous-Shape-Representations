import torch
import torch.nn as nn
import torch.nn.functional as F
from distribution_components import DistributionComponents

class ShapeNet128Vox_Low_Rank(nn.Module):
    def __init__(self, rank, type, hidden_dim=256):
        super(ShapeNet128Vox_Low_Rank, self).__init__()
        # accepts 128**3 res input
        self.functions = nn.ModuleList()
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
        if type == "m_p_d":
            self.functions.append(DistributionComponents(feature_size, 1, hidden_dim))
            self.functions.append(DistributionComponents(feature_size, rank, hidden_dim))
            self.functions.append(DistributionComponents(feature_size, 1, hidden_dim))
        elif type == "m_p":
            self.functions.append(DistributionComponents(feature_size, 1, hidden_dim))
            self.functions.append(DistributionComponents(feature_size, rank, hidden_dim))
        elif type == "m":
            self.functions.append(DistributionComponents(feature_size, 1, hidden_dim))
        else:
            raise NotImplementedError

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
        functions_list = []
        #x = x.unsqueeze(1)
        p_features = p.transpose(1, -1)
        #print(x.shape)

        p = p.unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=1)  # (B,1,7,num_samples,3)
        #print(p.shape)
        feature_0 = F.grid_sample(x, p, align_corners=False)  # out : (B,C (of x), 1,1,sample_num)
        #print(feature_0.shape)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, align_corners=False)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, align_corners=False)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, align_corners=False)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, align_corners=False)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, align_corners=False)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[2], shape[3]))  # (B, featues_per_sample, samples_num)
        for i in range(len(self.functions)):
            functions_list.append(self.functions[i](features))

        return functions_list



