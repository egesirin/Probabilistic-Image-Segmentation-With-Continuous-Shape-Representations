import torch
from mc_samples import get_coordinates
from if_net import ShapeNet128Vox
from training2 import train_low_rank
from lidc_data import LIDC_IDRI
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_folder = '/scratch/visual/esirin/data/'
batch_size = 12
dataset = LIDC_IDRI(dataset_location=data_folder)
train_size = int(0.6 * len(dataset))
train_size = train_size - (train_size % batch_size) #9048
validation_size = int(0.2 * len(dataset))
validation_size = validation_size - (validation_size % batch_size) #3012
test_size = len(dataset) - train_size - validation_size #3036
training_data, validation_data, test_data = torch.utils.data.random_split(dataset, [train_size, validation_size,
                                                                                    test_size])
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

epoch = 663
lr = 1e-3
mc_sample = 20
# input = torch.rand(batch, 128, 128)
coordinates = get_coordinates([128, 128], batch_size, 1, 1).to(device=DEVICE)
models = ShapeNet128Vox(rank=10).to(device=DEVICE)
criterion = nn.BCEWithLogitsLoss(reduction="sum")
optimizers = torch.optim.Adam(models.parameters(), lr=lr)
writer = SummaryWriter('runs/lcdi/ifnet2')


def main():
    '''
    print(len(dataset))
    print(train_size)
    print(train_size % batch_size)
    print(validation_size)
    print(validation_size % batch_size)
    #torch.autograd.set_detect_anomaly(True)
    '''
    for t in range(epoch):
        train_low_rank(t, models, criterion, optimizers, mc_sample, coordinates, train_dataloader, validation_dataloader,
                       batch_size, writer=writer)

    writer.close()


if __name__ == '__main__':
    main()