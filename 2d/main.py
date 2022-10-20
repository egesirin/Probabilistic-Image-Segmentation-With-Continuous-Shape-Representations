import torch
from mc_samples import get_coordinates
from model import ShapeNet128Vox_Low_Rank
from training import train, validation
from lidc_data import LIDC_IDRI
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def main():
    data_folder = '/scratch/visual/esirin/data/'
    batch_size = 12
    #batch_size = 2
    dataset = LIDC_IDRI(dataset_location=data_folder)
    #train_size = 2
    train_size = int(0.6 * len(dataset))  # 9057
    validation_size = int(0.2 * len(dataset))  # 3019
    test_size = len(dataset) - train_size - validation_size  # 3020
    training_data, validation_data, test_data = torch.utils.data.random_split(dataset, [train_size, validation_size,
                                                                                        test_size])
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    epoch = 663
    #epoch = 10
    lr = 1e-3
    number_of_mc_sample = 20
    number_of_val_sample = 4
    pixel = 128
    image_shape = torch.tensor([pixel, pixel], dtype=torch.float32)
    coordinates = get_coordinates(image_shape, batch_size, 1).cuda()
    #coordinates = get_coordinates([128, 128], batch_size, 1, 1).to(device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    iterations_per_epoch_training = train_size // batch_size
    iterations_per_epoch_validation = validation_size // batch_size
    print(len(validation_data))
    print(iterations_per_epoch_validation)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", required=True, type=str, help="model type: m = only mean,"
                                                                  " m_p = mean, covariance = PP^T,"
                                                                  "m_p_d = mean, covariance = PP^T +D")
    model_type = ap.parse_args().model_type
    writer = SummaryWriter('runs/lcdi/model/' + model_type)
    models = ShapeNet128Vox_Low_Rank(rank=10, type=model_type).to(device=DEVICE)
    optimizers = torch.optim.Adam(models.parameters(), lr=lr)

    for t in range(epoch):
        train(t, iterations_per_epoch_training, models, model_type, criterion, optimizers, number_of_mc_sample,
              coordinates, train_dataloader, batch_size, DEVICE, writer)
        validation(t, iterations_per_epoch_validation, models, model_type, criterion, number_of_mc_sample,
                   number_of_val_sample, coordinates, validation_dataloader, batch_size, DEVICE, writer)

    writer.close()


if __name__ == '__main__':
    main()
