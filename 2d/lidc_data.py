import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                print(file_path)
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                print(input_size)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)

        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        #Randomly select one of the four labels for this image

        label0 = self.labels[index][0].astype(float)
        label1 = self.labels[index][1].astype(float)
        label2 = self.labels[index][2].astype(float)
        label3 = self.labels[index][3].astype(float)
        if self.transform is not None:
            image = self.transform(image)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label0 = torch.from_numpy(label0)
        label1 = torch.from_numpy(label1)
        label2 = torch.from_numpy(label2)
        label3 = torch.from_numpy(label3)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        '''
        label0 = label0.type(torch.FloatTensor)
        label1 = label1.type(torch.FloatTensor)
        label2 = label2.type(torch.FloatTensor)
        label3 = label3.type(torch.FloatTensor)
        '''
        label0 = label0.type(torch.float32)
        label1 = label1.type(torch.float32)
        label2 = label2.type(torch.float32)
        label3 = label3.type(torch.float32)
        labels = [label0, label1, label2, label3]


        return image, labels, series_uid
        #return image, labels

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)



