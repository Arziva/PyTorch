#1.working with data


#importing essentials

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#downloading test data form open datasets
training_data = datasets.FashionMNIST(
    root = "data",
    train = "True",
    download = "True",
    transform = "ToTensor()"
)

#downloading test data from open datasets
test_data = datasets.FashionMNIST(
    root = "data",
    train = "True",
    download = "True",
    transform = "ToTensor()"
)


#datasets into DataLoader
batchsize = 64

train_dataloader = DataLoader(training_data, batch_size=batchsize) 
test_dataloader = DataLoader(test_data, batch_size=batchsize)

for X, y in train_dataloader:
    print(f"The size of training data{X.shape}")
    print(f"The size of testing data{y.dtype}")