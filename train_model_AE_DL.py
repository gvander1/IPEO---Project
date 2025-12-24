# Script training a DL model on the training dataset using the GPU
# Mainly inspired by the ex8 Jupyter Notebook of the IPEO course

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.lcamazon import LCAmazon
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Checking if GPU is available
if torch.cuda.is_available() == 1:
    print("GPU is available")
else:
    print("GPU to be started !")

train_dataset = LCAmazon(root="DATA", modality="AE", split="train")
val_dataset = LCAmazon(root="DATA", modality="AE", split="val")

'''
#sanity check
img, label = train_dataset[0]
print(np.shape(img))
'''

#Data Loader
train_dl = DataLoader(train_dataset, batch_size = 16, shuffle=True, num_workers=1)
val_dl = DataLoader(val_dataset, batch_size=16, num_workers=1)
#Sematic segmentation with deepl learning - model taken from ex 9
"""
We need pixel wise classification so semantic segmentation
Hypercolumn architecture : 
Performs down sampling via convolutions, pooling, etc... it keeps every intermediate output, upsamples (interpolates) them to the original image's size, 
stacks them together to a large tensor (a hypercolumn) and uses this to perform pixel-wise classification:
"""
import torch.nn as nn


class Hypercolumn(nn.Module):

    def __init__(self):
        super(Hypercolumn, self).__init__()

        #TODO: define your architecture and forward pass here
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=4),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=4),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(484, 256, kernel_size=1, stride=1),           # 485 = 256 + 128 + 64 + 32 + 4 (input bands)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 6, kernel_size=1, stride=1)
        )
    

    def forward(self, x):
        #TODO
        upsample = nn.Upsample(size=(x.size(2), x.size(3)))
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        hypercol = torch.cat(
            (x, upsample(x1), upsample(x2), upsample(x3), upsample(x4)),
            dim=1
        )
        return self.final(hypercol)