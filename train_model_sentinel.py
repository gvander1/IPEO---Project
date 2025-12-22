# Script training a DL model on the training dataset using the GPU
# Mainly inspired by the ex8 Jupyter Notebook of the IPEO course

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from models.lcamazon import LCAmazon
from torch.utils.data import DataLoader

#Checking if GPU is available
if torch.cuda.is_available() == 1:
    print("GPU is available")
else:
    print("GPU to be started !")

train_dataset = LCAmazon(root="DATA", modality="s2", split="train")

'''
#sanity check
img, label = train_dataset[0]
print(np.shape(img))
'''

#Data Loader
train_dl = DataLoader(train_dataset, batch_size = 16, shuffle=True, num_workers=2)

#for image, label in train_dl:
#    print(image.shape, label.shape)
#    image has shape [16, 47, 47, 12] label has shape [16, 47, 47]

#Loading of Computer Vision Models --> using torchvision, choose a resnet18 model
from torchvision.models import resnet18
model = resnet18(num_classes=38)  #we have 38 classes
#print(model) #just to see how it is structured

# Model Training
# Loss function
# taking cross-entropy as criterion --> can be edited later

from torch.nn import CrossEntropyLoss
criterion = CrossEntropyLoss()

'''
Checking if the loss function works (it works)

#create 21 dummy predictions with 21 classes and random logits
num_classes = 21
batch_size = 16

# dummy predictions
y_logit = torch.rand(batch_size, num_classes)
# dummy gt
y_true = torch.randint(num_classes, (batch_size,))

loss = criterion(y_logit, y_true)
print(loss)
'''

# Optimizer (use Stochastic Gradient Descent SGD)
from torch.optim import SGD

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Define single training step
weight_value_before = float(model.fc.weight[0,0])
print(weight_value_before)
idx, batch = next(enumerate(train_dl))
model.train()
optimizer.zero_grad()
x, y = batch
device = "cuda" #running on GPU
model = model.to(device)
x = x.to(device)
y = y.to(device)
#forward pass (i.e prediction)
y_logit = model(x)
loss = criterion(y_logit, x)
#backprop
loss.backward()
#update model parameters
optimizer.step()