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
from tqdm.auto import tqdm
from label_proportion import label_proportions

#Checking if GPU is available
if torch.cuda.is_available() == 1:
    print("GPU is available")
else:
    print("GPU to be started !")

train_dataset = LCAmazon(root="DATA", modality="s2", split="train")
val_dataset = LCAmazon(root="DATA", modality="s2", split="val")

print(label_proportions(train_dataset))

'''
#sanity check
img, label = train_dataset[0]
print(np.shape(img))
'''

#Data Loader
train_dl = DataLoader(train_dataset, batch_size = 16, shuffle=True, num_workers=1)
val_dl = DataLoader(val_dataset, batch_size=16, num_workers=1)

#for image, label in train_dl:
#    print(image.shape, label.shape)
#    image has shape [16, 47, 47, 12] label has shape [16, 47, 47]

#Loading of Computer Vision Models --> using torchvision, choose a fcn resnet50
from torchvision.models.segmentation import fcn_resnet50
model = fcn_resnet50(progress=True, num_classes=38)  #we have 38 classes
#print(model) #just to see how it is structured

# The resnet18 model is made for 3 bands --> need to change the first convolution layer to fix this
model.backbone.conv1 = torch.nn.Conv2d(
    in_channels=12,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

#print(model) #see if it worked

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
# Before training step
#weight_before = model.classifier[4].weight[0, 0, 0, 0].item()
#print("Weight before:", weight_before)
idx, batch = next(enumerate(train_dl))
model.train()
optimizer.zero_grad()
x, y = batch
x = x.permute(0, 3, 1, 2) #Putting tensor in right order (Batch, Channel, Height, Width)
device = "cuda" #running on GPU
model = model.to(device)
x = x.to(device)
y = y.to(device).long() #the crossentropy loss expects a type long
#forward pass (i.e prediction)
outputs = model(x)
y_logit = outputs["out"]
loss = criterion(y_logit, y)
#backprop
loss.backward()
#update model parameters
optimizer.step()

#weight_after = model.classifier[4].weight[0, 0, 0, 0].item()
#print("Weight after:", weight_after)

#Ok it seems to work correctly, let's now package the training step into a function
def training_step(batch, model, optimizer, device="cuda"):
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    x,y = batch
    x = x.permute(0, 3, 1, 2) #Putting tensor in right order (Batch, Channel, Height, Width)
    x = x.to(device)
    y = y.to(device).long() #the crossentropy loss expects a type long
    #forward pass (i.e prediction)
    outputs = model(x)
    y_hat = outputs["out"]
    loss = criterion(y_hat, y)
    loss.backward() # backprop
    optimizer.step() # update model param
    # lets also calculate accuracy for fun
    # FYI
    # .cpu() moves the data back to cpu (if on GPU)
    # .detach() removes gradients (we dont need them for accuracy)
    # .numpy() converts the tensor to numpy for better handling later
    predictions = y_hat.argmax(1).cpu().detach().numpy()
    ground_truth = y.cpu().detach().numpy()

    # accuracy is the mean of correct (1) and incorrect (0) classifications
    accuracy = (predictions == ground_truth).mean()

    return loss, accuracy

@torch.no_grad() #we wkip the calculation of the gradient graph here to save time
def prediction_step(batch, model, device="cuda"):
    model.eval()
    x, y = batch
    x = x.permute(0, 3, 1, 2) #Putting tensor in right order (Batch, Channel, Height, Width)
    x = x.to(device)
    y = y.to(device).long() #the crossentropy loss expects a type long
    #forward pass (i.e prediction)
    outputs = model(x)
    y_hat = outputs["out"]
    loss = criterion(y_hat, y)
    predictions = y_hat.argmax(1).cpu().detach().numpy()
    ground_truth = y.cpu().detach().numpy()
    # accuracy is the mean of correct (1) and incorrect (0) classifications
    accuracy = (predictions == ground_truth).mean()
    return loss, accuracy


def train_epoch(train_dl, val_dl, model, optimizer):
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    for batch in tqdm(train_dl):
        loss, accuracy = training_step(batch, model, optimizer)
        train_losses.append(loss.cpu().detach().numpy())
        train_accuracies.append(accuracy)
    for batch in tqdm(val_dl):
        loss, accuracy = prediction_step(batch, model)
        val_losses.append(loss.cpu().detach().numpy())
        val_accuracies.append(accuracy)
    val_losses, train_losses, val_accuracies, train_accuracies = np.stack(val_losses).mean(), np.stack(train_losses).mean(), np.stack(val_accuracies).mean(), np.stack(train_accuracies).mean()
    return val_losses, train_losses, val_accuracies, train_accuracies

# Model Training
num_epochs = 3 #computing 30 epochs takes a while (15 minutes currently), use 3 for debugging
stats = [] #after 30 epochs we reach 85 % accuracy with current hyperparameters, staying at 62% val accuracy (=overfitting !)
for epoch in range(num_epochs):
    valloss, trainloss, valaccuracy, trainaccuracy = train_epoch(train_dl, val_dl, model, optimizer)
    print(f"epoch {epoch}; trainloss {trainloss:.2f}, train accuracy {trainaccuracy*100:.2f}%")
    print(f"epoch {epoch}; valloss {valloss:.2f}, val accuracy {valaccuracy*100:.2f}%")
    stats.append({
        "trainloss":float(trainloss),
        "trainaccuracy":float(trainaccuracy),
        "valloss":float(valloss),
        "valaccuracy":float(valaccuracy),
        "epoch":epoch
    })

# Plotting train accuracy as a function of epoch
trainlosses = np.stack([stat["trainloss"] for stat in stats])
trainaccuracy = np.stack([stat["trainaccuracy"] for stat in stats])
vallosses = np.stack([stat["valloss"] for stat in stats])
valaccuracy = np.stack([stat["valaccuracy"] for stat in stats])
epoch = np.stack([stat["epoch"] for stat in stats])

fig, ax = plt.subplots()
ax.plot(epoch, trainaccuracy, label="Train Accuracy", marker='o')
ax.plot(epoch, valaccuracy, label="Validation Accuracy", marker='x')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Train vs Validation Accuracy")
ax.legend()
ax.grid(True)
plt.savefig("DATA/results/s2_train_val_accuracy.png")

## IDEAS to improve model accuracy
# Try Adam or AdamW optimizer
# Regularize through Dropouts or Data augmentation (add random rotations) using torchvision.transforms
# Add a pre-processing conv layer to reduce from 12 to 3 channels
# Use Weighted Cross-Entropy to deal with class imbalance
#       weights = torch.tensor([1.0, 2.0, 0.5, ...]).to(device)  # adjust per class
#       criterion = CrossEntropyLoss(weight=weights)