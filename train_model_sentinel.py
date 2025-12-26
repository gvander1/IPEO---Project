# Script training a DL model on the training dataset using the GPU
# Mainly inspired by the ex8 Jupyter Notebook of the IPEO course

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
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

'''
#sanity check
img, label = train_dataset[0]
print(np.shape(img))
'''

#Data Loader
train_dl = DataLoader(train_dataset, batch_size = 16, num_workers=1)
val_dl = DataLoader(val_dataset, batch_size=16, num_workers=1)

#for image, label in train_dl:
#    print(image.shape, label.shape)
#    image has shape [16, 47, 47, 12] label has shape [16, 47, 47]

# Normalization
mean=torch.tensor([1169.2191,  917.3188,  847.3458,  739.8301, 1033.2438, 1957.2441, 2428.3120, 2375.6396, 2794.8320,  518.3082, 2045.9958, 1027.1027])
std=torch.tensor([111.8526, 158.9845, 213.6219, 437.9030, 402.8841, 362.7410, 466.6497, 476.7180, 536.5454, 116.5597, 962.3427, 699.4785])
normalize = T.Normalize(mean, std)
std_inv = 1 / (std + 1e-7)
unnormalize = T.Normalize(-mean * std_inv, std_inv)

#Loading of Computer Vision Models --> using torchvision, choose a fcn resnet50
from torchvision.models.segmentation import fcn_resnet50
model = fcn_resnet50(progress=True, num_classes=34)  #we have 13 classes
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
'''
model.classifier = torch.nn.Sequential(
    torch.nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
    torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    torch.nn.ReLU(inplace=False),
    torch.nn.Dropout(p=0.5, inplace=True),
    torch.nn.Conv2d(512, 38, kernel_size=(1, 1), stride=(1, 1))
)
'''
#print(model) #see if it worked


# need to map labels to indices so they don't overshoot num_classes
ORIG_IDS = [3, 4, 6, 9, 11, 12, 15, 18, 24, 25, 30, 33]
ID_TO_IDX = {orig_id: i for i, orig_id in enumerate(ORIG_IDS)}
IDX_TO_ID = {i: orig_id for i, orig_id in enumerate(ORIG_IDS)}
def remap_labels_to_indices(target):
    # target: (B, H, W) with original IDs
    target_np = target.cpu().numpy()
    out = np.full_like(target_np, fill_value=-1)
    for orig_id, idx in ID_TO_IDX.items():
        out[target_np == orig_id] = idx
    return torch.from_numpy(out).to(target.device)


# Model Training
# Loss function
# taking cross-entropy as criterion --> can be edited later

from torch.nn import CrossEntropyLoss
criterion = CrossEntropyLoss()

# Optimizer (use Stochastic Gradient Descent SGD)
from torch.optim import SGD

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def training_step(batch, model, optimizer, device="cuda"):
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    x,y = batch
    x = x.permute(0, 3, 1, 2) #Putting tensor in right order (Batch, Channel, Height, Width)
    x = normalize(x) #Apply normalization
    model = model.to(device)
    x = x.to(device)
    y = y.to(device).long() #the crossentropy loss expects a type long
    #forward pass (i.e prediction)
#    y = remap_labels_to_indices(y)
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
    x = normalize(x) #Apply normalization
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
num_epochs = 2 #computation time about 30 seconds/epoch
stats = [] #after 30 epochs we reach 85 % accuracy with current hyperparameters, staying at 62% val accuracy (=overfitting !)
print("-------------- Training the model and validate it at the same time -------------------")
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

print(f"-------------- Training done ; number of epochs = {num_epochs} -------------------")
# Plotting train accuracy as a function of epoch
trainlosses = np.stack([stat["trainloss"] for stat in stats])
trainaccuracy = np.stack([stat["trainaccuracy"] for stat in stats])
vallosses = np.stack([stat["valloss"] for stat in stats])
valaccuracy = np.stack([stat["valaccuracy"] for stat in stats])
epoch = np.stack([stat["epoch"] for stat in stats])

fig, ax1 = plt.subplots()
ax1.plot(epoch, trainaccuracy, label="Train Accuracy", marker='o')
ax1.plot(epoch, valaccuracy, label="Validation Accuracy", marker='x')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(epoch, trainlosses, label="Train Loss", marker='o', linestyle='--', color='red')
ax2.plot(epoch, vallosses, label="Val Loss", marker='x', linestyle='--', color='orange')
ax2.set_ylabel("Loss")
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
plt.title("Training and Validation Accuracy & Loss")
plt.savefig("DATA/results/s2_train_val_accuracy.png")
print("Accuracy figure saved in DATA/results/")

## IDEAS to improve model accuracy
# Try Adam or AdamW optimizer
# Regularize through Dropouts or Data augmentation (add random rotations) using torchvision.transforms
# Add a pre-processing conv layer to reduce from 12 to 3 channels
# Use Weighted Cross-Entropy to deal with class imbalance
#       weights = torch.tensor([1.0, 2.0, 0.5, ...]).to(device)  # adjust per class
#       criterion = CrossEntropyLoss(weight=weights)