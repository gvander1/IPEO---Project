import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from models.lcamazon import LCAmazon

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
d_train = LCAmazon(root="DATA", modality="AE", split="train", aug_geometric=True)
d_val   = LCAmazon(root="DATA", modality="AE", split="val",   aug_geometric=False)
d_test  = LCAmazon(root="DATA", modality="AE", split="test",  aug_geometric=False)

# DataLoaders
batch_size = 16
train_dl = DataLoader(d_train, batch_size=batch_size, shuffle=True,  num_workers=1)
val_dl   = DataLoader(d_val,   batch_size=batch_size, shuffle=False, num_workers=1)


# ---------- Normalization ----------
read = np.load("mean_std/AE.npy")
mean, std = read[0], read[1]
mean = torch.tensor(mean)
std  = torch.tensor(std)
normalize = T.Normalize(mean, std)

# ---------- Model ----------
#We need pixel wise classification so semantic segmentation
#Starting with small fully connected convolutional network that does per pixel classification with local context on feature map
class PerPixelMLPWithContext(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=256, num_classes=13, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),

            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),

            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),

            nn.Conv2d(hidden_dim, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)

model = PerPixelMLPWithContext(in_channels=64, hidden_dim=256, num_classes=13).to(device)

# Loss function
# taking weighted cross-entropy with w(class)=1/sqrt(freq(class)) as criterion
from torch.nn import CrossEntropyLoss
class_frequency = np.load("class_frequency.npy")
weights = torch.tensor(1/np.sqrt(class_frequency), dtype=torch.float32).to("cuda")  #weights are the inverse of the square root of the frequency of class in the dataset
criterion = CrossEntropyLoss(weight=weights)

def setup_optimiser(model, learning_rate, weight_decay):
    # Adam usually converges better than high‑lr SGD for this setup [web:97][web:109]
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ---------- Train / val loops ----------
def train_epoch(data_loader, model, optimiser, device="cuda"):
    model.train()
    loss_total = 0.0
    oa_total = 0.0

    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        data = data.permute(0, 3, 1, 2)
        data = normalize(data)

        optimiser.zero_grad()
        target = target.long()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimiser.step()

        loss_total += loss.item()
        oa_total += torch.mean((pred.argmax(1) == target).float()).item()

    loss_total /= len(data_loader)
    oa_total /= len(data_loader)
    return loss_total, oa_total

def validate_epoch(data_loader, model, device="cuda"):
    model.eval()
    loss_total = 0.0
    oa_total = 0.0

    for data, target in tqdm(data_loader):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            data = data.permute(0, 3, 1, 2).contiguous()
            data = normalize(data)

            target = target.long()
            pred = model(data)
            loss = criterion(pred, target)

            loss_total += loss.item()
            oa_total += torch.mean((pred.argmax(1) == target).float()).item()

    loss_total /= len(data_loader)
    oa_total /= len(data_loader)
    return loss_total, oa_total


def save_model(model, epoch, fold):
    os.makedirs('models/AE_final', exist_ok=True)
    torch.save(model.state_dict(), open(f'models/AE_final/fold{fold+1}_epoch{epoch+1}.pth', 'wb'))


# ---------- Hyperparameters ----------
learning_rate = 1e-4        # Adam lr
weight_decay  = 1e-4
num_epochs    = 20        # train longer


# Load / init model and optimiser
optimizer = setup_optimiser(model, learning_rate, weight_decay)

# Model Training
num_folds = 5
num_epochs = 20 #computation time about 43 seconds/epoch
stats = []
for i in range(num_folds):
    stats.append([])
print("-------------- Training the model and validate it at the same time -------------------")
for fold in range(num_folds):
    for epoch in range(num_epochs):
        train_set = LCAmazon(root="DATA", split="train", modality="AE", aug_geometric=True, num_folds=num_folds, fold_index=fold)
        val_set = LCAmazon(root="DATA", split="val", modality="AE",num_folds=num_folds, fold_index=fold)
        # build dataloaders
        train_dl = DataLoader(train_set, batch_size=16, shuffle=True)
        val_dl = DataLoader(val_set, batch_size=16)
        trainloss, trainaccuracy = train_epoch(train_dl, model, optimizer)
        valloss, valaccuracy = validate_epoch(val_dl, model)
        print(f"fold {fold+1}; epoch {epoch+1}; trainloss {trainloss:.2f}, train accuracy {trainaccuracy*100:.2f}%")
        print(f"fold {fold+1}; epoch {epoch+1}; valloss {valloss:.2f}, val accuracy {valaccuracy*100:.2f}%")
        stats[fold].append({
            "trainloss":float(trainloss),
            "trainaccuracy":float(trainaccuracy), 
            "valloss":float(valloss),
            "valaccuracy":float(valaccuracy),
            "epoch":epoch
        })

print(f"-------------- Training done ; number of epochs = {num_epochs} -------------------")

#saving model
save_model(model, epoch, fold)

# Plotting accuracy as a function of epoch
for fold in range(num_folds):
    trainlosses = np.stack([stat["trainloss"] for stat in stats[fold]])
    trainaccuracy = np.stack([stat["trainaccuracy"] for stat in stats[fold]])
    vallosses = np.stack([stat["valloss"] for stat in stats[fold]])
    valaccuracy = np.stack([stat["valaccuracy"] for stat in stats[fold]])
    epoch = np.stack([stat["epoch"] for stat in stats[fold]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Accuracy plot
    ax1.plot(epoch, trainaccuracy, label="Train Accuracy", marker='o')
    ax1.plot(epoch, valaccuracy, label="Validation Accuracy", marker='x')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(f"Accuracy of fold {fold+1}")
    ax1.set_title("Accuracy")
    ax1.grid(True)
    ax1.legend()
    # Loss plot
    ax2.plot(epoch, trainlosses, label="Train Loss", marker='o', linestyle='--', color='red')
    ax2.plot(epoch, vallosses, label="Val Loss", marker='x', linestyle='--', color='orange')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Loss of fold {fold+1}")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    os.makedirs('modeloutputs/AE_accuracy_final', exist_ok=True)
    plt.savefig(f"modeloutputs/AE_accuracy_final/s2_train_val_accuracy_fold{fold+1}.png")
    print("Accuracy figure saved in modeloutputs/AE_accuracy_final/")
