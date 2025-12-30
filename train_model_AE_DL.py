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
from label_proportion import label_proportions
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

# ---------- Class weights (softer, aligned) ----------
num_classes = 13  # including background 0
freq = label_proportions(d_train)  # shape (13,)

assert len(freq) == num_classes, f"Expected {num_classes} frequencies, got {len(freq)}"

epsilon = 1e-6
inv_sqrt_freq = 1.0 / np.sqrt(freq + epsilon)
weights_np = inv_sqrt_freq / inv_sqrt_freq.mean()
weights_np = np.clip(weights_np, 0.5, 5.0)
class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
print("Class weights:", class_weights)

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

# Note: use hidden_dim=256 as defined above
model = PerPixelMLPWithContext(in_channels=64, hidden_dim=256, num_classes=13).to(device)

# ---------- Loss and optimiser ----------
criterion = nn.CrossEntropyLoss(weight=class_weights)  # use weights here

def setup_optimiser(model, learning_rate, weight_decay):
    # Adam usually converges better than high‑lr SGD for this setup [web:97][web:109]
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ---------- Train / val loops ----------
def train_epoch(data_loader, model, optimiser, device):
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
    return model, loss_total, oa_total

def validate_epoch(data_loader, model, device):
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

# ---------- Checkpointing ----------
os.makedirs("models/AE", exist_ok=True)

def load_model(epoch="latest"):
    model = PerPixelMLPWithContext()
    modelStates = glob.glob("models/AE/*.pth")

    # No checkpoints at all
    if not modelStates:
        print("No checkpoints found, returning fresh model.")
        return model.to(device), 0

    # Handle special 'best' checkpoint
    if epoch == "best":
        best_path = "models/AE/best.pth"
        if os.path.exists(best_path):
            stateDict = torch.load(best_path, map_location="cpu")
            model.load_state_dict(stateDict)
            return model.to(device), "best"
        else:
            print("No best.pth found, falling back to latest numeric checkpoint.")
            epoch = "latest"   # fall through to latest logic

    # For 'latest' or a specific numeric epoch, use numeric filenames only
    numeric_states = [
        int(os.path.basename(m).replace(".pth", ""))
        for m in modelStates
        if os.path.basename(m).replace(".pth", "").isdigit()
    ]

    if not numeric_states:
        print("No numeric checkpoints found, returning fresh model.")
        return model.to(device), 0

    if epoch == "latest":
        epoch_to_load = max(numeric_states)
    else:
        epoch_to_load = int(epoch)

    path = f"models/AE/{epoch_to_load}.pth"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} does not exist.")

    stateDict = torch.load(path, map_location="cpu")
    model.load_state_dict(stateDict)
    return model.to(device), epoch_to_load


def save_model(model, epoch, tag=None):
    if tag is None:
        path = f"models/AE/{epoch}.pth"
    else:
        path = f"models/AE/{tag}.pth"
    torch.save(model.state_dict(), open(path, "wb"))

# ---------- Hyperparameters ----------
start_epoch   = 0
learning_rate = 1e-3        # Adam lr
weight_decay  = 1e-4
num_epochs    = 50        # train longer
patience      = 8           # for early stopping


if start_epoch == 0:
    model = PerPixelMLPWithContext(in_channels=64, hidden_dim=256, num_classes=13).to(device)
    epoch = 0
else:
    model, epoch = load_model(epoch=start_epoch)


# Load / init model and optimiser
optim = setup_optimiser(model, learning_rate, weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode="min", factor=0.5, patience=3
)

best_val_oa = -1.0
epochs_no_improve = 0

# ---------- Training loop with best‑val checkpoint ----------
while epoch < num_epochs:
    model, loss_train, oa_train = train_epoch(train_dl, model, optim, device)
    loss_val, oa_val = validate_epoch(val_dl, model, device)
    scheduler.step(loss_val)

    print("[Ep. {}/{}] Loss train: {:.2f}, val: {:.2f}; OA train: {:.2f}, val: {:.2f}".format(
        epoch + 1, num_epochs,
        loss_train, loss_val,
        100 * oa_train, 100 * oa_val
    ))

    # Save last epoch
    save_model(model, epoch + 1)

    # Save best‑validation model
    if oa_val > best_val_oa:
        best_val_oa = oa_val
        epochs_no_improve = 0
        save_model(model, epoch + 1, tag="best")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    epoch += 1

# Later, for predictions/visualization, you can load "best" instead of "latest"
# model, _ = load_model(epoch="best")

# La partie visualisation se fait sur le jupyter notebook "plot.ipynb"
# J'implémente ici une feature qui permet de sauvegarder les images de prédiction dans modeloutputs/AE_prediction/
# Le dossier en question est dans gitignore, il faut donc créer un dossier vide qui s'appelle AE_prediction pour que ça marche

def save_predictions(dataset, model, device="cuda"):
    #we want to save the prediction map of the trained model to plot it later for qualitative assessment
    model.eval()
    model.to(device)
    for idx in tqdm(range(len(dataset)), desc="Saving predictions"):
        img, _ = dataset[idx]
        #get original GT path to recover filename + metadata
        _, gt_path = dataset.samples[idx]
        fname = os.path.basename(gt_path)
        out_path = os.path.join("modeloutputs/AE_prediction", fname)
        #prepare input
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        x = x.to(device)
        #forward pass
        pred = model(x).argmax(1).squeeze().cpu().detach().numpy().astype(np.uint8)
        #read gt metadata
        with rasterio.open(gt_path) as src:
            profile = src.profile
        #update profile for single-band label mask
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw"
        )
        #save prediction
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(pred, 1)


# Saving prediction maps
os.makedirs("modeloutputs/AE_prediction", exist_ok=True)
best_model, _ = load_model(epoch="best")
print("Saving training predictions with best model...")
save_predictions(d_train, best_model)
print("Predictions saved in modeloutputs/AE_prediction")

# Il y a encore une erreur dans la ligne 277         pred = model(x).argmax(1).squeeze().cpu().detach().numpy().astype(np.uint8)
# A fixer



def visualize(dataLoader, epochs, numImages=5):
  models = [load_model(e)[0] for e in epochs]
  numModels = len(models)
  for idx, (data, labels) in enumerate(dataLoader):
    if idx == numImages:
      break

    _, ax = plt.subplots(nrows=1, ncols=numModels+1, figsize = (20, 15))

    # plot ground truth
    ax[0].imshow(labels[0,...].cpu().numpy())
    ax[0].axis('off')
    if idx == 0:
      ax[0].set_title('Ground Truth')

    for mIdx, model in enumerate(models):
        model = model.to(device)
        with torch.no_grad():
            data_vis = data.to(device)
            data_vis = data_vis.permute(0, 3, 1, 2).contiguous()
            pred = model(data_vis)
        # get the label (i.e., the maximum position for each pixel along the class dimension)
        yhat = torch.argmax(pred, dim=1)

        # plot model predictions
        ax[mIdx+1].imshow(yhat[0,...].cpu().numpy())
        ax[mIdx+1].axis('off')
        if idx == 0:
          ax[mIdx+1].set_title(f'Epoch {epochs[mIdx]}')


# visualize predictions for a number of epochs

# load model states at different epochs
epochs = ["best", 1, 5, "latest"]
visualize(val_dl, epochs, numImages=5)


