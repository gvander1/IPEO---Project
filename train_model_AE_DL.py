# Script training a DL model on the training dataset using the GPU
# Mainly inspired by the ex8 Jupyter Notebook of the IPEO course

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from models.lcamazon import LCAmazon
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD
import os
import rasterio

#Checking if GPU is available
if torch.cuda.is_available() == 1:
    print("GPU is available")
else:
    print("GPU to be started !")


d_train = LCAmazon(root="DATA", modality="AE", split="train")
d_val=LCAmazon(root="DATA", modality="AE", split="val")
d_test=LCAmazon(root="DATA", modality="AE", split="test")


#Data Loader
train_dl = DataLoader(d_train, batch_size = 16, shuffle=True, num_workers=1)
val_dl = DataLoader(d_val, batch_size=16, num_workers=1)
#Sematic segmentation with deepl learning - model taken from ex 9
"""
We need pixel wise classification so semantic segmentation
Starting with small fully connected convolutional network that does per pixel classification with local context on feature map
"""

# @Gaétane je mets ici une fonction qui normalise les images sur chaque channel (hyper important avant de le passer au modèle)
# Normalization
read=np.load("mean_std/AE.npy")
mean, std = read[0], read[1]
mean=torch.tensor(mean)
std=torch.tensor(std)
normalize = T.Normalize(mean, std)


#MODEL 
class PerPixelMLPWithContext(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=128, num_classes=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),  # 3x3 filter size, keeps 47x47 size images - we want to keep the image size as we want one class per pixels
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1), #reimbed features into richer feature space ()
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, 64, 47, 47)
        return self.net(x)  # (B, num_classes, 47, 47)
    

train_dl =DataLoader(d_train, batch_size=2,shuffle=True, num_workers=1)
val_dl =DataLoader(d_val, batch_size=2,shuffle=True, num_workers=1)

device = torch.device("cuda")
model = PerPixelMLPWithContext(in_channels=64, hidden_dim=128, num_classes=13).to(device)

# # One batch from your train_loader
# data, target = next(iter(dataload_train_test))   # current: (B, H, W, C) from your dataset
# # If your LCAmazon __getitem__ returns (H, W, C), convert to (B, C, H, W):
# data = data.permute(0, 3, 1, 2).contiguous()
# data = data.to(device)
# model.to(device)

# with torch.no_grad():
#     logits = model(data)  # now shape (B, num_classes, H, W)


# print("Input shape: ", data.shape)
# print("Logits shape:", logits.shape)

# # Check shape consistency
# B, C, H, W = data.shape
# assert logits.shape == (B, 13, H, W), f"Got {logits.shape}, expected {(B, 13, H, W)}"

# # Turn logits into per‑pixel classes
# pred = logits.argmax(dim=1)              # (B, H, W)
# # map back to original IDs
# pred_np = pred.cpu().numpy()
# pred_ids = np.zeros_like(pred_np)
# for idx, orig_id in IDX_TO_ID.items():
#     pred_ids[pred_np == idx] = orig_id
# print("Pred label map shape:", pred.shape)

# # Optionally check that labels are in range [0, num_classes-1]
# print("Pred min/max:", pred.min().item(), pred.max().item())

#MODEL TRAINING (taken from exercises)
criterion = nn.CrossEntropyLoss()
def setup_optimiser(model, learning_rate, weight_decay):
  return SGD(
    model.parameters(),
    learning_rate,
    weight_decay
  )


def train_epoch(data_loader, model, optimiser, device):
  model.train()
  model.to(device)

  # stats
  loss_total = 0.0
  oa_total = 0.0

  # iterate over dataset
  for idx, (data, target) in enumerate(tqdm(data_loader)):

    data, target = data.to(device), target.to(device)
    data = data.permute(0, 3, 1, 2)
    data = normalize(data)

    # reset gradients
    optimiser.zero_grad()

    # forward pass
    target = target.long()
    pred = model(data)

    # loss
    loss = criterion(pred, target)      # CrossEntropyLoss

    # backward pass
    loss.backward()

    # parameter update
    optimiser.step()

    # stats update
    loss_total += loss.item()
    oa_total += torch.mean((pred.argmax(1) == target).float()).item()


  # normalise stats
  loss_total /= len(data_loader)
  oa_total /= len(data_loader)

  return model, loss_total, oa_total

#validation function

def validate_epoch(data_loader, model, device):       # note: no optimiser needed

  # set model to evaluation mode
  model.eval()
  model.to(device)

  # stats
  loss_total = 0.0
  oa_total = 0.0

  # iterate over dataset
  for idx, (data, target) in enumerate(tqdm(data_loader)):
    with torch.no_grad():

      #TODO: likewise, implement the validation routine. This is very similar, but not identical, to the training steps.

        data, target = data.to(device), target.to(device)
        data = data.permute(0, 3, 1, 2).contiguous()
        pred = model(data)

      # forward pass
        pred = model(data)
        target = target.long()
        loss = criterion(pred, target)      # CrossEntropyLoss

      # stats update
        loss_total += loss.item()
        oa_total += torch.mean((pred.argmax(1) == target).float()).item()


  # normalise stats
  loss_total /= len(data_loader)
  oa_total /= len(data_loader)

  return loss_total, oa_total

import glob

os.makedirs('models/AE', exist_ok=True)

def load_model(epoch='latest'):
  model = PerPixelMLPWithContext()
  modelStates = glob.glob('models/AE/*.pth')
  if len(modelStates) and (epoch == 'latest' or epoch > 0):
    modelStates = [int(m.replace('models/AE/','').replace('.pth', '')) for m in modelStates]
    if epoch == 'latest':
      epoch = max(modelStates)
    stateDict = torch.load(open(f'models/AE/{epoch}.pth', 'rb'), map_location='cpu')
    model.load_state_dict(stateDict)
  else:
    # fresh model
    epoch = 0
  return model, epoch


def save_model(model, epoch):
  torch.save(model.state_dict(), open(f'models/AE/{epoch}.pth', 'wb'))

# define hyperparameters
start_epoch = 0        # set to 0 to start from scratch again or to 'latest' to continue training from saved checkpoint
batch_size = 2
learning_rate = 0.1
weight_decay = 0.001
num_epochs = 10


# load model
model, epoch = load_model(epoch=start_epoch)
optim = setup_optimiser(model, learning_rate, weight_decay)

# do epochs
while epoch < num_epochs:

  # training
  model, loss_train, oa_train = train_epoch(train_dl, model, optim, device)

  # validation
  loss_val, oa_val = validate_epoch(val_dl, model, device)

  # print stats
  print('[Ep. {}/{}] Loss train: {:.2f}, val: {:.2f}; OA train: {:.2f}, val: {:.2f}'.format(
      epoch+1, num_epochs,
      loss_train, loss_val,
      100*oa_train, 100*oa_val
  ))

  # save model
  epoch += 1
  save_model(model, epoch)












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
print("Saving training predictions...")
save_predictions(d_train, model)
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
epochs = [0, 1, 5, 'latest']                                        

visualize(val_dl, epochs, numImages=5)
