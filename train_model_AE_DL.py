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


#first create a mask to keep only the indices of images that have labels in classes that are represented in training, validation and test
d_train = LCAmazon(root="DATA", modality="AE", split="train")
d_val=LCAmazon(root="DATA", modality="AE", split="val")
d_test=LCAmazon(root="DATA", modality="AE", split="test")

'''
UNCOMMENT UNIQUELABEL DEFINITION TO CREATE FILTERING MASK
'''

# # create df with class value, nb of pixels and relative proportions
# def uniquelabels(dataset):
#     """
#     dataset: e.g. LCAmazon(..., split='train'), where dataset[i] -> (img, lbl)
#              and lbl has shape (H, W) with integer class ids.
#     """
#     all_labels = []

#     for i in tqdm(range(len(dataset))):
#         _, lbl = dataset[i]
#         all_labels.append(lbl.ravel())

#     labels_flat = np.concatenate(all_labels, axis=0)  # shape: (N_pixels,)

#     # Just return the unique class IDs
#     classes = np.unique(labels_flat)  # 1D array of sorted unique labels.[web:19]

#     return classes


# unique_label_v= uniquelabels(d_val)
# print(unique_label_v)
# unique_label_test= uniquelabels(d_test)
# print(unique_label_test)

# unique_label_v_set    = set(unique_label_v)
# unique_label_test_set = set(unique_label_test)

# commonlabels = unique_label_v_set & unique_label_test_set

# print(commonlabels)



# def keep_image_no_0_29(label_map):
#     """Return True if the image has NO pixels with label 0 or 29."""
#     labels = np.unique(label_map)                      # all distinct labels in this image [web:21]
#     banned = np.array([0, 29], dtype=labels.dtype)
#     # True if none of the labels are in {0, 29}
#     return ~np.isin(labels, banned).any()

# def build_mask_exclude_0_29(dataset):
#     """
#     dataset: e.g. LCAmazon(...), where dataset[i] -> (img, lbl)
#     returns: boolean mask of length len(dataset)
#     """
#     mask = []
#     for i in tqdm(range(len(dataset))):
#         _, lbl = dataset[i]
#         mask.append(keep_image_no_0_29(lbl))
#     return np.array(mask, dtype=bool)

# train_mask = build_mask_exclude_0_29(d_train)
# val_mask   = build_mask_exclude_0_29(d_val)

train_mask = np.load("train_mask.npy")
val_mask   = np.load("val_mask.npy")

print("train kept:", np.count_nonzero(train_mask))
print("val kept:",   np.count_nonzero(val_mask))

# boolean mask -> integer indices
train_indices = np.nonzero(train_mask)[0]
val_indices   = np.nonzero(val_mask)[0]

# train_mask and val_mask are boolean NumPy arrays
np.save("train_mask.npy", train_mask)
np.save("val_mask.npy",  val_mask)

d_train_filtered = Subset(d_train, train_indices)
d_val_filtered   = Subset(d_val,   val_indices)

#Verify that only images with label of interests are taken into account

labels_train = set()
for i in tqdm(range(len(d_train_filtered))):
    _, lbl = d_train_filtered[i]
    labels_train.update(np.unique(lbl).tolist())

labels_val = set()
for i in tqdm(range(len(d_val_filtered))):
    _, lbl = d_val_filtered[i]
    labels_val.update(np.unique(lbl).tolist())

print("train unique:", sorted(labels_train))
print("val unique:",   sorted(labels_val))



print(len(d_train_filtered), len(d_val_filtered))


#Data Loader
train_dl = DataLoader(d_train_filtered, batch_size = 16, shuffle=True, num_workers=1)
val_dl = DataLoader(d_val_filtered, batch_size=16, num_workers=1)
#Sematic segmentation with deepl learning - model taken from ex 9
"""
We need pixel wise classification so semantic segmentation
Starting with small fully connected convolutional network that does per pixel classification with local context on feature map
"""

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
    
#REMAPPING TO MAKE LABEL FIT CUDA CONDITIONS

dataload_train_test =DataLoader(d_train_filtered, batch_size=2,shuffle=True, num_workers=1)
dataload_val_test =DataLoader(d_val_filtered, batch_size=2,shuffle=True, num_workers=1)

ORIG_IDS = [3, 4, 6, 9, 11, 12, 15, 18, 21, 24, 25, 30, 33]
ID_TO_IDX = {orig_id: i for i, orig_id in enumerate(ORIG_IDS)}
IDX_TO_ID = {i: orig_id for i, orig_id in enumerate(ORIG_IDS)}
def remap_labels_to_indices(target):
    # target: (B, H, W) with original IDs
    target_np = target.cpu().numpy()
    out = np.full_like(target_np, fill_value=-1)
    for orig_id, idx in ID_TO_IDX.items():
        out[target_np == orig_id] = idx
    return torch.from_numpy(out).to(target.device)

# debug: inspect raw label IDs
for data_dbg, target_dbg in dataload_train_test:
    print("RAW target min/max:", target_dbg.min(), target_dbg.max())
    t_remap = remap_labels_to_indices(target_dbg)
    print("REMAP target min/max:", t_remap.min(), t_remap.max())
    print("Unique remapped labels:", torch.unique(t_remap))
    break


#VERIFY MODEL WORKS  

def simple_model_check():
    B, C, H, W = 2, 64, 47, 47
    num_classes = 13

    model = PerPixelMLPWithContext(in_channels=C, hidden_dim=128, num_classes=num_classes)
    x = torch.randn(B, C, H, W)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (B, num_classes, H, W), \
        f"Expected {(B, num_classes, H, W)}, got {tuple(out.shape)}"

    print("OK: output shape =", tuple(out.shape))

if __name__ == "__main__":
    simple_model_check()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PerPixelMLPWithContext(in_channels=64, hidden_dim=128, num_classes=13).to(device)
model.eval()

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

  # set model to training mode. This is important because some layers behave differently during training and testing
  model.train(True)
  model.to(device)

  # stats
  loss_total = 0.0
  oa_total = 0.0

  # iterate over dataset
  for idx, (data, target) in enumerate(tqdm(data_loader)):

    data, target = data.to(device), target.to(device)
    data = data.permute(0, 3, 1, 2).contiguous()

    # reset gradients
    optimiser.zero_grad()

    # forward pass
    target = remap_labels_to_indices(target).long()
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
        target = remap_labels_to_indices(target).long()
        loss = criterion(pred, target)      # CrossEntropyLoss

      # stats update
        loss_total += loss.item()
        oa_total += torch.mean((pred.argmax(1) == target).float()).item()


  # normalise stats
  loss_total /= len(data_loader)
  oa_total /= len(data_loader)

  return loss_total, oa_total

import glob

os.makedirs('modeloutputs', exist_ok=True)

def load_model(epoch='latest'):
  model = PerPixelMLPWithContext()
  modelStates = glob.glob('modeloutputs/*.pth')
  if len(modelStates) and (epoch == 'latest' or epoch > 0):
    modelStates = [int(m.replace('modeloutputs/','').replace('.pth', '')) for m in modelStates]
    if epoch == 'latest':
      epoch = max(modelStates)
    stateDict = torch.load(open(f'modeloutputs/{epoch}.pth', 'rb'), map_location='cpu')
    model.load_state_dict(stateDict)
  else:
    # fresh model
    epoch = 0
  return model, epoch


def save_model(model, epoch):
  torch.save(model.state_dict(), open(f'modeloutputs/{epoch}.pth', 'wb'))

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
