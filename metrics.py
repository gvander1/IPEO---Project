# Script that loads our trained models and computes important metrics for performance comparison
# It also saves prediction maps in dedicated folder for later display using a jupyter notebook

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import rasterio
import os
import seaborn as sns
from models.lcamazon import LCAmazon
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from sklearn.metrics import confusion_matrix

#Checking if GPU is available
if torch.cuda.is_available() == 1:
    print("GPU is available")
else:
    print("GPU to be started !")

# Define necessary functions
# Normalization
s2=np.load("mean_std/s2.npy") # Mean and std of Sentinel-2
s2_mean, s2_std = s2[0], s2[1]
s2_mean, s2_std = torch.tensor(s2_mean), torch.tensor(s2_std)
s2_normalize = T.Normalize(s2_mean, s2_std)

AE=np.load("mean_std/s2.npy") # Mean and std of AE-Embeddings
AE_mean, AE_std = AE[0], AE[1]
AE_mean, AE_std = torch.tensor(AE_mean), torch.tensor(AE_std)
AE_normalize = T.Normalize(AE_mean, AE_std)


@torch.no_grad() #without gradients
def forward_pass(dataset, model, device="cuda", num_classes=13, modality="s2"):
    #we want to save the prediction maps of the trained model to plot it later for qualitative assessment
    model.eval()
    model.to(device)
    # initialize conf_mat
    overall_conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    # output directories
    output_dir = f"modeloutputs/{modality}_prediction"
    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Forward pass"):
        #load image and ground truth
        img, label = dataset[idx]
        #get original GT path to recover filename + metadata
        _, gt_path = dataset.samples[idx]
        fname = os.path.basename(gt_path)
        out_path = os.path.join(output_dir, fname)
        #prepare input and noramlize
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        if modality == "s2":
            x = s2_normalize(x)
        else:
            x = AE_normalize(x)
        x.to(device)
        #forward pass
        outputs = model(x)
        pred = outputs["out"].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        #update matrix
        y_true = label.flatten()
        y_pred = pred.flatten()
        current_conf = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        overall_conf_mat += current_conf
        #save prediction map
        with rasterio.open(gt_path) as src:
            profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(pred, 1)
    return overall_conf_mat


#Loading models
from torchvision.models.segmentation import fcn_resnet50
model_s2 = fcn_resnet50(progress=True, num_classes=13)  #we have 13 classes (including 0)
# The resnet18 model is made for 3 bands --> need to change the first convolution layer to fix this
model_s2.backbone.conv1 = torch.nn.Conv2d(
    in_channels=12,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)
# State of parameters saved in "models/s2/"
state_dict = torch.load("models/s2/epoch19_fold4.pth", map_location="cpu")
model_s2.load_state_dict(state_dict)

# DATA
dataset=LCAmazon(root="DATA", modality="s2", split="train_all", aug_geometric=False)
print(len(dataset))
conf_mat=forward_pass(dataset, model_s2, modality="s2")

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_mat,
    cmap="viridis",
    norm=plt.matplotlib.colors.LogNorm(),
    xticklabels=False,
    yticklabels=False
)
plt.xlabel("Predicted class")
plt.ylabel("Ground truth class")
plt.title("Validation Confusion Matrix (log scale)")
plt.tight_layout()
os.makedirs('modeloutputs', exist_ok=True)
plt.savefig("modeloutputs/confusion_matrix_BRAVOMAXIME.png")
