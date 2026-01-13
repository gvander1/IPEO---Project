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
import joblib
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

AE=np.load("mean_std/AE.npy") # Mean and std of AE-Embeddings
AE_mean, AE_std = AE[0], AE[1]
AE_mean, AE_std = torch.tensor(AE_mean), torch.tensor(AE_std)
AE_normalize = T.Normalize(AE_mean, AE_std)

@torch.no_grad() #without gradients
def forward_pass(dataset, model, device="cuda", num_classes=13, modality="s2", split="not-specified"):
    '''
    This function runs a forward pass on the data with selected models to
    1. Compute metrics (Confusion Matrix)
    2. Save prediction maps in modeloutputs/*_prediction
    '''
    model.eval()
    model.to(device)
    # initialize confusion  matrix
    overall_conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    # output directories
    output_dir = f"modeloutputs/{modality}_prediction"
    os.makedirs(output_dir, exist_ok=True)
    print(f"----------------- Computing predictions and metrics for {modality} model over {split} --------------------------")
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
            #forward pass
            outputs = model(x)
            pred = outputs["out"].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            x = AE_normalize(x)
            #forward pass
            pred = model(x)
            pred = pred.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        x.to(device)
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

def random_forest_eval(dataset, rf, scaler, split):
    all_preds = []
    all_labels = []
    print(f"----------------- Computing predictions and metrics for RF_AE model over {split} --------------------------")
    for i in tqdm(range(len(dataset))):
        img, lbl = dataset[i]
        img = img.astype(np.float32)
        #get original GT path to recover filename + metadata
        _, gt_path = dataset.samples[i]
        fname = os.path.basename(gt_path)
        os.makedirs("modeloutputs/AE_RF_prediction", exist_ok=True)
        out_path = os.path.join("modeloutputs/AE_RF_prediction", fname)
        H, W, C = img.shape
        X_img = img.reshape(-1, C)
        y_img = lbl.reshape(-1)
        # normalization
        X_img_scaled = scaler.transform(X_img)
        # prediction
        y_pred = rf.predict(X_img_scaled)
        # metrics
        all_preds.append(y_pred)
        all_labels.append(y_img)
        # reshape prediction to (H,W)
        pred = y_pred.reshape(H,W)
        #save prediction map
        with rasterio.open(gt_path) as src:
            profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(pred, 1)
    # metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    conf_mat = confusion_matrix(all_labels, all_preds)
    return conf_mat
    

# Function to save confusion matrix and metrics in the directory
def save_metrics(conf_mat, modality ="s2", split="non-specified"):
    class_names = ["Background/unknown", "Forest Formation", "Savanna Formation", "Floodable Forest",
                   "Forest Plantation", "Wetland", "Grassland", "Pasture", "Agriculture", "Urban Area",
                   "Other non Vegetated Areas", "Mining", "River/Lake/Ocean"]
    # Compute metrics
    # Recall (TP / row sum)
    recall = np.diag(conf_mat) / conf_mat.sum(axis=1).clip(min=1)
    # Precision (TP / column sum)
    precision = np.diag(conf_mat) / conf_mat.sum(axis=0).clip(min=1)
    # F1 score per class
    f1 = 2*(recall*precision)/(recall+precision+1e-12) #avoid division by zero
    # Over accuracy
    oa = np.trace(conf_mat) / conf_mat.sum()

    # Build confusion matrix for final output
    # add precision column
    final_conf_mat = np.hstack([conf_mat, precision.reshape(-1,1)])
    # add recall row
    recall_row = np.append(recall, np.nan) # add empty cell
    final_conf_mat = np.vstack([final_conf_mat, recall_row])
    # add F1 row
    f1_row = np.append(f1, np.nan)
    final_conf_mat = np.vstack([final_conf_mat, f1_row])
    # labels of classes
    xticks = class_names + ["Precision"]
    yticks = class_names + ["Recall", "F1"]
    # create annotation matrix
    annot = final_conf_mat.astype(object)
    # fill precision column (last column)
    for i in range(len(class_names)):
        annot[i, -1] = f"{precision[i]*100:.1f}%"
    # fill recall row
    for j in range(len(class_names)):
        annot[-2, j] = f"{recall[j]*100:.1f}%"
    # fill F1 row
    for j in range(len(class_names)):
        annot[-1, j] = f"{f1[j]*100:.1f}%"
    # replace NaN with empty string
    annot = np.where(np.isnan(final_conf_mat), "", annot)
    # Plot
    plt.figure(figsize=(22, 13))
    sns.heatmap(final_conf_mat, cmap="viridis", annot=annot, fmt="",
                norm=plt.matplotlib.colors.LogNorm(), xticklabels=xticks, yticklabels=yticks)
    plt.xlabel("Predicted class")
    plt.ylabel("Ground truth class")
    plt.title(f"Confusion Matrix and metrics for {modality} Model")
    # Add metics on the figure
    text_x = 1.25
    plt.gca().text(
        text_x, 0.5,
        f"Overall Accuracy: {oa:.4f}\n\n" 
        f"Mean User Accuracy (Precision): {np.nanmean(precision):.4f}\n"
        f"Mean Producer Accuracy (Recall): {np.nanmean(recall):.4f}\n"
        f"Mean F1 Score: {np.nanmean(f1):.4f}",
        transform=plt.gca().transAxes, fontsize=12, verticalalignment='center')
    plt.tight_layout()
    os.makedirs('modeloutputs/metrics', exist_ok=True)
    plt.savefig(f"modeloutputs/metrics/{modality}_{split}_confusion_matrix_.png", dpi=300)
    plt.close()



# Loading models
# Trained Sentinel-2 Model
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
state_dict_s2 = torch.load("models/s2/epoch19_fold4.pth", map_location="cpu")
model_s2.load_state_dict(state_dict_s2)

# Trained AE DL Model (from train_model_AE_DL)
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

model_AE = PerPixelMLPWithContext(in_channels=64, hidden_dim=256, num_classes=13)
# State of parameters saved in "models/AE/"
state_dict_AE = torch.load("models/AE/best.pth", map_location="cpu")
model_AE.load_state_dict(state_dict_AE)

# Trained Random Forest for AE
rf=joblib.load("models/AE_RF/rf_model.pkl")
scaler=joblib.load("models/AE_RF/scaler.pkl")

# DATA
# Sentinel-2
dataset_s2=LCAmazon(root="DATA", modality="s2", split="train_all", aug_geometric=False) #All train images the models have seen
test_dataset_s2=LCAmazon(root="DATA", modality="s2", split="test") #Images the model has never seen before
# AE
dataset_AE=LCAmazon(root="DATA", modality="AE", split="train_all", aug_geometric=False) #All train images the models have seen
test_dataset_AE=LCAmazon(root="DATA", modality="AE", split="test") #Images the model has never seen before
'''
conf_mat_s2=forward_pass(dataset_s2, model_s2, modality="s2", split="train_all")
conf_mat_AE=forward_pass(dataset_AE, model_AE, modality ="AE", split="train_all")
test_conf_mat_s2=forward_pass(test_dataset_s2, model_s2, modality="s2", split="test")
test_conf_mat_AE=forward_pass(test_dataset_AE, model_AE, modality ="AE", split="test")

# Save metrics
save_metrics(conf_mat_s2, modality="s2", split="train_all")
save_metrics(conf_mat_AE, modality="AE", split="train_all")
save_metrics(test_conf_mat_s2, modality="s2", split="test")
save_metrics(test_conf_mat_AE, modality="AE", split="test")
'''



conf_mat_AE_RF=random_forest_eval(dataset_AE, rf, scaler, split="train_all")
test_conf_mat_AE_RF=random_forest_eval(test_dataset_AE, rf, scaler, split="test")
save_metrics(conf_mat_AE_RF, modality="AE_RF", split="train_all")
save_metrics(test_conf_mat_AE_RF, modality="AE_RF", split="test")