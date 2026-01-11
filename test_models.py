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

# TEST
test_s2 = LCAmazon(root="DATA", modality="s2", split="test", aug_geometric=False)
s2_dl = DataLoader(test_s2, batch_size=16, num_workers=1)

# Define necessary functions (same as in the training script)
# Normalization
read=np.load("mean_std/s2.npy")
mean, std = read[0], read[1]
mean=torch.tensor(mean)
std=torch.tensor(std)
normalize = T.Normalize(mean, std)

# Criterion
from torch.nn import CrossEntropyLoss
class_frequency = np.load("class_frequency.npy")
weights = torch.tensor(1/np.sqrt(class_frequency), dtype=torch.float32).to("cuda")  #weights are the inverse of the square root of the frequency of class in the dataset
criterion = CrossEntropyLoss(weight=weights)

# Confusion Matrix
def compute_confusion_matrix(pred, gt, num_classes):
    """
    pred, target: numpy arrays of shape (B, H, W)
    returns: (num_classes, num_classes) confusion matrix
    """
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    for gt, p in zip(gt, pred):
        conf_mat[gt, p] += 1
    return conf_mat

# Save predictions
@torch.no_grad() #without gradients
def save_predictions(dataset, model, device="cuda"):
    #we want to save the prediction maps of the trained model to plot it later for qualitative assessment
    model.eval()
    model.to(device)
    for idx in tqdm(range(len(dataset)), desc="Saving predictions"):
        img, _ = dataset[idx]
        #get original GT path to recover filename + metadata
        _, gt_path = dataset.samples[idx]
        fname = os.path.basename(gt_path)
        out_path = os.path.join("modeloutputs/s2_prediction", fname)
        #prepare input
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        x = normalize(x)
        x = x.to(device)
        #forward pass
        outputs = model(x)
        pred = outputs["out"].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
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

# Evaluation of the model on test set

@torch.no_grad()
def evaluate_test(test_dl, model, device="cuda"):
    model.eval()
    model.to(device)
    test_losses, test_accuracies = [], []
    test_confusion = np.zeros((13, 13), dtype=np.int64)

    for batch in tqdm(test_dl, desc="Testing"):
        x, y = batch
        x = x.permute(0, 3, 1, 2)
        x = normalize(x)
        x = x.to(device)
        y = y.to(device).long()

        outputs = model(x)
        y_hat = outputs["out"]
        loss = criterion(y_hat, y)
        preds = y_hat.argmax(1).cpu().numpy()
        gts = y.cpu().numpy()

        test_losses.append(loss.cpu().item())
        test_accuracies.append((preds == gts).mean())
        test_confusion += compute_confusion_matrix(preds, gts, num_classes=13)

    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_accuracies)
    return test_loss, test_acc, test_confusion


print("-------------- Evaluating Sentinel-2 model on TEST set -------------------")
test_loss, test_accuracy, test_confusion = evaluate_test(s2_dl, model_s2)
print(f"Test loss {test_loss:.2f}, test accuracy {test_accuracy*100:.2f}%")
plt.figure(figsize=(10, 8))
sns.heatmap(
    test_confusion,
    cmap="viridis",
    norm=plt.matplotlib.colors.LogNorm(),
    xticklabels=False,
    yticklabels=False,
)
plt.xlabel("Predicted class")
plt.ylabel("Ground truth class")
plt.title("Test Confusion Matrix (log scale)")
plt.tight_layout()
plt.savefig("modeloutputs/test_confusion_matrix.png")
print("Test confusion matrix saved as modeloutputs/test_confusion_matrix.png")
print("Saving TEST predictions...")
save_predictions(test_s2, model_s2)
print("Test predictions saved in modeloutputs/s2_prediction")
