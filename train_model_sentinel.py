# Script training a DL model on the training dataset using the GPU
# Mainly inspired by the ex8 Jupyter Notebook of the IPEO course
# Workflow: imports a resnet model, trains it on the train split, then validate it for each epoch
# After training is done saves the model as a .pth in the folder "models/s2/"
# To run the trained model on the data, run script called run_model_sentinel.py

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

#Checking if GPU is available
if torch.cuda.is_available() == 1:
    print("GPU is available")
else:
    print("GPU to be started !")

train_dataset = LCAmazon(root="DATA", modality="s2", split="train", aug_geometric=True)
val_dataset = LCAmazon(root="DATA", modality="s2", split="val", aug_geometric=False)

#Data Loader
train_dl = DataLoader(train_dataset, batch_size=16, num_workers=1)
val_dl = DataLoader(val_dataset, batch_size=16, num_workers=1)

#for image, label in train_dl:
#    print(image.shape, label.shape)
#    image has shape [16, 47, 47, 12] label has shape [16, 47, 47]

# Normalization
read=np.load("mean_std/s2.npy")
mean, std = read[0], read[1]
mean=torch.tensor(mean)
std=torch.tensor(std)
normalize = T.Normalize(mean, std)

#Loading of Computer Vision Models --> using torchvision, choose a fcn resnet50
from torchvision.models.segmentation import fcn_resnet50
model = fcn_resnet50(progress=True, num_classes=13)  #we have 13 classes (including 0)
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

# Model Training
# Loss function
# taking weighted cross-entropy with w(class)=1/sqrt(freq(class)) as criterion
from torch.nn import CrossEntropyLoss
class_frequency = np.load("class_frequency.npy")
weights = torch.tensor(1/np.sqrt(class_frequency), dtype=torch.float32).to("cuda")  #weights are the inverse of the square root of the frequency of class in the dataset
criterion = CrossEntropyLoss(weight=weights)

# Optimizer (use Stochastic Gradient Descent SGD --> actually AdamW is better)
from torch.optim import AdamW

learning_rate = 1e-4
weight_decay = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


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
    outputs = model(x)
    y_hat = outputs["out"]
    loss = criterion(y_hat, y)
    loss.backward() # backprop
    optimizer.step() # update model param
    #calculate accuracy
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
    model = model.to(device)
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
    # confusion matrix
    conf_mat = compute_confusion_matrix(predictions, ground_truth, num_classes=13)
    return loss, accuracy, conf_mat


def train_epoch(train_dl, val_dl, model, optimizer):
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    val_confusion = np.zeros((13, 13), dtype=np.int64)
    for batch in tqdm(train_dl):
        loss, accuracy = training_step(batch, model, optimizer)
        train_losses.append(loss.cpu().detach().numpy())
        train_accuracies.append(accuracy)
    for batch in tqdm(val_dl):
        loss, accuracy, conf_mat = prediction_step(batch, model)
        val_losses.append(loss.cpu().detach().numpy())
        val_accuracies.append(accuracy)
        val_confusion += conf_mat
    val_losses, train_losses, val_accuracies, train_accuracies = np.stack(val_losses).mean(), np.stack(train_losses).mean(), np.stack(val_accuracies).mean(), np.stack(train_accuracies).mean()
    return val_losses, train_losses, val_accuracies, train_accuracies, val_confusion

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

# Also compute confusion Matrix as important metric
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

def per_class_accuracy(conf_mat):
    correct = np.diag(conf_mat)
    total = conf_mat.sum(axis=1)
    acc = correct / np.maximum(total,1)
    return acc

def save_model(model, epoch, fold):
  os.makedirs('models/s2', exist_ok=True)
  torch.save(model.state_dict(), open(f'models/s2/epoch{epoch}_fold{fold}.pth', 'wb'))

# Model Training
num_folds = 5
num_epochs = 20 #computation time about 30 seconds/epoch
stats = [] #after 30 epochs we reach 93 % train accuracy with current hyperparameters, staying at 62% val accuracy (=overfitting !)
for i in range(num_folds):
    stats.append([])
print("-------------- Training the model and validate it at the same time -------------------")
for fold in range(num_folds):
    for epoch in range(num_epochs):
        train_set = LCAmazon(root="DATA", split="train", aug_geometric=True, num_folds=num_folds, fold_index=fold)
        val_set = LCAmazon(root="DATA", split="val", num_folds=num_folds, fold_index=fold)
        # build dataloaders
        train_dl = DataLoader(train_set, batch_size=8, shuffle=True)
        val_dl = DataLoader(val_set, batch_size=8)
        valloss, trainloss, valaccuracy, trainaccuracy, val_confusion = train_epoch(train_dl, val_dl, model, optimizer)
        #save_model(model, epoch)
        print(f"fold {fold+1}; epoch {epoch}; trainloss {trainloss:.2f}, train accuracy {trainaccuracy*100:.2f}%")
        print(f"fold {fold+1}; epoch {epoch}; valloss {valloss:.2f}, val accuracy {valaccuracy*100:.2f}%")
        stats[fold].append({
            "trainloss":float(trainloss),
            "trainaccuracy":float(trainaccuracy), 
            "valloss":float(valloss),
            "valaccuracy":float(valaccuracy),
            "valconfusion":val_confusion,
            "epoch":epoch
        })

print(f"-------------- Training done ; number of epochs = {num_epochs} -------------------")
#saving model
save_model(model, epoch, fold)

# Saving confusion matrix as a heatplot
conf_mat = stats[-1][-1]["valconfusion"]
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
plt.savefig("modeloutputs/confusion_matrix.png")


# Saving prediction maps
print("Saving training predictions...")
save_predictions(train_dataset, model)
print("Predictions saved in modeloutputs/s2_prediction")


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
    os.makedirs('modeloutputs/s2_accuracy', exist_ok=True)
    plt.savefig(f"modeloutputs/s2_accuracy/s2_train_val_accuracy_fold{fold+1}.png")
    print("Accuracy figure saved in modeloutputs/s2_accuracy/")


## IDEAS to improve model accuracy
# Try Adam or AdamW optimizer --> DONE
# Regularize through Dropouts or Data augmentation (add random rotations) using torchvision.transforms
# Add a pre-processing conv layer to reduce from 12 to 3 channels
# Use Weighted Cross-Entropy to deal with class imbalance
#       weights = torch.tensor([1.0, 2.0, 0.5, ...]).to(device)  # adjust per class
#       criterion = CrossEntropyLoss(weight=weights)

#TEST
test_dataset = LCAmazon(root="DATA", modality="s2", split="test", aug_geometric=False)
test_dl = DataLoader(test_dataset, batch_size=16, num_workers=1)
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
print("-------------- Evaluating on TEST set -------------------")
test_loss, test_accuracy, test_confusion = evaluate_test(test_dl, model)
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
save_predictions(test_dataset, model)
print("Test predictions saved in modeloutputs/s2_prediction")
