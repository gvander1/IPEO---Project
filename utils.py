import torch

# Loading models
# Trained Sentinel-2 Model
def load_model_sentinel(model_path):
    import torch
    import torch.nn as nn
    from torchvision.models.segmentation import fcn_resnet50
    model_s2 = fcn_resnet50(progress=True, num_classes=13)  #we have 13 classes (including 0)
    # The resnet18 model is made for 3 bands --> need to change the first convolution layer to fix this
    model_s2.backbone.conv1 = torch.nn.Conv2d(
        in_channels=12,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False)
    # State of parameters saved in "models/s2/"
    state_dict_s2 = torch.load(model_path, map_location="cpu")
    model_s2.load_state_dict(state_dict_s2)
    return model_s2

# Trained AE DL Model (from train_model_AE_DL)
def load_model_AE(model_path):
    import torch
    import torch.nn as nn
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

                nn.Conv2d(hidden_dim, num_classes, 1),)

        def forward(self, x):
            return self.net(x)

    model_AE = PerPixelMLPWithContext(in_channels=64, hidden_dim=256, num_classes=13)
    # State of parameters saved in "models/AE/"
    state_dict_AE = torch.load(model_path, map_location="cpu")
    model_AE.load_state_dict(state_dict_AE)
    return model_AE

# Trained Random Forest for AE
def load_model_AE_RF():
    import joblib
    rf=joblib.load("models/AE_RF/rf_model.pkl")
    scaler=joblib.load("models/AE_RF/scaler.pkl")
    return(rf, scaler)


@torch.no_grad() #without gradients
def inference(dataset, idx, model, modality="s2", scaler=None):
    import tqdm
    import numpy as np
    import torchvision.transforms as T
    # Only runs on CPU because of the limited amount of images to predict
    model.eval()
    predictions = []
    for id in tqdm(idx, desc=f"Forward pass of {modality} model"):
        #load image and ground truth
        print(idx)
        print(id)
        img, label = dataset[id]
        #prepare input and noramlize
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        #normalization
        s2=np.load("mean_std/s2.npy") # Mean and std of Sentinel-2
        s2_mean, s2_std = s2[0], s2[1]
        s2_mean, s2_std = torch.tensor(s2_mean), torch.tensor(s2_std)
        s2_normalize = T.Normalize(s2_mean, s2_std)
        AE=np.load("mean_std/AE.npy") # Mean and std of AE-Embeddings
        AE_mean, AE_std = AE[0], AE[1]
        AE_mean, AE_std = torch.tensor(AE_mean), torch.tensor(AE_std)
        AE_normalize = T.Normalize(AE_mean, AE_std) 
        #inference
        if modality == "s2":
            x = s2_normalize(x)
            #forward pass
            outputs = model(x)
            pred = outputs["out"].argmax(1).squeeze(0).numpy().astype(np.uint8)
        elif modality == "AE":
            x = AE_normalize(x)
            #forward pass
            pred = model(x)
            pred = pred.argmax(1).squeeze(0).numpy().astype(np.uint8)
        elif modality == "AE_RF":
            x = img.astype(np.float32)
            x = scaler.transform(x)
            #prediction
            pred = model.predict(x)
        else:
            print("Modality must be 'AE', 'AE_RF' or 's2'")
        predictions.append(pred)
    return predictions
