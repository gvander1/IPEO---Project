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
    rf=joblib.load("final_models/AE_RF_final/rf_model.pkl")
    scaler=joblib.load("final_models/AE_RF_final/scaler.pkl")
    return(rf, scaler)