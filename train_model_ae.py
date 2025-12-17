import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # pip install joblib if not already there
from models.lcamazon import LCAmazon

def train_rf_per_pixel_many(
    features,        # (N, 64, 47, 47)
    labels,          # (N, 47, 47)
    ignore_label=None,
    max_samples=200_000,
    n_estimators=200,
    random_state=10,
    n_jobs=-1
):
    """
    Train a per-pixel RandomForestClassifier from many feature images
    of shape (64, 47, 47) and label maps of shape (47, 47).

    features : np.ndarray, shape (N, 64, 47, 47)
    labels   : np.ndarray, shape (N, 47, 47)
    Random forest va classifier pixel par pixel. 1 pixel dans AE = 1 vecteur de 64 dimension associé
    à un pixel avec la classe correspondante dans le label
    1 sample c'est enfait un vecteur, c'est a dire que Random for traite pixels par pixel
    dans le cas de notre dataset on a en quelques sorte 5000x45x45 training exaample en quelque sortes
    """

    feats = np.asarray(features)   # (N, 64, 47, 47)
    lbls  = np.asarray(labels)     # (N, 47, 47)

    assert feats.ndim == 4 and feats.shape[1] == 64, "features must be (N,64,47,47)"
    N, C, H, W = feats.shape
    assert lbls.shape == (N, H, W), "labels must be (N,47,47)"

    # (N, 64, 47, 47) -> (N, H, W, C)
    feats_nhwc = np.transpose(feats, (0, 2, 3, 1))   # (N, 47, 47, 64)

    # Flatten across all images and pixels
    X = feats_nhwc.reshape(-1, C)   # (N*H*W, 64)
    y = lbls.reshape(-1)            # (N*H*W,)

    # Optionally drop ignore_label pixels
    if ignore_label is not None:
        mask = (y != ignore_label)
        X = X[mask]
        y = y[mask]

    # Optionally subsample pixels (to limit memory/time)
    rng = np.random.default_rng(random_state)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight="balanced_subsample"
    )
    rf.fit(X_tr, y_tr)

    print(classification_report(y_val, rf.predict(X_val)))
    return rf


# 1) Create the dataset with AE features
dataset = LCAmazon(root="DATA", modality="AE", split="train")

# 2) Iterate over the dataset and collect all (img, lbl)
features_list = []
labels_list = []

for i in range(len(dataset)):
    img, lbl = dataset[i]   # img: (64, 47, 47), lbl: (47, 47)
    features_list.append(np.array(img))   # ensure numpy
    labels_list.append(np.array(lbl))

# 3) Stack into arrays suitable for train_rf_per_pixel_many
features_all = np.stack(features_list, axis=0)   # (N, 64, 47, 47)
labels_all   = np.stack(labels_list, axis=0)     # (N, 47, 47)

rf = train_rf_per_pixel_many(
    features_all,   # (N, 64, 47, 47)
    labels_all,     # (N, 47, 47)
    ignore_label=None,      # or e.g. 0 or 27 if you want to ignore a class
    max_samples=200000,
    n_estimators=200,
    random_state=10,
    n_jobs=-1
)
joblib.dump(rf, "rf_ae_model.joblib")
print("Saved model to rf_ae_model.joblib")





