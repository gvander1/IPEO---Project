from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, accuracy_score
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.lcamazon import LCAmazon
from tqdm.auto import tqdm
import pandas as pd
from train_model_ae.py import train_rf_from_flat

def spatial_cv_rf(dataset, n_splits=5, **rf_kwargs):
    # groups = tile ids; here simply indices 0..len(dataset)-1
    groups = np.arange(len(dataset))
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(groups, groups=groups)):
        # build flat train from train tiles
        X_tr_list, y_tr_list = [], []
        for i in train_idx:
            img, lbl = dataset[i]
            img = img.astype(np.float32)
            C, H, W = img.shape
            X_img = img.reshape(-1, W)
            y_img = lbl.reshape(-1)

            # your per-image subsampling
            n_img = X_img.shape[0]
            n_take = min(n_img, 500)
            idx = np.random.choice(np.arange(n_img), size=n_take, replace=False)
            X_tr_list.append(X_img[idx])
            y_tr_list.append(y_img[idx])

        X_tr = np.concatenate(X_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)

        # build flat val from val tiles
        X_val_list, y_val_list = [], []
        for i in val_idx:
            img_v, lbl_v = dataset[i]
            img_v = img_v.astype(np.float32)
            C, H, W = img_v.shape
            X_img_v = img_v.reshape(-1, W)
            y_img_v = lbl_v.reshape(-1)
            X_val_list.append(X_img_v)
            y_val_list.append(y_img_v)

        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        # class-aware subsampling + RF
        rf = train_rf_from_flat(
            X_tr, y_tr,
            X_val, y_val,
            **rf_kwargs
        )

        y_pred = rf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        fold_results.append({"fold": fold, "accuracy": acc,
                             "macro_f1": report["macro avg"]["f1-score"],
                             "macro_recall": report["macro avg"]["recall"]})

    return pd.DataFrame(fold_results)

dataset = LCAmazon(root="DATA", modality="AE", split="train")
cv_results = spatial_cv_rf(dataset, n_splits=5,
                           max_per_class=5000,
                           n_estimators=200,
                           random_state=10,
                           n_jobs=-1)
print(cv_results)
print(cv_results.mean(numeric_only=True))
