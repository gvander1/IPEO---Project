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

def eval_rf_grid(
    X_tr, y_tr,
    X_val, y_val,
    max_depth_list=(5, 10, 15, None),
    n_estimators_list=(50, 100, 200),
    max_per_class=5000,
    random_state=10,
    n_jobs=-1,
):
    rng_global = np.random.default_rng(random_state)
    results = []

    # fixed class-aware subsample once per (X_tr, y_tr) to keep it consistent
    classes = np.unique(y_tr)

    def subsample_class_aware(X_src, y_src):
        X_sub_list, y_sub_list = [], []
        for cls in classes:
            idx_cls = np.where(y_src == cls)[0]
            if len(idx_cls) == 0:
                continue
            n_take = min(len(idx_cls), max_per_class)
            chosen = rng_global.choice(idx_cls, n_take, replace=False)
            X_sub_list.append(X_src[chosen])
            y_sub_list.append(y_src[chosen])
        X_sub = np.concatenate(X_sub_list, axis=0)
        y_sub = np.concatenate(y_sub_list, axis=0)
        return X_sub, y_sub

    X_sub, y_sub = subsample_class_aware(X_tr, y_tr)

    for max_depth, n_estimators in itertools.product(max_depth_list, n_estimators_list):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
        rf.fit(X_sub, y_sub)

        y_pred = rf.predict(X_val)

        # scalar metrics
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(
            y_val, y_pred, output_dict=True, zero_division=0
        )
        macro_f1 = report["macro avg"]["f1-score"]
        macro_rec = report["macro avg"]["recall"]

        # per-class F1 / recall stats (excluding "accuracy", "macro avg", etc.)
        f1_scores = []
        recalls = []
        for k, v in report.items():
            if k in ["accuracy", "macro avg", "weighted avg"]:
                continue
            f1_scores.append(v["f1-score"])
            recalls.append(v["recall"])

        f1_scores = np.array(f1_scores)
        recalls = np.array(recalls)

        results.append({
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_recall": macro_rec,
            "f1_min": f1_scores.min(),
            "f1_max": f1_scores.max(),
            "f1_mean": f1_scores.mean(),
            "f1_std": f1_scores.std(),
            "recall_min": recalls.min(),
            "recall_max": recalls.max(),
            "recall_mean": recalls.mean(),
            "recall_std": recalls.std(),
        })

    df_results = pd.DataFrame(results)
    return df_results

# 1) Create the dataset with AE features
dataset = LCAmazon(root="DATA", modality="AE", split="train")
val_set=LCAmazon(root="DATA", modality="AE", split="val")

# prop = label_proportions(dataset)
#2) --- SUBSAMPLING
# 2.a per-image lookp 

X_list, y_list = [], []
rng = np.random.default_rng(10)
max_pixels_total = 3_000_000

for i in tqdm(range(len(dataset))):
    img, lbl = dataset[i]          # img: (47,47,64), lbl: (47,47)
    img = img.astype(np.float32)
    C, H, W = img.shape
    X_img = img.reshape(-1, W)  # (H*W, 64)
    y_img = lbl.reshape(-1)                              # (H*W,)

    n_img = X_img.shape[0]
    if n_img == 0:
        continue

    n_take = min(n_img, 500)
    idx = rng.choice(np.arange(n_img), size=n_take, replace=False)
    assert y_img.shape[0] == n_img


    X_list.append(X_img[idx])
    y_list.append(y_img[idx])

    #optional hard cap on global size
    if sum(x.shape[0] for x in X_list) > max_pixels_total:
        break

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print("per image loop sampling done")
print("X shape:", X.shape, "y shape:", y.shape)
print(len(np.unique(y)))

#build flat validation data from val_set


X_val_list, y_val_list = [], []

for i in tqdm(range(len(val_set))):
    img_v, lbl_v = val_set[i]   # img_v: (64, 47, 47), lbl_v: (47, 47)
    img_v = img_v.astype(np.float32)  # ensure float32
    C, H, W = img_v.shape
    X_img_v = img_v.reshape(-1, W)
    y_img_v = lbl_v.reshape(-1)

    X_val_list.append(X_img_v)
    y_val_list.append(y_img_v)

X_val = np.concatenate(X_val_list, axis=0)
y_val = np.concatenate(y_val_list, axis=0)
print("val set extraction done")
print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)
print(np.unique(y_val))

df_rf = eval_rf_grid(
    X, y,
    X_val, y_val,
    max_depth_list=[5, 10, 15, None],
    n_estimators_list= [50, 100, 200],
    max_per_class=5000,
    random_state=10,
    n_jobs=-1,
)

print(df_rf.sort_values("accuracy", ascending=False))
df_rf.head(15)
df_rf.to_csv("acc_for_hp_RF", index=False)
