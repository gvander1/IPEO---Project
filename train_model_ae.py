import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # pip install joblib if not already there
from models.lcamazon import LCAmazon
from tqdm.auto import tqdm
from label_proportion import label_proportions
from sklearn.preprocessing import StandardScaler

#Function class aware subsampling + random forest fitting

def train_rf_from_flat(
    X_tr, y_tr,
    X_val, y_val,
    max_per_class=5000,
    n_estimators=200,
    random_state=10,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=1,
    class_weight=None,
):
    # class-aware subsample on flat y_tr
    rng = np.random.default_rng(random_state)
    X_sub_list, y_sub_list = [], []
    classes, counts = np.unique(y_tr, return_counts=True)
    print("Full train class counts:", dict(zip(classes, counts)))

    for cls in classes:
        idx_cls = np.where(y_tr == cls)[0]
        n_take = min(len(idx_cls), max_per_class)
        chosen = rng.choice(idx_cls, n_take, replace=False)
        X_sub_list.append(X_tr[chosen])
        y_sub_list.append(y_tr[chosen])

    X_sub = np.concatenate(X_sub_list, axis=0)
    y_sub = np.concatenate(y_sub_list, axis=0)
    print("Subsample class counts:", dict(zip(*np.unique(y_sub, return_counts=True))))

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rf.fit(X_sub, y_sub)
    # --- training performance ---
    y_tr_pred = rf.predict(X_sub)
    print("=== TRAIN PERFORMANCE ===")
    print("Train accuracy:", accuracy_score(y_sub, y_tr_pred))
    print(classification_report(y_sub, y_tr_pred))

    # --- validation performance ---
    y_val_pred = rf.predict(X_val)
    print("=== VAL PERFORMANCE ===")
    print("Val accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    return rf



# 1) Create the dataset with AE features

dataset = LCAmazon(root="DATA", modality="AE", split="train")
val_set=LCAmazon(root="DATA", modality="AE", split="val")


prop = label_proportions(dataset) # variable pas utilisée pour l'instant je crois

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

#normalize 
scaler = StandardScaler()
X = scaler.fit_transform(X)          # train features [web:155][web:162]
X_val = scaler.transform(X_val)

#call RF
rf = train_rf_from_flat(
    X, y,
    X_val, y_val,
    max_per_class=10000,
    n_estimators=300,
    min_samples_leaf=5,
    random_state=10,
    n_jobs=-1
)


#TEST
# 1) Create test dataset
test_set = LCAmazon(root="DATA", modality="AE", split="test")
# Later for test:
proptest= label_proportions(test_set)

# 2) Flatten test pixels
X_test_list, y_test_list = [], []

for i in tqdm(range(len(test_set))):
    img_t, lbl_t = test_set[i]      # img_t: (64, 47, 47), lbl_t: (47, 47)
    img_t = img_t.astype(np.float32)
    C, H, W = img_t.shape
    X_img_t = img_t.reshape(-1, W)
    y_img_t = lbl_t.reshape(-1)

    X_test_list.append(X_img_t)
    y_test_list.append(y_img_t)

X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)
#normalize test set
X_test = scaler.transform(X_test)

print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
y_pred_test = rf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))



