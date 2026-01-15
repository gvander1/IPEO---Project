# Script training a Random Forest on a subsample of pixels
# A limited number of pixels is taken for each class to deal with class imbalance

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # pip install joblib if not already there
from models.lcamazon import LCAmazon
from tqdm.auto import tqdm
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
    rf.fit(X_sub, y_sub) # Fitting the model
    
    # Compute training performance
    y_tr_pred = rf.predict(X_sub)
    print("=== TRAIN PERFORMANCE ===")
    print("Train accuracy:", accuracy_score(y_sub, y_tr_pred))
    print(classification_report(y_sub, y_tr_pred))

    # Compute validation performance
    y_val_pred = rf.predict(X_val)
    print("=== VAL PERFORMANCE ===")
    print("Val accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))
    
    return rf



dataset = LCAmazon(root="DATA", modality="AE", split="train")
val_set=LCAmazon(root="DATA", modality="AE", split="val")

# Subsampling
X_list, y_list = [], []
rng = np.random.default_rng(10)
max_pixels_total = 3_000_000

for i in tqdm(range(len(dataset))):
    img, lbl = dataset[i]  
    img = img.astype(np.float32)
    C, H, W = img.shape
    X_img = img.reshape(-1, W)
    y_img = lbl.reshape(-1)                        
    n_img = X_img.shape[0]
    if n_img == 0:
        continue
    n_take = min(n_img, 500)
    idx = rng.choice(np.arange(n_img), size=n_take, replace=False)
    assert y_img.shape[0] == n_img

    X_list.append(X_img[idx])
    y_list.append(y_img[idx])

    # Capping
    if sum(x.shape[0] for x in X_list) > max_pixels_total:
        break

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print("per image loop sampling done")

X_val_list, y_val_list = [], []

for i in tqdm(range(len(val_set))):
    img_v, lbl_v = val_set[i]   
    img_v = img_v.astype(np.float32) 
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
X = scaler.fit_transform(X)          
X_val = scaler.transform(X_val)

#call RF and train
rf = train_rf_from_flat(
    X, y,
    X_val, y_val,
    max_per_class=10000,
    n_estimators=300,
    min_samples_leaf=5,
    random_state=10,
    n_jobs=-1
)

# Saving the model to evaluate it later in metrics.py
os.makedirs("final_models/AE_RF_final", exist_ok=True)
joblib.dump(rf, "final_models/AE_RF_final/rf_model.pkl")
joblib.dump(scaler, "final_models/AE_RF_final/scaler.pkl")
    
