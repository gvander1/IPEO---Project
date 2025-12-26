### Function computing the proportion of each label class over the dataset to cope with rare classes (or even absent ones)
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from models.lcamazon import LCAmazon

def label_proportions(dataset):
    """
    dataset: e.g. LCAmazon(..., split='train'), where dataset[i] -> (img, lbl)
             and lbl has shape (H, W) with integer class ids.
    """
    print("---------- Computing proporion of each class in the dataset -----------------")
    all_labels = []

    for i in tqdm(range(len(dataset))):
        _, lbl = dataset[i]          # ignore image, keep label map
        all_labels.append(lbl.ravel())

    labels_flat = np.concatenate(all_labels, axis=0)  # shape: (N_pixels,)
    classes, counts = np.unique(labels_flat, return_counts=True)
    total = labels_flat.size
    proportions = counts / total

    df = pd.DataFrame({
        "class_id": classes,
        "pixel_count": counts,
        "proportion": proportions,
    })
    return df

train=LCAmazon(root="DATA", modality="s2", split="train")
test=LCAmazon(root="DATA", modality="s2", split="test")
val=LCAmazon(root="DATA", modality="s2", split="val")
print(label_proportions(val))