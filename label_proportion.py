### Function computing the proportion of each label class over the dataset to cope with rare classes (or even absent ones)
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from models.lcamazon import LCAmazon
from torch.utils.data import ConcatDataset

def label_proportions(dataset):
    # Computing the frequency of each class within the dataset
    print("---------- Computing frequency of each class in the dataset -----------------")
    all_labels = []

    for i in tqdm(range(len(dataset))):
        _, lbl = dataset[i]          # ignore image, keep label map
        all_labels.append(lbl.ravel())

    labels_flat = np.concatenate(all_labels, axis=0)  # shape: (N_pixels,)
    classes, counts = np.unique(labels_flat, return_counts=True)
    total = labels_flat.size
    proportions = counts / total
    class_mapping = {
        # mapping classes using the new indices from 1 to 12 instead of indices from the original dataset
        new_id: class_name
        for class_name, old_id in LCAmazon.LABEL_CLASSES.items()
        if old_id in LCAmazon.LABEL_REMAP
        for new_id in [LCAmazon.LABEL_REMAP[old_id]]
    }
    class_mapping[0] = "Background/unknown"
    names = [class_mapping.get(c, f"Unknown ({c})") for c in classes]
    df = pd.DataFrame({
        "class_id": classes,
        "class_name": names,
        "pixel_count": counts,
        "proportion": proportions,
    })
    proportions= np.array(proportions)
    np.save("class_frequency.npy", proportions)
    print(df)
    print("--------------- Class frenquencies saved in class_frequency.npy -------------")
    return proportions #returning proportions instead of df


train=LCAmazon(root="DATA", modality="s2", split="train")
val=LCAmazon(root="DATA", modality="s2", split="val")
dataset=ConcatDataset([train,val])
label_proportions(dataset)