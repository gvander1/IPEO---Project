from torch.utils.data import Dataset

from PIL import Image

import os
from glob import glob
import math
import random

class LCAmazon(Dataset):
    # mapping between label class names and indices
    #based on codigos da legenda.csv file 
    LABEL_CLASSES = {
    "Forest": 1,
    "Forest Formation": 3,
    "Savanna Formation": 4,
    "Mangrove": 5,
    "Floodable Forest": 6,
    "Wooded Sandbank Vegetation": 49,
    "Herbaceous and Shrubby Vegetation": 10,
    "Wetland": 11,
    "Grassland": 12,
    "Hypersaline Tidal Flat": 32,
    "Rocky Outcrop": 29,
    "Herbaceous Sandbank Vegetation": 50,
    "Farming": 14,
    "Pasture": 15,
    "Agriculture": 18,
    "Temporary Crop": 19,
    "Soybean": 39,
    "Sugar cane": 20,
    "Rice": 40,
    "Cotton (beta)": 62,
    "Other Temporary Crops": 41,
    "Perennial Crop": 36,
    "Coffee": 46,
    "Citrus": 47,
    "Palm Oil": 35,
    "Other Perennial Crops": 48,
    "Forest Plantation": 9,
    "Mosaic of Uses": 21,
    "Non vegetated area": 22,
    "Beach, Dune and Sand Spot": 23,
    "Urban Area": 24,
    "Mining": 30,
    "Other non Vegetated Areas": 25,  
    "Water": 26,
    "River, Lake and Ocean": 33,
    "Aquaculture": 31,
    "Not Observed": 27,
}
    ID_TO_NAME = {v: k for k, v in LABEL_CLASSES.items()}
    def __init__(self, root, modality="s2", split="train", transforms=None,
                 val_ratio=0.15, seed=42):
        """
        split: "train", "val", or "test"
        """
        self.transforms = transforms
        self.modality = modality
        self.split = split

        # 1) build all TRAIN samples from filenames train_*.tif
        train_labels = sorted(glob(os.path.join(root, "labels", "train_*.tif")))
        train_samples = []
        for lbl_path in train_labels:
            fname = os.path.basename(lbl_path)
            img_dir = "S2" if modality == "s2" else "AE"
            img_path = os.path.join(root, img_dir, fname)
            if os.path.exists(img_path):
                train_samples.append((img_path, lbl_path))

        # 2) deterministic shuffle (so train/val are fixed)
        rnd = random.Random(seed)
        rnd.shuffle(train_samples)

        n_total = len(train_samples)        # should be 5000
        n_val = int(math.floor(val_ratio * n_total))  # 15%
        n_train = n_total - n_val

        train_subset = train_samples[:n_train]
        val_subset   = train_samples[n_train:]

        # 3) build TEST samples from test_*.tif
        test_labels = sorted(glob(os.path.join(root, "labels", "test_*.tif")))
        test_samples = []
        for lbl_path in test_labels:
            fname = os.path.basename(lbl_path)
            img_dir = "S2" if modality == "s2" else "AE"
            img_path = os.path.join(root, img_dir, fname)
            if os.path.exists(img_path):
                test_samples.append((img_path, lbl_path))

        if split == "train":
            self.samples = train_subset
        elif split == "val":
            self.samples = val_subset
        elif split == "test":
            self.samples = test_samples
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path)
        label = Image.open(lbl_path)
        if self.transforms is not None:
            img, label = self.transforms(img, label)
        return img, label
    
