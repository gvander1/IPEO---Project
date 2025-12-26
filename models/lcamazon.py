from torch.utils.data import Dataset

from PIL import Image
import rasterio

import os
from glob import glob
import math
import random
import numpy as np
from scipy.ndimage import rotate

images_to_discard = np.load("Images_to_discard.npy")

class LCAmazon(Dataset):
    # mapping between label class names and indices
    # based on codigos da legenda.csv file 
    LABEL_CLASSES = {
#    "Forest": 1, #not in dataset
    "Forest Formation": 3, 
    "Savanna Formation": 4,
 #   "Mangrove": 5, #only in test
    "Floodable Forest": 6,
#    "Wooded Sandbank Vegetation": 49, #not in dataset
#    "Herbaceous and Shrubby Vegetation": 10, #not in dataset
    "Wetland": 11,
    "Grassland": 12,
#    "Hypersaline Tidal Flat": 32, #not in dataset
#    "Rocky Outcrop": 29, #only in train
#    "Herbaceous Sandbank Vegetation": 50, #not in dataset
#    "Farming": 14, #not in dataset
    "Pasture": 15,
    "Agriculture": 18,
#    "Temporary Crop": 19, #not in dataset
#    "Soybean": 39, #not in dataset
#    "Sugar cane": 20, #not in dataset
#    "Rice": 40, #not in dataset
#    "Cotton (beta)": 62, #not in dataset
#    "Other Temporary Crops": 41, #not in dataset
#    "Perennial Crop": 36, #not in dataset
#    "Coffee": 46, #not in dataset
#    "Citrus": 47, #not in dataset
#    "Palm Oil": 35, #not in dataset
#    "Other Perennial Crops": 48, #not in dataset
    "Forest Plantation": 9,
#    "Mosaic of Uses": 21, #only in train
#    "Non vegetated area": 22, #not in dataset
#    "Beach, Dune and Sand Spot": 23, #not in dataset
    "Urban Area": 24,
    "Mining": 30,
    "Other non Vegetated Areas": 25,  
#    "Water": 26, #not in dataset
    "River, Lake and Ocean": 33,
#    "Aquaculture": 31, #not in dataset
#    "Not Observed": 27, #not in dataset
}
    ID_TO_NAME = {v: k for k, v in LABEL_CLASSES.items()}
    def __init__(self, root, modality="s2", split="train", transforms=None,
                 val_ratio=0.15, seed=42):
        """
        split: "train", "val", or "test" ---- modality: "s2" for sentinel "AE" else
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
                if not fname in images_to_discard: #discard classes that are not in both train and test
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
                if not fname in images_to_discard: #discard classes that are not in both train and test
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

        # Read multispectral S2 image
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)  # (C, H, W)

        # Read label mask (single channel)
        with rasterio.open(lbl_path) as src:
            label = src.read(1).astype(np.int32)
        # la ligne suivante cause une transposition cheloue qui fait qu'on doit à nouveau transposer pour les passer à nos models
        # If transforms expect (H, W, C), transpose:
        img = np.transpose(img, (1, 2, 0))  # -> (H, W, C)

        if self.transforms is not None:
            img, label = self.transforms(img, label)

        return img, label
"""
    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        with rasterio.open(img_path) as src:
            # returns array shape (bands, H, W)
            img = src.read().astype(np.float32)
        with rasterio.open(lbl_path) as src:
            label = src.read(1)  # read single band
        if self.transforms is not None:
            img, label = self.transforms(img, label)
        return img, label
"""

class RandomRotate:
    def __init__(self, angles=(0,90,180,270), p=0.5):
        self.angles = angles
        self.p = p
    def __call__(self, img, label):
        if random.random()>self.p:
            return img, label
        else:
            angle=random.choice(self.angles)
            #rotating image
            img = rotate(img, angle, order=1, mode="reflect")
            #rotating label as well (important !)
            label = rotate(label, angle, order=0, mode="constant")
        return img, label
