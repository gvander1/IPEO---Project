from torch.utils.data import Dataset

from PIL import Image
import rasterio

import os
from glob import glob
import math
import random
import numpy as np


class LCAmazon(Dataset):
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
        self.transforms = transforms
        self.modality = modality
        self.split = split

        # build train samples
        train_labels = sorted(glob(os.path.join(root, "labels", "train_*.tif")))
        train_samples = []
        for lbl_path in train_labels:
            fname = os.path.basename(lbl_path)
            img_dir = "S2" if modality == "s2" else "AE"
            img_path = os.path.join(root, img_dir, fname)
            if os.path.exists(img_path):
                train_samples.append((img_path, lbl_path))

        self.samples = train_samples  # <--- store on self

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # load image (example with rasterio)
        with rasterio.open(img_path) as src:
            img = src.read()  # (C, H, W) np array

        # load label raster & convert to a single class ID as you need
        with rasterio.open(lbl_path) as src:
            lbl = src.read(1)

        # apply transforms if any
        if self.transforms is not None:
            img, lbl = self.transforms(img, lbl)

        return img, lbl

