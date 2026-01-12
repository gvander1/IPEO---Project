
# Targeting images to disregard because they don't appear in both train and test
import rasterio
import os
from glob import glob
import numpy as np
from models.lcamazon import LCAmazon

TARGET_CLASSES = {"Mangrove": 5,"Rocky Outcrop": 29,"Mosaic of Uses": 21}
# Hardcoded IDs of classes to discard
# If image contains it, the file name is added to the list image_to_discard
image_to_discard = []

label_paths = glob(os.path.join("DATA", "labels", "*.tif"))

for lbl_path in label_paths:
    with rasterio.open(lbl_path) as src:
        label = src.read(1)

    unique_vals = np.unique(label)

    for name, class_id in TARGET_CLASSES.items():
        if class_id in unique_vals:
            image_to_discard.append(os.path.basename(lbl_path))

image_to_discard= np.array(image_to_discard)
print("Images to discard saved in Images_to_discard.npy")
np.save("Images_to_discard.npy", image_to_discard)
