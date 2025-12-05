from torch.utils.data import Dataset

from PIL import Image

import os
import glob

class LCAmazon(Dataset):
    # mapping between label class names and indices
    #based on codigos da legenda.csv file 
    LABEL_CLASSES = {
        "Forest": 1,
        "Forest Formation": 2,
        "Savanna Formation": 3,
        "Mangrove": 5,
        "Floodable Forest": 5, 
        "Wooded Sandbank Vegetation": 49, 
        "Herbaceous and Shrubby Vegetation": 10,
        "Wetland": 11, 
        "Grassland": 12, 
        "Hypersaline Tidal Flat": 32, 
        "Rocky Outcrop": 29
        }
    return None 

    