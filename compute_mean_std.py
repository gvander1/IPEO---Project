# compute mean and std for normalization later
import torch
import numpy as np
import os
from torch.utils.data import ConcatDataset
from models.lcamazon import LCAmazon
from tqdm.auto import tqdm

def compute_mean_std(dataset, channels):
    n_pixels=0
    channel_sum = np.zeros(channels, dtype=np.float64)
    channel_sum_sq = np.zeros(channels, dtype=np.float64)
    for x, _ in tqdm(dataset):
        # x is (H, W, C)
        x = x.astype(np.float64)
        h,w,c = x.shape
        # reshape to [H*W, C] a list basically
        x = x.reshape(-1, c)

        channel_sum += x.sum(axis=0)
        channel_sum_sq += (x ** 2).sum(axis=0)
        n_pixels += x.shape[0]
    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sum_sq / n_pixels - mean ** 2)
#    print(f"mean={mean}, std={std}")
    return mean, std

s2_train=LCAmazon(root="DATA", modality="s2", split="train")
s2_val=LCAmazon(root="DATA", modality="s2", split="val")
sentinel=ConcatDataset([s2_train,s2_val])
channels = 12

print("--------------- Computing mean and std of sentinel-2 images -----------------")
s2_mean, s2_std = compute_mean_std(sentinel, channels)
s2_mean_std = np.array([s2_mean, s2_std])
os.makedirs('mean_std', exist_ok=True)
np.save("mean_std/s2.npy", s2_mean_std)
print("--------------- Successfully saved in mean_std/s2.npy ----------------")

AE_train=LCAmazon(root="DATA", modality="AE", split="train")
AE_val=LCAmazon(root="DATA", modality="AE", split="val")
AE_embeddings=ConcatDataset([AE_train,AE_val])
channels = 64

print("--------------- Computing mean and std of AE Embeddings -----------------")
AE_mean, AE_std = compute_mean_std(AE_embeddings, channels)
AE_mean_std = np.array([AE_mean, AE_std])
np.save("mean_std/AE.npy", AE_mean_std)
print("--------------- Successfully saved in mean_std/AE.npy ----------------")