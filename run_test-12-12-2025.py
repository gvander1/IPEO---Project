from models.lcamazon import LCAmazon
import torch
import torchvision.transforms as T
import matplotlib




#mean=torch.tensor([1169.2191,  917.3188,  847.3458,  739.8301, 1033.2438, 1957.2441, 2428.3120, 2375.6396, 2794.8320,  518.3082, 2045.9958, 1027.1027])
#std=torch.tensor([111.8526, 158.9845, 213.6219, 437.9030, 402.8841, 362.7410, 466.6497, 476.7180, 536.5454, 116.5597, 962.3427, 699.4785])
transforms = T.Compose([
    T.RandomVerticalFlip(),
    T.RandomHorizontalFlip(),
    T.Normalize(mean, std)
])

train=LCAmazon(root="DATA", modality="s2", split="train")
#val=LCAmazon(root="DATA", modality="s2", split="val")
#test=LCAmazon(root="DATA", modality="s2", split="test")
#print(f"train of length {len(train)}, val of length {len(val)}, test of length {len(test)}")
#print(len(test))


# compute mean and std for normalization later
n_pixels=0
channel_sum = torch.zeros(12)
channel_sum_sq = torch.zeros(12)
for x, _ in train_dl:
    x = x.float()
    b,h,w,c = x.shape
    # reshape to [B*H*W, C] a list basically
    x = x.view(-1, c)
    channel_sum += x.sum(dim=0)
    channel_sum_sq += (x ** 2).sum(dim=0)
    n_pixels += x.shape[0]
mean = channel_sum / n_pixels
std = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)
print(f"mean={mean}, std={s}")
