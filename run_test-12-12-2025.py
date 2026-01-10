from models.lcamazon import LCAmazon
import torch
import torchvision.transforms as T
import matplotlib


transforms = T.Compose([
    T.RandomVerticalFlip(),
    T.RandomHorizontalFlip(),
    T.Normalize(mean, std)
])

#train=LCAmazon(root="DATA", modality="s2", split="train")
#val=LCAmazon(root="DATA", modality="s2", split="val")
test=LCAmazon(root="DATA", modality="s2", split="test")
#print(f"train of length {len(train)}, val of length {len(val)}, test of length {len(test)}")
print(len)