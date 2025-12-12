from models.lcamazon import LCAmazon
import matplotlib

dataset=LCAmazon(root="DATA", modality="s2", split="train")
print(f"dataset of length {len(dataset)}")

