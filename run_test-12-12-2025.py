from models.lcamazon import LCAmazon

dataset=LCAmazon(root="DATA", split="val")
print(f"dataset of length {len(dataset)}")