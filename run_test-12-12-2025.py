from models.lcamazon import LCAmazon
import matplotlib

train=LCAmazon(root="DATA", modality="s2", split="train")
val=LCAmazon(root="DATA", modality="s2", split="val")
test=LCAmazon(root="DATA", modality="s2", split="test")
print(f"train of length {len(train)}, val of length {len(val)}, test of length {len(test)}")
