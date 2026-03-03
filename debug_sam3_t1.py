from transformers import Sam3TrackerProcessor, Sam3TrackerModel
import torch
from PIL import Image
import requests

device = "cuda"

model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

image_url = "/mnt/shared-storage-user/intern7shared/liuyifei/code/posevlm/truck.jpg"
raw_image = Image.open(image_url).convert("RGB")

input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

# The model outputs multiple mask predictions ranked by quality score
print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")


import os
import numpy as np
from PIL import Image

save_dir = "./sam3_results"
os.makedirs(save_dir, exist_ok=True)

# masks: shape 通常是 [num_objects?, num_masks, H, W] 或 [1, num_masks, H, W]
# 你这里用的是 [0]，所以 masks.shape 很可能是 [num_masks, H, W] 或 [1, num_masks, H, W]
m = masks

# 统一成 [num_masks, H, W]
if m.ndim == 4:
    # 例如 [1, num_masks, H, W]
    m = m.squeeze(0)

num_masks = m.shape[0]
for i in range(num_masks):
    mask_i = m[i]

    # 转 numpy
    if hasattr(mask_i, "detach"):
        mask_i = mask_i.detach().cpu().numpy()
    else:
        mask_i = np.asarray(mask_i)

    # 若是 float，通常阈值化；若已经是 bool/0-1，仍然可以这样做
    mask_bin = (mask_i > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask_bin).save(os.path.join(save_dir, f"mask_{i:02d}.png"))

print(f"Saved {num_masks} masks to {save_dir}")
