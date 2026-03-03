from PIL import Image
import requests
import torch
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "danelcsb/sam3_tracker.1_hiera_tiny"
model = Sam3TrackerModel.from_pretrained(model_id).to(device)
processor = Sam3TrackerProcessor.from_pretrained(model_id)

raw_image = Image.open("/mnt/shared-storage-user/intern7shared/liuyifei/code/posevlm/truck.jpg").convert("RGB")
input_points = [[[[400, 650]]]]     # 这套嵌套是 tracker 的格式（batch / point_batch / num_points / 2）
input_labels = [[[1]]]

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)
outputs = model(**inputs)

# 注意：outputs.pred_masks 是 5D (B,PBS,NM,H,W)，要先 reshape 成 4D 再 postprocess，避免你之前的 interpolate 坑
pred = outputs.pred_masks.detach().cpu()
b, pbs, nm, h, w = pred.shape
pred_4d = pred.view(b, pbs * nm, h, w)
masks = processor.post_process_masks(pred_4d, inputs["original_sizes"].detach().cpu())
