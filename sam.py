import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import cv2
import os
import numpy as np
checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
gt = Image.open('a3_half/images/1733301718254105000.jpg').convert("RGB")
gt = np.array(gt)
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
input_point = np.array([[250, 187]])
# 单点 prompt  输入格式为(x, y)和并表示出点所带有的标签1(前景点)或0(背景点)。
input_label = np.array([1])  # 点所对应的标签
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(gt)
    masks, _, _ = predictor.predict(point_coords=input_point,
    point_labels=input_label,)
    cv2.imwrite(os.path.join("test.jpg"), masks.transpose(1, 2, 0))