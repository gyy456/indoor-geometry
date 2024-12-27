import torch
from PIL import Image

# Load an image
import os

# 设置文件夹路径
folder_path = 'pic'

# 获取文件夹下所有的 .jpg 文件
jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)
# 打印结果
# print(jpg_files)
predictor = predictor.to('cuda')
for jpg in jpg_files:

    input_image = Image.open(os.path.join(folder_path,jpg))

    # Create predictor instance

    # Apply the model to the image
    delight_image = predictor(input_image)

    # Save or display the result
    delight_image.save("pic/pic_delight/"+jpg)