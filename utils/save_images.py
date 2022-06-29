# save images from [-1,1]

import numpy as np
from PIL import Image
import torch,os

def save_images(image, save_dir, start_index=0, max_index=np.inf):
    # image is a numpy array of shape NHWC
    # label is a numpy array of shape N
    image = ((image + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    image = image.permute(0, 2, 3, 1)
    image = image.contiguous()

    image = image.detach().cpu().numpy()
    # label = label.detach().cpu().numpy()
    os.makedirs(save_dir,exist_ok=True)
    for i in range(image.shape[0]):
        index = start_index + i
        save_single_image(image[i], index, save_dir, max_index=max_index)

def save_single_image(image, index, save_dir, max_index=np.inf):
    # image is a numpy array of shape HWC
    # image = normalize_and_quantize(image)
    if index < max_index:
        save_name = '%d_.png' % (index) 
        Image.fromarray(image).save(os.path.join(save_dir, save_name))