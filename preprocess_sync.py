import os
import glob
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import rembg


def chop_image_into_16(image):
    # Assuming 'image' is a cv2 image
    height, width, _ = image.shape

    # Calculating the width of each slice
    slice_width = width // 16

    # Slicing the image into 16 pieces
    slices = [image[:, i*slice_width:(i+1)*slice_width] for i in range(16)]

    return slices

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--model', default='u2net', type=str, help="rembg model, see https://github.com/danielgatis/rembg#models")
    parser.add_argument('--size', default=512, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=bool, default=False, help="recenter, potentially not helpful for multiview zero123")    
    opt = parser.parse_args()

    session = rembg.new_session(model_name=opt.model)

    if os.path.isdir(opt.path):
        print(f'[INFO] processing directory {opt.path}...')
        files = glob.glob(f'{opt.path}/*')
        out_dir = opt.path
    else: # isfile
        files = [opt.path]
        out_dir = os.path.dirname(opt.path)
 
    os.makedirs(out_dir,exist_ok=True)
    for file in files:
        if file.endswith('jpg') or  file.endswith('png') and not '_rgba.png' in file:
            out_base = os.path.basename(file).split('.')[0]

            # load image
            print(f'[INFO] loading image {file}...')
            image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

            slices = chop_image_into_16(image)
            
            for idx, image in enumerate(slices):
                # carve background
                print(f'[INFO] background removal...')
                carved_image = rembg.remove(image, session=session) # [H, W, 4]
                mask = carved_image[..., -1] > 0
                # else:
                final_rgba = carved_image
                
                # write image
                out_rgba = os.path.join(opt.path, out_base + f'_{idx}_rgba.png')
                cv2.imwrite(out_rgba, final_rgba)
                print('out path:',out_rgba)
