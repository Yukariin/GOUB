import os
import cv2
import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils


########### denoising ###############
def add_noise(tensor, sigma):
    sigma = sigma / 255 if sigma > 1 else sigma
    return tensor + torch.randn_like(tensor) * sigma


######## inpainting ###########
def random_mask(height=256, width=256, pad=50,
                min_stroke=2, max_stroke=5,
                min_vertex=2, max_vertex=12,
                min_brush_width=7, max_brush_width=20,
                min_lenght=10, max_length=50):
    mask = np.zeros((height, width))

    max_angle = 2*np.pi
    num_stroke = np.random.randint(min_stroke, max_stroke+1)

    for _ in range(num_stroke):
        num_vertex = np.random.randint(min_vertex, max_vertex+1)
        brush_width = np.random.randint(min_brush_width, max_brush_width+1)
        start_x = np.random.randint(pad, height-pad)
        start_y = np.random.randint(pad, width-pad)

        for _ in range(num_vertex):
            angle = np.random.uniform(max_angle)
            length = np.random.randint(min_lenght, max_length+1)
            #length = np.random.randint(min_lenght, height//num_vertex)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)
            end_x = max(0, min(end_x, height))
            end_y = max(0, min(end_y, width))

            cv2.line(mask, (start_x, start_y), (end_x, end_y), 1., brush_width)

            start_x, start_y = end_x, end_y

    if np.random.random() < 0.5:
        mask = np.fliplr(mask)
    if np.random.random() < 0.5:
        mask = np.flipud(mask)
    
    mask = cv2.merge((mask,mask,mask))

    return mask.astype(np.float32)

def mask_to(tensor, mask_root='data/datasets/gt_keep_masks/genhalf', mask_id=-1, n=100):
    batch = tensor.shape[0]
    mask = random_mask(tensor.shape[2], tensor.shape[3])[None, ...]

    mask = torch.tensor(mask).permute(0, 3, 1, 2).float()
    # for images are clipped or scaled
    mask = F.interpolate(mask, size=tensor.shape[2:], mode='nearest')
    masked_tensor = mask * tensor
    return masked_tensor + (1. - mask)

######## super-resolution ###########

def upscale(tensor, scale=4, mode='bicubic'):
    tensor = F.interpolate(tensor, scale_factor=scale, mode=mode)
    return tensor




