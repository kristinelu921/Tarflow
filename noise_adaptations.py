import torch
import torch.nn as nn
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def add_noise(images, cfg):
    """
    Adds noise to the images for training
    images: (bsize, channels, height, width)
    cfg: transformer config
    std: standard dev of the noise
    """
    std = cfg.noise_std
    noise = torch.randn_like(images) * std
    noisy_images = images + noise
    return noisy_images

