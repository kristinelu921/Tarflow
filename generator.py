import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformer_config import Config

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def generate_latents(self, model, num_samples, device = None):
        """
        Take samples from thelatent space
        Generate images
        """

        if device is None:
            device = next(model.parameters()).device

        model.eval()

        cfg = model.cfg
        #num_patches = cfg.num_patches + 1

        z = torch.randn(num_samples, cfg.num_patches, cfg.d_patch).to(device)

        with torch.no_grad():
            generated_images = model.decode(z)


        return generated_images
    
    def visualize_generated_images(self, images, nrow = 4, title = "Generated Images"):
        """
        Visualize generated images in a grid
        """

        if images.device.type != 'cpu':
            images = images.cpu()

        #make a grid
        from torchvision.utils import make_grid
        grid = make_grid(images, nrow = nrow, normalize = True)
        grid = grid.numpy()

        #convert to numpy
        grid_np = np.transpose(grid, (1, 2, 0))

        #Plot
        plt.figure(figsize = (10, 10))
        if grid_np.shape[2] == 1: # if grayscale
            plt.imshow(grid_np[:, :, 0], cmap = "gray")
        else:
            plt.imshow(grid_np)
        plt.title(title)
        plt.axis("off")
        plt.savefig(f'generated_{title.lower().replace(" ", "_")}.png')
        plt.show()

        