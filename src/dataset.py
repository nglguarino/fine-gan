import os
import glob
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FFHQDataset(Dataset):
    """Custom PyTorch Dataset for loading FFHQ images."""
    def __init__(self, img_dir, transform=None, num_images=None): # Add num_images parameter
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_images (int, optional): Number of images to use. If None, use all images.
        """
        # Find all files with .png or .jpg extension
        self.img_paths = glob.glob(os.path.join(img_dir, '*.png'))
        self.img_paths.extend(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.transform = transform

        if num_images:
            # If a number is specified, shuffle all paths and take a random subset
            random.shuffle(self.img_paths)
            self.img_paths = self.img_paths[:num_images]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def create_mask(image, mask_percentage=0.025):
    """
    Creates masks that are more likely to cover facial features.
    """
    batch_size, _, height, width = image.shape
    mask = torch.ones_like(image)

    # Face regions typically in center 60% of image
    center_bias = 0.3  # 30% border on each side

    for i in range(batch_size):
        mask_h = int(np.sqrt(height * width * mask_percentage))
        mask_w = mask_h

        # Bias towards center
        top_min = int(height * center_bias)
        top_max = int(height * (1 - center_bias)) - mask_h
        left_min = int(width * center_bias)
        left_max = int(width * (1 - center_bias)) - mask_w

        top = np.random.randint(top_min, max(top_min + 1, top_max))
        left = np.random.randint(left_min, max(left_min + 1, left_max))

        mask[i, :, top:top+mask_h, left:left+mask_w] = 0

    masked_image = image * mask
    return masked_image, mask

