from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from tqdm import tqdm
import random

import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import torch
import numpy as np

class FusionDataset(Dataset):
    def __init__(self, basedir, size, patch_size=100, patches_per_image=4):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)), 
            transforms.ToTensor()
        ])
        self.size = size
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.image_groups = self._load_image_groups(basedir)
        # Generate fixed patch locations at init
        self.patch_locations = self._generate_patch_locations()
        
    def _generate_patch_locations(self):
        """Generate fixed patch locations that will be used across all images"""
        locations = []
        for _ in range(self.patches_per_image):
            top = random.randint(0, self.size - self.patch_size)
            left = random.randint(0, self.size - self.patch_size)
            locations.append((top, left))
        return locations

    def _load_image_groups(self, basedir):
        image_groups = []
        for p in os.listdir(basedir):
            level_0_dir = os.path.join(basedir, p, "FusedImages_Level_0")
            if not os.path.exists(level_0_dir):
                continue
            
            num_levels = len([d for d in os.listdir(os.path.join(basedir, p)) 
                              if d.startswith("FusedImages_Level_")])
            
            for base_idx in range(len(os.listdir(level_0_dir))):
                paths = []
                valid_group = True
                # Start with level 0 image
                paths.append(os.path.join(level_0_dir, f"Fused_Image_Level_0_{base_idx}.tif"))
                current_idx = base_idx
                
                # Add paths from other levels
                for level in range(1, num_levels):
                    current_idx //= 2
                    path = os.path.join(basedir, p, f"FusedImages_Level_{level}",
                                        f"Fused_Image_Level_{level}_{current_idx}.tif")
                    if not os.path.exists(path):
                        valid_group = False
                        break
                    paths.append(path)
                
                if valid_group:
                    image_groups.append(paths)
        return image_groups

    def _apply_transform(self, img):
        img = Image.fromarray(img)
        return self.transform(img)

    def __len__(self):
        return len(self.image_groups) * self.patches_per_image

    def __getitem__(self, idx):
        group_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        
        paths = self.image_groups[group_idx]
        images = []
        
        # Load and transform all images in the group
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            img = self._apply_transform(img)  # Shape: [1, size, size]
            images.append(img)
        
        # Stack all images in the group
        stacked_images = torch.stack(images)  # Shape: [num_levels, 1, size, size]
        
        # Get the pre-generated patch location for this patch_idx
        top, left = self.patch_locations[patch_idx]
        
        # Extract the patch from all levels
        # Note: we want to keep the channel dimension
        patch = stacked_images[:, :, top:top + self.patch_size, left:left + self.patch_size]
        
        return patch, patch.clone()


def collate_fn(batch):
    # Each item in batch is [num_levels, 1, patch_size, patch_size]
    patches = torch.cat([item[0] for item in batch], dim=0)
    targets = torch.cat([item[1] for item in batch], dim=0)
    return patches, targets

def visualize_batch_patches(batch_data, num_levels, patch_size=100, max_samples=4):
    """
    Visualize patches from a batch of data.
    
    Args:
        batch_data (torch.Tensor): Tensor of shape [batch_size * num_levels, 1, patch_size, patch_size]
        num_levels (int): Number of pyramid levels
        patch_size (int): Size of patches
        max_samples (int): Maximum number of samples to display
    """
    # Reshape the batch to separate levels
    batch_size = batch_data.shape[0] // num_levels
    samples_to_show = min(batch_size, max_samples)
    
    # Create a figure
    fig, axes = plt.subplots(samples_to_show, num_levels, figsize=(3*num_levels, 3*samples_to_show))
    if samples_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(samples_to_show):
        for level in range(num_levels):
            # Get the patch for this sample and level
            patch_idx = sample_idx * num_levels + level
            patch = batch_data[patch_idx, 0].numpy()  # Remove channel dimension
            
            # Plot the patch
            ax = axes[sample_idx, level]
            im = ax.imshow(patch, cmap='gray')
            ax.set_title(f'Sample {sample_idx}\nLevel {level}')
            ax.axis('off')
    
    plt.tight_layout()
    return fig

def visualize_full_images_with_patches(dataset, sample_indices=[0]):
    """
    Visualize full images with highlighted patch locations for specific samples.
    
    Args:
        dataset (FusionDataset): The dataset instance
        sample_indices (list): List of sample indices to visualize
    """
    for idx in sample_indices:
        group_idx = idx // dataset.patches_per_image
        patch_idx = idx % dataset.patches_per_image
        
        # Load the full images from the group
        paths = dataset.image_groups[group_idx]
        images = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            img = dataset._apply_transform(img)
            images.append(img)
        
        # Create figure
        fig, axes = plt.subplots(1, len(images), figsize=(4*len(images), 4))
        if len(images) == 1:
            axes = [axes]
        
        # Get patch location
        top, left = dataset.patch_locations[patch_idx]
        
        for level, (ax, img) in enumerate(zip(axes, images)):
            # Plot full image
            ax.imshow(img[0].numpy(), cmap='gray')
            
            # Draw rectangle around patch
            rect = plt.Rectangle((left, top), dataset.patch_size, dataset.patch_size, 
                               fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            ax.set_title(f'Level {level}\nPatch {patch_idx}')
            ax.axis('off')
        
        plt.tight_layout()
    
    return fig

def get_loaders():
    basedir = "../FusedDataset"
    dataset = FusionDataset(basedir, size=512, patch_size=100, patches_per_image=20)
    print(f"Total dataset length: {len(dataset)}")

    train_size = int(len(dataset) * 0.95)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, 
        batch_size=8,
        collate_fn=collate_fn,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=8,
        collate_fn=collate_fn
    )

    # Verify shapes
    #sample_batch = next(iter(train_loader))
    #print(f"Batch shape: {sample_batch[0].shape}")

    batch_data, _ = next(iter(train_loader))

    # Calculate number of levels from the data
    num_levels = len(dataset.image_groups[0])

    # Visualize patches from the batch
    #fig1 = visualize_batch_patches(batch_data, num_levels)
    #plt.show()

    # Visualize full images with patch locations
    #fig2 = visualize_full_images_with_patches(dataset, sample_indices=[0, 1])
    #plt.show()

    return dataset, train_loader, val_loader

if __name__ == "__main__":
    main()