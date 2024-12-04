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

class FusionDataset:
    def __init__(self, basedir, size, transform=None):
        self.transform = transform if transform else transforms.ToTensor()
        self.transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        self.image_groups = []
        self.size = size
        
        # Iterate through each patient/directory
        for p in os.listdir(basedir):
            level_0_dir = f"{basedir}/{p}/FusedImages_Level_0"
            if not os.path.exists(level_0_dir):
                continue
                
            # Get number of levels for this patient
            num_levels = sum(1 for d in os.listdir(f"{basedir}/{p}") 
                           if d.startswith("FusedImages_Level_"))
            
            # Process each base image
            for base_idx in range(len(os.listdir(level_0_dir))):
                paths = []
                names = []
                current_idx = base_idx
                valid_group = True
                
                # Level 0
                name = f"Fused_Image_Level_0_{base_idx}.tif"
                path = f"{basedir}/{p}/FusedImages_Level_0/{name}"
                if not os.path.exists(path):
                    continue
                    
                paths.append(path)
                names.append(name)
                
                # Higher levels
                for level in range(1, num_levels):
                    current_idx = current_idx // 2
                    name = f"Fused_Image_Level_{level}_{current_idx}.tif"
                    path = f"{basedir}/{p}/FusedImages_Level_{level}/{name}"
                    
                    # Check if this path exists
                    if not os.path.exists(path):
                        valid_group = False
                        break
                        
                    paths.append(path)
                    names.append(name)
                
                # Only add complete groups
                if valid_group:
                    self.image_groups.append((paths, names))

    def _apply_transform(self, img):
        """Applies the transformation pipeline to an image."""
        img = Image.fromarray(img)  # Convert NumPy array to PIL Image
        img = self.transform(img)  # Apply transformation (e.g., ToTensor)
        img = img.unsqueeze(0)  # Add batch dimension
        #img = F.interpolate(img, size=(128, 128), mode="bilinear", align_corners=False)
        img = img.squeeze(0)  # Remove batch dimension
        return img
    
    def __len__(self):
        return len(self.image_groups)
    
    def __getitem__(self, idx):
        paths, names = self.image_groups[idx]
        images = []
        
        for path in paths:
            # Read image
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
                
            # Apply transform
            img = self._apply_transform(img)
            images.append(img)
        
        # Stack images along a new dimension
        stacked_images = torch.stack(images)
        
        return [stacked_images, names]  # Return both images and names
    
def get_dataset(size=512):
    basedir = "../FusedDataset"

    dataset = FusionDataset(basedir, size=size)

    return dataset

def get_loaders(dataset):

    train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.95), int(len(dataset)*0.05)+1])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    print(len(train_loader), len(val_loader))

    return train_loader, val_loader

def visualise_loader(loader):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(10, 10))

    for i, (images, names) in enumerate(loader):
        print(images.shape)
        print(names)

        num_images = images.shape[1]
        images_to_plot = [images[0, j].squeeze(0).numpy() for j in range(num_images)]
        
        while len(images_to_plot) < 9:
            images_to_plot.append(np.zeros_like(images_to_plot[0]))
        
        for idx in range(9):
            ax = fig.add_subplot(3, 3, idx + 1) 
            ax.imshow(images_to_plot[idx], cmap="gray")
            ax.axis('off') 
        
        break 
    plt.tight_layout()
    plt.show()
