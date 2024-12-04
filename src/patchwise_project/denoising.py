import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from fetch_dataset import *

class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU activation."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class ProgressiveFusionModel(nn.Module):
    """Progressive fusion model with attention mechanism."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 features: list = [64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention_layers = nn.ModuleDict()

        # Downsampling path
        in_features = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_features, feature))
            in_features = feature

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            self.attention_layers[str(feature * 2)] = nn.Sequential(
                nn.Conv2d(feature * 2, feature, kernel_size=1),
                nn.Sigmoid()
            )

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            concat_skip = torch.cat((skip, x), dim=1)
            attention_weights = self.attention_layers[str(concat_skip.shape[1])](concat_skip)
            x = torch.cat((attention_weights * skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

class ImageProcessor:
    """Handle image processing operations including patches and visualization."""
    @staticmethod
    def divide_into_patches(image: torch.Tensor, patch_size: int, stride: int) -> tuple:
        _, height, width = image.shape
        patches = []
        patch_locations = []

        for top in range(0, height - patch_size + 1, stride):
            for left in range(0, width - patch_size + 1, stride):
                patch = image[:, top:top+patch_size, left:left+patch_size]
                patches.append(patch)
                patch_locations.append((top, left))

        return torch.stack(patches), patch_locations

    @staticmethod
    def recombine_patches(patches: torch.Tensor, patch_locations: list, 
                         image_size: tuple, patch_size: int, stride: int) -> torch.Tensor:
        height, width = image_size
        full_image = torch.zeros((1, height, width), dtype=patches.dtype, device=patches.device)
        weight_map = torch.zeros((1, height, width), dtype=patches.dtype, device=patches.device)

        for i, (top, left) in enumerate(patch_locations):
            full_image[:, top:top+patch_size, left:left+patch_size] += patches[i]
            weight_map[:, top:top+patch_size, left:left+patch_size] += 1

        weight_map[weight_map == 0] = 1
        return full_image / weight_map

    @staticmethod
    def normalize_output(x: torch.Tensor, target_min: float = None, 
                        target_max: float = None) -> torch.Tensor:
        if target_min is None or target_max is None:
            return x
        
        x_min, x_max = x.min(), x.max()
        if x_max - x_min == 0:
            return torch.full_like(x, target_min)
        
        x_normalized = (x - x_min) / (x_max - x_min)
        return x_normalized * (target_max - target_min) + target_min

class Visualizer:
    """Handle all visualization tasks."""

    is_window_open = False

    @staticmethod
    def visualize_batch_patches(batch_data: torch.Tensor, predictions: torch.Tensor, 
                              num_levels: int, patch_size: int = 100, max_samples: int = 4):
        if Visualizer.is_window_open:
            return
            
        Visualizer.is_window_open = True
        batch_size = batch_data.shape[0] // num_levels
        samples_to_show = min(batch_size, max_samples)

        fig, axes = plt.subplots(samples_to_show, num_levels * 2, 
                               figsize=(1.5*num_levels*2, 1.5*samples_to_show))
        if samples_to_show == 1:
            axes = axes.reshape(1, -1)

        for sample_idx in range(samples_to_show):
            for level in range(num_levels):
                patch_idx = sample_idx * num_levels + level
                
                input_patch = batch_data[patch_idx, 0].cpu().numpy()
                ax_input = axes[sample_idx, level * 2]
                ax_input.imshow(input_patch, cmap='gray')
                ax_input.set_title(f"Input {sample_idx}\nLevel {level}")
                ax_input.axis("off")

                pred_patch = predictions[patch_idx, 0].detach().cpu().numpy()
                ax_pred = axes[sample_idx, level * 2 + 1]
                ax_pred.imshow(pred_patch, cmap='gray')
                ax_pred.set_title(f"Prediction {sample_idx}\nLevel {level}")
                ax_pred.axis("off")

        plt.tight_layout()
        plt.draw()
        plt.pause(5.0)  # Show for 1 second
        plt.close('all')
        Visualizer.is_window_open = False

    @staticmethod
    def visualize_denoising_results(original_image: torch.Tensor, 
                                  denoised_image: torch.Tensor, 
                                  denoised_normalized: torch.Tensor):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_image[0].cpu().numpy(), cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Denoised Image")
        plt.imshow(denoised_image[0].cpu().numpy(), cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Denoised and Normalized Image")
        plt.imshow(denoised_normalized[0].cpu().numpy(), cmap='gray')
        plt.axis("off")

        plt.tight_layout()
        plt.show()

class Trainer:
    """Handle model training and evaluation."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str = 'cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.visualizer = Visualizer()


    def train_epoch(self, train_loader: torch.utils.data.DataLoader, num_levels: int):
        self.model.train()
        
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = batch[0].to(self.device)

            for level in range(num_levels - 2):
                current_level_images = images[level::num_levels]
                target_level_images = images[level + 1::num_levels]

                pred = self.model(current_level_images)
                loss = F.mse_loss(pred, target_level_images)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

def denoise_image(model: nn.Module, image_path: str, patch_size: int = 64, stride: int = 32):
    """Denoise a single image using the trained model."""
    # Load and prepare image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = torch.from_numpy(image).float().unsqueeze(0)
    
    # Get the device from the model
    device = next(model.parameters()).device
    original_image = original_image.to(device)
    
    # Process image
    model.eval()
    processor = ImageProcessor()
    
    # Divide into patches and process
    patches, patch_locations = processor.divide_into_patches(original_image, patch_size, stride)
    patches = patches.to(device)  # Move patches to the same device as model
    
    with torch.no_grad():
        denoised_patches = model(patches)
    
    # Recombine and normalize
    denoised_image = processor.recombine_patches(denoised_patches, patch_locations, 
                                               original_image.shape[1:], patch_size, stride)
    original_min, original_max = original_image.min(), original_image.max()
    denoised_normalized = processor.normalize_output(denoised_image, original_min, original_max)
    
    # Move tensors back to CPU for visualization
    original_image = original_image.cpu()
    denoised_image = denoised_image.cpu()
    denoised_normalized = denoised_normalized.cpu()
    
    # Visualize results
    visualizer = Visualizer()
    visualizer.visualize_denoising_results(original_image, denoised_image, denoised_normalized)

def train(num_epochs=10, num_levels=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProgressiveFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, optimizer, device)
    
    train_loader, val_loader = get_loaders()
    
    for epoch in range(num_epochs):
        trainer.train_epoch(train_loader, num_levels)

def main():
    train(num_epochs=10)

if __name__ == "__main__":
    main()