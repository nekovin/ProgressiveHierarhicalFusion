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

def analyze_structural_features(x, window_size=5, contrast_threshold=0.5, intensity_variation_threshold=0.01):
    """
    Analyzes both high contrast structures and subtle intensity variations.
    Returns a mask where 1 indicates areas to preserve and 0 indicates pure noise.
    """
    batch_size, channels, height, width = x.shape
    
    # 1. Detect intensity variations using standard deviation in local windows
    padding = window_size // 2
    mean_filter = torch.ones(channels, 1, window_size, window_size,
                           device=x.device) / (window_size * window_size)
    
    # Compute local statistics
    local_mean = F.conv2d(x, mean_filter, padding=padding, groups=channels)
    local_var = F.conv2d(x**2, mean_filter, padding=padding, groups=channels) - local_mean**2
    local_std = torch.sqrt(local_var + 1e-6)
    
    # 2. Detect structural patterns using directional gradients
    directions = [
        [[-1, 0, 1],  # Horizontal
         [-1, 0, 1],
         [-1, 0, 1]],
        [[-1, -1, -1],  # Vertical
         [0, 0, 0],
         [1, 1, 1]],
        [[-1, -1, 0],  # Diagonal 1
         [-1, 0, 1],
         [0, 1, 1]],
        [[0, -1, -1],  # Diagonal 2
         [1, 0, -1],
         [1, 1, 0]]
    ]
    
    # Convert kernels to tensors and stack them
    kernels = torch.stack([torch.tensor(d, dtype=x.dtype, device=x.device).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
                          for d in directions])
    
    # Apply directional filters
    pattern_responses = []
    for kernel in kernels:
        response = F.conv2d(x, kernel, padding=1, groups=channels)
        pattern_responses.append(response)
    
    # Combine directional responses
    pattern_strength = torch.stack(pattern_responses).abs().max(dim=0)[0]
    
    # Create preservation masks
    structure_mask = (pattern_strength > contrast_threshold).float()  # High contrast features
    variation_mask = (local_std > intensity_variation_threshold).float()  # Subtle variations
    
    # Combine masks - preserve area if either condition is met
    preservation_mask = torch.clamp(structure_mask + variation_mask, 0, 1)
    
    # Additional check for very dark areas with no structure
    intensity_mask = (x > 0.1).float()  # Basic intensity threshold
    final_mask = preservation_mask * intensity_mask
    
    return final_mask

def adaptive_noise_suppression(x, preservation_mask):
    """
    Applies noise suppression based on the preservation mask.
    Gradually suppresses noise while preserving structural features.
    """
    # Normalize input to [0,1] range
    x_min = x.min()
    x_max = x.max()
    x_normalized = (x - x_min) / (x_max - x_min + 1e-6)
    
    # Create graduated suppression factor
    suppression_factor = preservation_mask
    
    # Apply adaptive suppression
    x_denoised = x_normalized * (suppression_factor + 0.1)  # Add small offset to prevent total blackout
    
    # Areas marked as pure noise (mask = 0) get normalized to 0
    noise_areas = (preservation_mask < 0.1).float()
    x_denoised = x_denoised * (1 - noise_areas)
    
    # Rescale back to original range
    x_denoised = x_denoised * (x_max - x_min) + x_min
    
    return x_denoised



def compute_cnr(signal, background):
    """Simple function to compute CNR."""
    mean_signal = signal.mean()
    mean_background = background.mean()
    std_background = background.std()
    cnr = abs(mean_signal - mean_background) / (std_background + 1e-6)  # Avoid division by zero
    return cnr

def high_freq_loss(x):
    # Sobel filters for capturing high frequency details
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    
    hf_x = F.conv2d(x, sobel_x, padding=1)
    hf_y = F.conv2d(x, sobel_y, padding=1)
    high_freq_map = torch.sqrt(hf_x**2 + hf_y**2 + 1e-6)
    
    # Dark region mask (low intensity regions)
    dark_mask = (x < 0.3).float()  # Areas with intensity < 0.3 (adjust threshold if needed)
    
    # Emphasize high frequencies in dark regions
    dark_high_freq_map = high_freq_map * dark_mask
    return dark_high_freq_map

def fusion_loss_v3(outputs, targets, lambda_levels=0.5, lambda_edge=0.1, lambda_ssim=0.3, lambda_contrast=0.2):
    """Enhanced loss function with SSIM and contrast matching"""
    level_losses = []
    edge_losses = []
    ssim_losses = []
    contrast_losses = []
    
    # Sobel filters for edge detection  
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    
    def compute_contrast(x):
        # Local contrast
        mean_local = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        variance = F.avg_pool2d((x - mean_local) ** 2, kernel_size=3, stride=1, padding=1)
        
        # Global contrast
        mean_global = torch.mean(x)
        std_global = torch.std(x)
        
        return variance, mean_global, std_global
    
    for i, pred in enumerate(outputs['level_outputs']):
        target = targets[:, i, :, :, :]
        
        # Regular reconstruction loss
        level_losses.append(F.mse_loss(pred, target))
        
        # Edge preservation loss
        pred_edges_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, sobel_y, padding=1)
        target_edges_x = F.conv2d(target, sobel_x, padding=1)
        target_edges_y = F.conv2d(target, sobel_y, padding=1)
        
        edge_loss = F.mse_loss(pred_edges_x, target_edges_x) + F.mse_loss(pred_edges_y, target_edges_y)
        edge_losses.append(edge_loss)
        
        # SSIM loss
        #ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
        # Modify SSIM calculation in fusion_loss_v3
        ssim_val = ssim(pred, target, data_range=max(1.0, torch.max(target)-torch.min(target)), size_average=True)
        ssim_losses.append(1 - ssim_val)
        
        pred_contrast = compute_contrast(pred)
        target_contrast = compute_contrast(target)
        
        contrast_loss = (F.mse_loss(pred_contrast[0], target_contrast[0]) +  # Local contrast
                        F.mse_loss(pred_contrast[1], target_contrast[1]) +  # Mean
                        F.mse_loss(pred_contrast[2], target_contrast[2]))   # Standard deviation
        contrast_losses.append(contrast_loss)
    
    level_loss = sum(level_losses) / len(level_losses)
    edge_loss = sum(edge_losses) / len(edge_losses)
    ssim_loss = sum(ssim_losses) / len(ssim_losses)
    contrast_loss = sum(contrast_losses) / len(contrast_losses)
    
    # Final output losses
    final_output = outputs['final_output']
    final_target = targets[:, -1, :, :, :]
    
    final_loss = F.mse_loss(final_output, final_target)
    final_ssim_loss = 1 - ssim(final_output, final_target, data_range=1.0, size_average=True)
    
    # Final contrast loss
    final_pred_contrast = compute_contrast(final_output)
    final_target_contrast = compute_contrast(final_target)
    final_contrast_loss = (F.mse_loss(final_pred_contrast[0], final_target_contrast[0]) +
                          F.mse_loss(final_pred_contrast[1], final_target_contrast[1]) +
                          F.mse_loss(final_pred_contrast[2], final_target_contrast[2]))
    
    detail_loss = F.l1_loss(
        high_freq_loss(outputs['final_output']), 
        high_freq_loss(targets[:, -1, :, :, :])
    )
    
    # Combine all losses
    total_loss = (final_loss + 
                 lambda_levels * level_loss + 
                 lambda_edge * (edge_loss + detail_loss) + 
                 lambda_ssim * (ssim_loss + final_ssim_loss) +
                 lambda_contrast * (contrast_loss + final_contrast_loss))
    
    return total_loss

def fusion_loss_v3_fixed(outputs, targets, lambda_final=0.4, lambda_levels=0.2, lambda_edge=0.1, lambda_ssim=0.2, lambda_contrast=0.1):
    """Fixed loss function with properly weighted components"""
    level_losses = []
    edge_losses = []
    ssim_losses = []
    contrast_losses = []
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    
    def compute_contrast(x):
        mean_local = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        variance = F.avg_pool2d((x - mean_local) ** 2, kernel_size=3, stride=1, padding=1)
        
        mean_global = torch.mean(x)
        std_global = torch.std(x)
        
        return variance, mean_global, std_global
    
    for i, pred in enumerate(outputs['level_outputs']):
        target = targets[:, i, :, :, :]
        
        level_losses.append(F.mse_loss(pred, target))
        
        pred_edges_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, sobel_y, padding=1)
        target_edges_x = F.conv2d(target, sobel_x, padding=1)
        target_edges_y = F.conv2d(target, sobel_y, padding=1)
        
        edge_loss = F.mse_loss(pred_edges_x, target_edges_x) + F.mse_loss(pred_edges_y, target_edges_y)
        edge_losses.append(edge_loss)
        
        # SSIM loss
        ssim_val = ssim(pred, target, data_range=max(1.0, torch.max(target)-torch.min(target)), size_average=True)
        ssim_losses.append(1 - ssim_val)
        
        # Contrast matching loss
        pred_contrast = compute_contrast(pred)
        target_contrast = compute_contrast(target)
        
        contrast_loss = (F.mse_loss(pred_contrast[0], target_contrast[0]) +  # Local contrast
                        F.mse_loss(pred_contrast[1], target_contrast[1]) +  # Mean
                        F.mse_loss(pred_contrast[2], target_contrast[2]))   # Standard deviation
        contrast_losses.append(contrast_loss)
    
    level_loss = sum(level_losses) / len(level_losses)
    edge_loss = sum(edge_losses) / len(edge_losses)
    ssim_loss = sum(ssim_losses) / len(ssim_losses)
    contrast_loss = sum(contrast_losses) / len(contrast_losses)
    
    final_output = outputs['final_output']
    final_target = targets[:, -1, :, :, :]
    
    final_loss = F.mse_loss(final_output, final_target)
    final_ssim_loss = 1 - ssim(final_output, final_target, data_range=1.0, size_average=True)
    
    final_pred_contrast = compute_contrast(final_output)
    final_target_contrast = compute_contrast(final_target)
    final_contrast_loss = (F.mse_loss(final_pred_contrast[0], final_target_contrast[0]) +
                          F.mse_loss(final_pred_contrast[1], final_target_contrast[1]) +
                          F.mse_loss(final_pred_contrast[2], final_target_contrast[2]))
    
    detail_loss = F.l1_loss(
        high_freq_loss(outputs['final_output']), 
        high_freq_loss(targets[:, -1, :, :, :])
    )
    
    assert abs(lambda_final + lambda_levels + lambda_edge + lambda_ssim + lambda_contrast - 1.0) < 1e-6, \
        "Lambda coefficients must sum to 1.0"
    
    total_loss = (lambda_final * final_loss + 
                 lambda_levels * level_loss + 
                 lambda_edge * (edge_loss + detail_loss) + 
                 lambda_ssim * (ssim_loss + final_ssim_loss) +
                 lambda_contrast * (contrast_loss + final_contrast_loss))
    
    return total_loss

class SequentialProgressiveFusionNetwork(nn.Module):
    def __init__(self, num_fusion_levels=1):
        super().__init__()
        self.num_fusion_levels = num_fusion_levels
        self.channels = 64
        
        # Initial feature extraction
        self.init_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, self.channels, 3, padding=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU()
            ) for _ in range(num_fusion_levels)
        ])
        
        # Feature fusion
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.channels * (i + 2), self.channels, 1),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, 3, padding=1),
                nn.ReLU()
            ) for i in range(num_fusion_levels)
        ])
        
        # Decoders
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 3, padding=1)
            ) for _ in range(num_fusion_levels)
        ])
        
        self.fusion_weights = nn.Parameter(torch.ones(num_fusion_levels) / num_fusion_levels)
        
    def forward(self, x):
        all_levels = [x[:, i, :, :, :] for i in range(self.num_fusion_levels)]
        target = x[:, -1, :, :, :]
        input_min, input_max = target.min(), target.max()
        
        # Initial features 
        level_features = [self.init_convs[i](level) for i, level in enumerate(all_levels)]
        
        current_features = level_features[0]
        level_outputs = []
        
        for i in range(self.num_fusion_levels):

            if i > 0:
                fusion_input = torch.cat([current_features] + level_features[:i+1], dim=1)
            else:
                fusion_input = torch.cat([current_features, level_features[0]], dim=1)
            
            current_features = self.feature_fusion[i](fusion_input)
            
            level_output = self.decoders[i](current_features)
            level_output = self.normalize_output(level_output, input_min, input_max)
            level_outputs.append(level_output)
        
        weights = F.softmax(self.fusion_weights, dim=0)
        final_output = sum(w * out for w, out in zip(weights, level_outputs))
        final_output = self.normalize_output(final_output, input_min, input_max)
        
        return {
            'level_outputs': level_outputs,
            'final_output': final_output,
            'fusion_weights': weights
        }
        
    def normalize_output(self, x, target_min=None, target_max=None):
        if target_min is None or target_max is None:
            return x
        x_min, x_max = x.min(), x.max()
        x_normalized = (x - x_min) / (x_max - x_min)
        return x_normalized * (target_max - target_min) + target_min

    
def train_step(model, optimizer, batch):
    # Ensure input requires gradients for training
    batch.requires_grad_(True)

    #print("!")

    # Forward pass
    outputs = model(batch)
    #print("!")
    # Compute loss
    loss = fusion_loss_v3_fixed(outputs, batch,
                           lambda_final=0.4,    # Base MSE loss
                           lambda_levels=0.2,   # Multi-level reconstruction
                           lambda_edge=0.1,     # Edge preservation
                           lambda_ssim=0.2,     # Structural similarity
                           lambda_contrast=0.1  # Contrast matching
                           )
    
    '''loss = fusion_loss_v4(outputs, batch, 
                             lambda_cnr=0.3,
                             lambda_msssim=0.2, 
                             lambda_edge=0.2,
                             lambda_contrast=0.2)'''
    #print("!")
    # Backpropagation for training
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), outputs

def train_model(model, train_loader, val_loader, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    tolerance = 1e-4
    max_patience = 5
    patience = 0
    prev_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_outputs = None
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch = batch[0].to(device)
            loss, outputs = train_step(model, optimizer, batch)
            epoch_loss += loss
            if batch_idx == len(train_loader) - 1:  # Save last batch outputs
                epoch_outputs = outputs
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1} Validation"):
                batch = batch[0].to(device)
                val_loss += loss

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")

        if epoch % 2 == 0 and epoch_outputs is not None:
            visualize_results(model, batch)
            

        if epoch_loss > prev_loss - tolerance:
            patience += 1
        else:
            patience = 0
        if patience > max_patience:
            break
        prev_loss = epoch_loss

def visualize_results(model, test_batch):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Move batch to device and get model output
        test_batch = test_batch.to(device)
        outputs = model(test_batch)
        
        # Extract images [batch, levels, channels, height, width]
        original_image = test_batch[0, 0, 0].cpu()  # First level, first channel
        final_output = outputs['final_output'][0, 0].cpu()  # First image, first channel
        final_target = test_batch[0, -1, 0].cpu()  # Last level, first channel
        
        # Add dimensions for SSIM calculation
        final_output_ssim = final_output.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        final_target_ssim = final_target.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        #analysis_results = analyze_dark_regions(final_output_ssim)
        #visualize_analysis(analysis_results)
        
        # Calculate metrics
        ssim_val = ssim(
            final_output_ssim, 
            final_target_ssim,
            data_range=final_target.max() - final_target.min()
        )
        
        msssim_val = ms_ssim(
            final_output_ssim,
            final_target_ssim,
            data_range=1.0,
            size_average=True
        ).item()

        # Convert to numpy for CNR calculation
        final_output_np = final_output.numpy()
        
        # Define regions for CNR (adjust coordinates based on your image)
        h, w = final_output_np.shape
        signal_region = final_output_np[h//4:h//2, w//4:w//2]
        background_region = final_output_np[0:h//4, 0:w//4]
        cnr_val = compute_cnr(signal_region, background_region)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        fig.suptitle('Progressive Fusion Results', fontsize=16)
            
        # Plot results
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(final_output, cmap='gray')
        axes[1].set_title(f'Denoised Output\nSSIM: {ssim_val:.3f}\nMS-SSIM: {msssim_val:.3f}\nCNR: {cnr_val:.3f}')
        axes[1].axis('off')
        
        axes[2].imshow(final_target, cmap='gray')
        axes[2].set_title('Target (Final Fusion Level)')
        axes[2].axis('off')
        
        
        plt.tight_layout()
        plt.show()
        
        return {
            'ssim': ssim_val.item(),
            'ms_ssim': msssim_val,
            'cnr': cnr_val
        }

