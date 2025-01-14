import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss

mse_loss = MSELoss()

def low_signal_constraint_loss(output, low_signal_mask):
        """Apply loss penalty to suppress low-signal areas."""
        low_signal_region = output * low_signal_mask  # Focus on low-signal areas
        penalty = (low_signal_region ** 2).mean()  # Penalise non-dark pixels in low-signal areas
        return penalty

def compute_cnr_loss(output):
    '''Untraditional computation of CNR'''
    # https://howradiologyworks.com/x-ray-cnr/
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=output.device).float().view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=output.device).float().view(1, 1, 3, 3)
    
    edges = torch.sqrt(
        F.conv2d(output, sobel_x, padding=1)**2 + 
        F.conv2d(output, sobel_y, padding=1)**2 + 1e-6
    )
    
    feature_mask = (edges > edges.mean()).float()
    feature_region = output * feature_mask
    non_feature_region = output * (1 - feature_mask)
    
    mean_feature = (feature_region.sum() / (feature_mask.sum() + 1e-6))
    mean_background = (non_feature_region.sum() / ((1 - feature_mask).sum() + 1e-6))
    std_background = torch.sqrt(((non_feature_region - mean_background) ** 2).sum() / ((1 - feature_mask).sum() + 1e-6) + 1e-6)

    cnr = -torch.abs(mean_feature - mean_background) / (std_background + 1e-6)
    
    return cnr #+ 0.5 * low_contrast_penalty

def compute_loss(output, target, fusion_weights=None): # evil loss function
    """Compute total loss with focus on CNR, SSIM, and MSE"""

    '''Plan is to make this dynamic based on the fusion level'''
    
    # mse
    mse = mse_loss(output, target)
    
    # ssim
    ssim_loss = 1 - ssim(output, target) 
    
    cnr_loss = compute_cnr_loss(output) #-torch.abs(mean_signal - mean_background) / (std_background + 1e-6)
    
    #total_loss = 1 * mse + 0.5 * ssim_loss + 0.05 * cnr_loss
    
    total_loss = mse + ssim_loss + cnr_loss*0.01

    return {
        'total_loss': total_loss,
        'mse_loss': mse,
        'ssim_loss': ssim_loss,
        'cnr_loss': cnr_loss
    }

def compute_dynamic_loss(output, target, level=None, num_levels=1): # evil loss function
    """Compute total loss with focus on CNR, SSIM, and MSE"""

    '''Plan is to make this dynamic based on the fusion level'''

    assert output.shape == target.shape
    
    mse = mse_loss(output, target) # mse
    
    ssim_loss = 1 - ssim(output, target) # sssim
    
    cnr_loss = compute_cnr_loss(output)#-torch.abs(mean_signal - mean_background) / (std_background + 1e-6)
    
    #total_loss = 1 * mse + 0.5 * ssim_loss + 0.05 * cnr_loss

    total_loss = mse + ssim_loss + cnr_loss*0.01

    return {
        'total_loss': total_loss,
        'mse_loss': mse,
        'ssim_loss': ssim_loss,
        'cnr_loss': cnr_loss
    }

def ssim(x, y):
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, padding=1)
    mu_y = F.avg_pool2d(y, 3, 1, padding=1)
    
    sigma_x = F.avg_pool2d(x ** 2, 3, 1, padding=1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, padding=1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, padding=1) - mu_x * mu_y
    
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    
    return ssim_map.mean()
