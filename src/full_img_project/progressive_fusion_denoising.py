import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from ssn2v_like import ProgressiveFusionUNet 

class ImprovedFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.feature_gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = F.relu(self.bn1(self.conv1(x)))
        features = F.relu(self.bn2(self.conv2(features)))
        
        gate = self.feature_gate(x)
        return features * gate + features

class EfficientProgressiveFusion(nn.Module):
    def __init__(self, fusion_levels=8, base_channels=32):
        super().__init__()
        self.fusion_levels = fusion_levels
        
        self.init_conv = nn.Conv2d(1, base_channels, 3, padding=1)
        
        self.fusion_blocks = nn.ModuleList([
            ImprovedFusionBlock(base_channels * (i + 1), base_channels)
            for i in range(fusion_levels)
        ])
        
        self.final_conv = nn.Conv2d(base_channels, 1, 3, padding=1)
        
        self.fusion_weights = nn.Parameter(torch.ones(fusion_levels) / fusion_levels)
        
    def forward(self, x):
        batch_size = x.shape[0]
        features = []
        outputs = []
        
        for i in range(self.fusion_levels):
            curr_input = x[:, i, :, :, :]
            curr_feat = F.relu(self.init_conv(curr_input))
            features.append(curr_feat)
        
        curr_features = features[0]
        for i in range(self.fusion_levels):
            if i > 0:
                fusion_input = torch.cat([curr_features] + features[:i+1], dim=1)
            else:
                fusion_input = curr_features
            
            curr_features = self.fusion_blocks[i](fusion_input)
            
            level_output = self.final_conv(curr_features)
            outputs.append(level_output)
        
        weights = F.softmax(self.fusion_weights, dim=0)
        final_output = sum(w * out for w, out in zip(weights, outputs))
        
        return {
            'level_outputs': outputs,
            'final_output': final_output,
            'fusion_weights': weights
        }

def improved_fusion_loss(outputs, targets, lambda_mse=0.4, lambda_feature=0.3, lambda_ssim=0.3):
    """Simplified but effective loss function"""
    
    def feature_preservation_loss(pred, target):
        # Edge detection for feature preservation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                             device=pred.device).float().view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                             device=pred.device).float().view(1, 1, 3, 3)
        
        pred_edges = torch.sqrt(
            F.conv2d(pred, sobel_x, padding=1)**2 + 
            F.conv2d(pred, sobel_y, padding=1)**2 + 1e-6
        )
        target_edges = torch.sqrt(
            F.conv2d(target, sobel_x, padding=1)**2 + 
            F.conv2d(target, sobel_y, padding=1)**2 + 1e-6
        )
        return F.mse_loss(pred_edges, target_edges)
    
    final_output = outputs['final_output']
    final_target = targets[:, -1, :, :, :]
    
    mse_loss = F.mse_loss(final_output, final_target)
    
    feature_loss = feature_preservation_loss(final_output, final_target)
    
    ssim_loss = 1 - ssim(final_output, final_target, data_range=1.0, size_average=True)

    total_loss = (
        lambda_mse * mse_loss +
        lambda_feature * feature_loss +
        lambda_ssim * ssim_loss
    )
    
    return total_loss

def create_progressive_fusion_model(n_fusion_levels=8):
    return ProgressiveFusionUNet(n_channels=1, n_fusion_levels=n_fusion_levels)