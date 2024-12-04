import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

from denoising import *

class NoiseDistributionModule(nn.Module):
    """Learn and model the noise distribution in OCT images"""
    def __init__(self, channels):
        super().__init__()
        self.noise_encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1)
        )
        
    def forward(self, x, clean):
        # Estimate noise as difference between input and clean
        noise = x - clean
        # Learn noise characteristics
        noise_features = self.noise_encoder(noise)
        return noise_features

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module for feature refinement"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate query, key, value
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H, W)
        q, k, v = qkv.unbind(1)
        
        # Reshape for attention
        q = q.flatten(3)
        k = k.flatten(3)
        v = v.flatten(3)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).reshape(B, C, H, W)
        x = self.proj(x)
        return x

class EnhancedDoubleConv(nn.Module):
    """Enhanced double convolution with attention and residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = MultiHeadAttention(out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual_conv(x)
        out = self.double_conv(x)
        out = self.attention(out)
        return out + identity

class CustomLoss(nn.Module):
    """Custom loss function combining MSE, SSIM, and CNR objectives"""
    def __init__(self, alpha=0.5, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def compute_cnr(self, img, roi_size=7):
        # Simulate ROI by taking center patch
        B, C, H, W = img.shape
        center_h = H // 2
        center_w = W // 2
        roi = img[:, :, 
                 center_h-roi_size//2:center_h+roi_size//2,
                 center_w-roi_size//2:center_w+roi_size//2]
        background = img[:, :, :roi_size, :roi_size]  # Use top-left corner as background
        
        roi_mean = roi.mean()
        roi_std = roi.std()
        bg_mean = background.mean()
        bg_std = background.std()
        
        cnr = torch.abs(roi_mean - bg_mean) / torch.sqrt(roi_std**2 + bg_std**2)
        return cnr

    def forward(self, pred, target):
        # MSE Loss
        mse_loss = F.mse_loss(pred, target)
        
        # SSIM Loss (1 - SSIM to minimize)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        
        # CNR Loss (negative since we want to maximize CNR)
        cnr_loss = -self.compute_cnr(pred)
        
        # Combined loss
        total_loss = mse_loss + self.alpha * ssim_loss + self.beta * cnr_loss
        return total_loss

class EnhancedProgressiveFusionModel(nn.Module):
    """Enhanced progressive fusion model with noise modeling and advanced attention"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Noise distribution learning
        self.noise_module = NoiseDistributionModule(in_channels)
        
        # Downsampling path
        in_features = in_channels
        for feature in features:
            self.downs.append(EnhancedDoubleConv(in_features, feature))
            in_features = feature
            
        # Upsampling path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(EnhancedDoubleConv(feature * 2, feature))
            
        self.bottleneck = EnhancedDoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, return_noise=False):
        # Store skip connections
        skips = []
        
        # Initial noise estimation (using input as proxy for clean)
        noise_features = self.noise_module(x, x.detach())
        
        # Add noise features to input
        x = x + noise_features
        
        # Downsampling
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skips = skips[::-1]
        
        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
                
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)
            
        x = self.final_conv(x)
        
        if return_noise:
            return x, noise_features
        return x

class EnhancedTrainer(Trainer):
# In your EnhancedTrainer class:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = CustomLoss()
        self.visualizer = SimpleVisualizer()

    def train_epoch(self, train_loader, num_levels):
        self.model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch[0].to(self.device)
            
            for level in range(num_levels - 2):
                current_level_images = images[level::num_levels]
                target_level_images = images[level + 1::num_levels]
                
                pred, noise = self.model(current_level_images, return_noise=True)
                loss = self.criterion(pred, target_level_images)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update visualization every 10 batches
                if batch_idx % 50 == 0:
                    self.visualizer.update_plots(
                        loss.item(),
                        current_level_images,
                        pred
                    )
    
class SimpleVisualizer:
    def __init__(self):
        self.losses = []

    def update_plots(self, loss, input_image, output_image):
        self.losses.append(loss)
        
        plt.clf()  # Clear current figure
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses)
        plt.title('Training Loss')
        
        # Plot Images
        plt.subplot(1, 2, 2)
        comparison = np.hstack([
            input_image[0, 0].detach().cpu().numpy(),
            output_image[0, 0].detach().cpu().numpy()
        ])
        plt.imshow(comparison, cmap='gray')
        plt.title('Input | Output')
        plt.axis('off')
        
        plt.pause(0.1)  # Short pause to update display

def train(num_epochs=10, num_levels=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedProgressiveFusionModel().to(device)  # Move entire model to GPU/CPU
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = EnhancedTrainer(model, optimizer, device)
    
    dataset, train_loader, val_loader = get_loaders()
    
    for epoch in range(num_epochs):
        trainer.train_epoch(train_loader, num_levels)
    
# Initialize and train as before
device = 'cuda'
model = EnhancedProgressiveFusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = EnhancedTrainer(model, optimizer, device='cuda')
#trainer.train_epoch(train_loader, num_levels=8)
train()