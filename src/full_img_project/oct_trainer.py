import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

class OCTDenoiseTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create visualization directory
        self.vis_dir = Path('visualizations')
        self.vis_dir.mkdir(exist_ok=True)
        
        # Optimization
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Loss functions
        self.mse_loss = MSELoss()
        self.l1_loss = L1Loss()
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}

    def compute_loss(self, output, target, fusion_weights=None):
        """Compute total loss"""
        # MSE loss
        mse = self.mse_loss(output, target)
        
        # L1 loss
        l1 = self.l1_loss(output, target)
        
        # SSIM loss
        ssim_loss = 1 - self.ssim(output, target)
        
        # Total loss
        total_loss = mse + 0.5 * l1 + 0.5 * ssim_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse,
            'l1_loss': l1,
            'ssim_loss': ssim_loss
        }
    
    def compute_cnr_loss(self, output):
        # Detect edges/features using Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=output.device).float().view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=output.device).float().view(1, 1, 3, 3)
        
        edges = torch.sqrt(
            F.conv2d(output, sobel_x, padding=1)**2 + 
            F.conv2d(output, sobel_y, padding=1)**2 + 1e-6
        )
        
        # Use edge information to identify feature regions
        feature_mask = (edges > edges.mean()).float()
        
        # Calculate CNR only in feature regions
        feature_region = output * feature_mask
        non_feature_region = output * (1 - feature_mask)
        
        mean_feature = (feature_region.sum() / (feature_mask.sum() + 1e-6))
        mean_background = (non_feature_region.sum() / ((1 - feature_mask).sum() + 1e-6))
        std_background = torch.sqrt(((non_feature_region - mean_background) ** 2).sum() / ((1 - feature_mask).sum() + 1e-6) + 1e-6)
        
        return -torch.abs(mean_feature - mean_background) / (std_background + 1e-6)
    
    def compute_loss(self, output, target, fusion_weights=None):
        """Compute total loss with focus on CNR, SSIM, and MSE"""
        
        # MSE loss
        mse = self.mse_loss(output, target)
        
        # SSIM loss
        ssim_loss = 1 - self.ssim(output, target)
        
        # CNR loss
        h, w = output.shape[2:]
        signal_region = output[:, :, h//4:h//2, w//4:w//2]
        background_region = output[:, :, 0:h//4, 0:w//4]
        
        mean_signal = signal_region.mean()
        mean_background = background_region.mean()
        std_background = background_region.std()
        
        cnr_loss = self.compute_cnr_loss(output)#-torch.abs(mean_signal - mean_background) / (std_background + 1e-6)
        
        # Total loss with weights
        #total_loss = 0.6 * mse + 0.3 * ssim_loss + 0.1 * cnr_loss
        
        total_loss = 0.45 * mse + 0.5 * ssim_loss + 0.05 * cnr_loss

        return {
            'total_loss': total_loss,
            'mse_loss': mse,
            'ssim_loss': ssim_loss,
            'cnr_loss': cnr_loss
        }
    
    
    
    def ssim(self, x, y):
        """Calculate SSIM"""
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

    def visualize_batch(self, epoch, batch_idx, input_images, output_images, target_images):
        """Visualize a batch of images"""
        # Create figure with subplots for each image type
        fig, axes = plt.subplots(3, min(4, input_images.shape[0]), 
                                figsize=(15, 10))
        
        # Ensure axes is always 3D even with single image
        if input_images.shape[0] == 1:
            axes = axes.reshape(-1, 1)
            
        for i in range(min(4, input_images.shape[0])):
            # Get images
            input_img = input_images[i].cpu().squeeze().numpy()
            output_img = output_images[i].detach().cpu().squeeze().numpy()
            target_img = target_images[i].cpu().squeeze().numpy()
            
            # Plot images
            axes[0, i].imshow(input_img, cmap='gray')
            axes[0, i].set_title(f'Input {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(output_img, cmap='gray')
            axes[1, i].set_title(f'Output {i+1}')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(target_img, cmap='gray')
            axes[2, i].set_title(f'Target {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / f'epoch_{epoch}_batch_{batch_idx}.png')
        plt.close()

    def visualize_fusion_levels(self, epoch, batch_idx, data):
        """Visualize all fusion levels for a single image"""
        n_levels = data.shape[1]
        fig, axes = plt.subplots(1, n_levels, figsize=(20, 4))
        
        for i in range(n_levels):
            img = data[0, i].cpu().squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Level {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / f'fusion_levels_epoch_{epoch}_batch_{batch_idx}.png')
        plt.close()

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                   desc=f'Epoch {epoch}')
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.device)
            
            # Get input and target images
            input_images = data[:, 0, :, :, :]
            target_images = data[:, -2, :, :, :]
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)

            output = normalize_to_target(output, target_images)
            
            # Compute loss
            losses = self.compute_loss(output, target_images)
            total_loss = losses['total_loss']
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            epoch_losses.append(total_loss.item())
            pbar.set_postfix({'loss': f"{sum(epoch_losses)/len(epoch_losses):.4f}"})
            
            # Visualize periodically
            if batch_idx % 50 == 0:
                print("Saving visualizations")
                self.visualize_batch(epoch, batch_idx, input_images, output, target_images)
                self.visualize_fusion_levels(epoch, batch_idx, data)
        
        return sum(epoch_losses) / len(epoch_losses)

    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for data, _ in tqdm(self.val_loader, desc='Validating'):
                data = data.to(self.device)
                
                input_images = data[:, 0, :, :, :]
                target_images = data[:, -2, :, :, :]
                
                output = self.model(data)

                output = normalize_to_target(output, target_images)
                
                losses = self.compute_loss(output, target_images)
                
                val_losses.append(losses['total_loss'].item())
        
        return sum(val_losses) / len(val_losses)

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

    def plot_losses(self):
        """Plot training history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.vis_dir / 'loss_history.png')
        plt.close()

    def train(self, num_epochs):
        """Main training loop"""
        best_val_loss = float('inf')
        
        print(f"Training on device: {self.device}")
        print(f"Training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            
            # Print progress and plot
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            self.plot_losses()

def normalize_to_target(input_img, target_img):
    """
    Normalize input image to match target image statistics
    Args:
        input_img: Input tensor (B, C, H, W)
        target_img: Target tensor (B, C, H, W)
    Returns:
        Normalized input tensor
    """
    # Get target statistics
    target_mean = target_img.mean()
    target_std = target_img.std()
    
    # Get input statistics
    input_mean = input_img.mean()
    input_std = input_img.std()
    
    # Normalize
    normalized = ((input_img - input_mean) / input_std) * target_std + target_mean
    
    return normalized