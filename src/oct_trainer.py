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
        checkpoint_dir='../checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.vis_dir = Path('../visualizations')
        self.vis_dir.mkdir(exist_ok=True)
        
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.mse_loss = MSELoss()
        self.l1_loss = L1Loss()
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def compute_cnr_loss(self, output):
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
        
        return -torch.abs(mean_feature - mean_background) / (std_background + 1e-6)

    def compute_loss(self, output, target, fusion_weights=None): # evil loss function
        """Compute total loss with focus on CNR, SSIM, and MSE"""
        
        mse = self.mse_loss(output, target) # mse
        
        ssim_loss = 1 - self.ssim(output, target) # sssim
        
        '''
        h, w = output.shape[2:]
        signal_region = output[:, :, h//4:h//2, w//4:w//2]
        background_region = output[:, :, 0:h//4, 0:w//4]
        
        mean_signal = signal_region.mean()
        mean_background = background_region.mean()
        std_background = background_region.std()'''
        
        cnr_loss = self.compute_cnr_loss(output)#-torch.abs(mean_signal - mean_background) / (std_background + 1e-6)
        
        #total_loss = 1 * mse + 0.5 * ssim_loss + 0.05 * cnr_loss
        
        total_loss = mse + ssim_loss + cnr_loss*0.01

        return {
            'total_loss': total_loss,
            'mse_loss': mse,
            'ssim_loss': ssim_loss,
            'cnr_loss': cnr_loss
        }
    
    
    
    def ssim(self, x, y):
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

        fig, axes = plt.subplots(3, min(4, input_images.shape[0]), 
                                figsize=(15, 10))
        
        if input_images.shape[0] == 1:
            axes = axes.reshape(-1, 1)
            
        for i in range(min(4, input_images.shape[0])):
            input_img = input_images[i].cpu().squeeze().numpy()
            output_img = output_images[i].detach().cpu().squeeze().numpy()
            target_img = target_images[i].cpu().squeeze().numpy()
            
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
            
            input_images = data[:, 0, :, :, :]
            target_images = data[:, -2, :, :, :]
            
            self.optimizer.zero_grad()
            output = self.model(data)

            output = normalize_to_target(output, target_images)

            losses = self.compute_loss(output, target_images)
            total_loss = losses['total_loss']
            
            total_loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(total_loss.item())
            pbar.set_postfix({'loss': f"{sum(epoch_losses)/len(epoch_losses):.4f}"})
            
            if batch_idx % 50 == 0:
                print("saving visualizations")
                self.visualize_batch(epoch, batch_idx, input_images, output, target_images)
                self.visualize_fusion_levels(epoch, batch_idx, data)
        
        return sum(epoch_losses) / len(epoch_losses)

    def validate(self):
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
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            self.plot_losses()

    def predict(self, model, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            checkpoint = torch.load('checkpoints/checkpoint_epoch_5.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            print("No checkpoint found")
            return
        
        model.to(device)
        model.eval()
        with torch.no_grad():
            batch = next(iter(data))[0].to(device)
            output = model(batch)

            input_img = batch[0,0]
            fused_img = batch[0,-1]
            output_img = output[0]

            ssim_val = 1 - self.ssim(output_img.unsqueeze(0), fused_img.unsqueeze(0))
            
            mse_val = F.mse_loss(output_img, fused_img)
            
            cnr_input = -self.compute_cnr_loss(input_img.unsqueeze(0))  # Negative because loss is negative CNR
            cnr_fused = -self.compute_cnr_loss(fused_img.unsqueeze(0))
            cnr_output = -self.compute_cnr_loss(output_img.unsqueeze(0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(input_img.cpu().squeeze().numpy(), cmap='gray')
        axes[0].set_title(f'Input\nCNR: {cnr_input:.3f}')
        axes[0].axis('off')

        axes[1].imshow(fused_img.cpu().squeeze().numpy(), cmap='gray')
        axes[1].set_title(f'Fused\nCNR: {cnr_fused:.3f}')
        axes[1].axis('off')

        axes[2].imshow(output_img.cpu().squeeze().numpy(), cmap='gray')
        axes[2].set_title(f'Output\nCNR: {cnr_output:.3f}\nMSE to Fused: {mse_val:.3f}\nSSIM to Fused: {ssim_val:.3f}')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
            
        return output

def normalize_to_target(input_img, target_img):

    target_mean = target_img.mean()
    target_std = target_img.std()
    input_mean = input_img.mean()
    input_std = input_img.std()
    normalized = ((input_img - input_mean) / input_std) * target_std + target_mean
    
    return normalized

