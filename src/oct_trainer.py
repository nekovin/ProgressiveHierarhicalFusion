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
        checkpoint_dir=f'../checkpoints',
        img_size=512,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.img_size = img_size

        self.vis_dir = Path('../visualizations')
        self.vis_dir.mkdir(exist_ok=True)
        
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.mse_loss = MSELoss()
        self.l1_loss = L1Loss()
        
        self.history = {'train_loss': [], 'val_loss': []}

    def compute_low_signal_mask(output, threshold_factor=0.5):
        """Compute mask for low-signal areas based on intensity."""
        mean_intensity = output.mean()
        threshold = threshold_factor * mean_intensity  # Define threshold as a fraction of the mean
        low_signal_mask = (output < threshold).float()
        return low_signal_mask
    
    def low_signal_constraint_loss(output, low_signal_mask):
        """Apply loss penalty to suppress low-signal areas."""
        low_signal_region = output * low_signal_mask  # Focus on low-signal areas
        penalty = (low_signal_region ** 2).mean()  # Penalise non-dark pixels in low-signal areas
        return penalty

    
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

        cnr = -torch.abs(mean_feature - mean_background) / (std_background + 1e-6)
        
        return cnr #+ 0.5 * low_contrast_penalty

    def compute_loss(self, output, target, fusion_weights=None): # evil loss function
        """Compute total loss with focus on CNR, SSIM, and MSE"""

        '''Plan is to make this dynamic based on the fusion level'''
        
        mse = self.mse_loss(output, target) # mse
        
        ssim_loss = 1 - self.ssim(output, target) # sssim
        
        cnr_loss = self.compute_cnr_loss(output)#-torch.abs(mean_signal - mean_background) / (std_background + 1e-6)
        
        #total_loss = 1 * mse + 0.5 * ssim_loss + 0.05 * cnr_loss
        
        total_loss = mse + ssim_loss + cnr_loss*0.01

        return {
            'total_loss': total_loss,
            'mse_loss': mse,
            'ssim_loss': ssim_loss,
            'cnr_loss': cnr_loss
        }
    
    def compute_dynamic_loss(self, output, target, level=None, num_levels=1): # evil loss function
        """Compute total loss with focus on CNR, SSIM, and MSE"""

        '''Plan is to make this dynamic based on the fusion level'''
        
        mse = self.mse_loss(output, target) # mse
        
        ssim_loss = 1 - self.ssim(output, target) # sssim
        
        cnr_loss = self.compute_cnr_loss(output)#-torch.abs(mean_signal - mean_background) / (std_background + 1e-6)
        
        #total_loss = 1 * mse + 0.5 * ssim_loss + 0.05 * cnr_loss

    
        '''if level:
            #print("Level provided")
            # how to implement levels for dymaic loading?
            level_factor = level / (num_levels - 1)
            #level_factor = min(level / num_levels, 1.0)

            mse_weight = level_factor * 0.7  # Decreases with level (1.0 to 0.5)
            ssim_weight = level_factor * 0.3  # Increases with level (0.5 to 1.0)
            cnr_weight = 0.1 * (1 + level_factor)

            total_loss = (mse_weight * mse + 
                     ssim_weight * ssim_loss + 
                     cnr_weight * cnr_loss)

        else:
            #print("No level provided")'''
    
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

## Training ##
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

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                desc=f'Epoch {epoch}')
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.device)
            num_levels = data.shape[1] - 1
            batch_loss = 0
            
            self.optimizer.zero_grad()  # Zero gradients once per batch
            
            for level in range(num_levels):
                input_img = data[:, level, :, :, :]
                target_images = data[:, level+1, :, :, :]
                
                output = self.model(input_img)
                output = self.normalize_to_target(output, target_images)
                
                level_losses = self.compute_dynamic_loss(output, target_images, level, num_levels)

                level_weight = (level + 1) / num_levels
                batch_loss += level_losses['total_loss'] * level_weight

            input_img = data[:, 0, :, :, :]
            target_images = data[:, -1, :, :, :]
            output = self.model(input_img)
            output = self.normalize_to_target(output, target_images)
            final_losses = self.compute_loss(output, target_images)

            final_weight = 1.0
            batch_loss += final_losses['total_loss'] * final_weight

            
            batch_loss.backward()

            self.optimizer.step()
            
            epoch_losses.append(batch_loss.item())
            pbar.set_postfix({'loss': f"{sum(epoch_losses)/len(epoch_losses):.4f}"})
            
            if batch_idx % 50 == 0:
                self.visualize_batch(epoch, batch_idx, input_img, output, target_images)
        
        return sum(epoch_losses) / len(epoch_losses)

    def validate(self):
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for data, _ in tqdm(self.val_loader, desc='Validating'):
                data = data.to(self.device)
                
                input_images = data[:, 0, :, :, :]
                target_images = data[:, -1, :, :, :]
                
                #output = self.model(data)
                output = self.model(input_images)

                output = self.normalize_to_target(output, target_images)
                
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
        torch.save(checkpoint, self.checkpoint_dir / f'{self.img_size}_checkpoint_epoch_{epoch}.pt')
    
    def predict(self, model, data):
        import glob

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint_files = glob.glob(f'checkpoints/{self.img_size}_checkpoint_epoch_*.pt')

        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError("No checkpoint files found")
        
        model.to(device)
        model.eval()
        with torch.no_grad():
            for data, _ in data:
                #print(data)
                data = data.to(device)
                input_img = data[:, 0, :, :, :]
                fused_img = data[:, -1, :, :, :]

                #batch = next(iter(data))[0].to(device)
                output_img = model(input_img)

                output_img = self.normalize_to_target(output_img, fused_img)

                break

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(input_img.cpu().squeeze().numpy(), cmap='gray')
        axes[0].axis('off')

        axes[1].imshow(fused_img.cpu().squeeze().numpy(), cmap='gray')
        axes[1].axis('off')

        axes[2].imshow(output_img.cpu().squeeze().numpy(), cmap='gray')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        return output_img
    
    '''
    def normalize_to_target(self, input_img, target_img):

        target_mean = target_img.mean()
        target_std = target_img.std()
        input_mean = input_img.mean()
        input_std = input_img.std()
        normalized = ((input_img - input_mean) / input_std) * target_std + target_mean
        
        return normalized
'''
    def normalize_to_target(self, input_img, target_img):
        target_mean = target_img.mean()
        target_std = target_img.std()
        input_mean = input_img.mean()
        input_std = input_img.std()
        
        # Prevent division by very small numbers
        eps = 1e-8
        input_std = torch.clamp(input_std, min=eps)
        target_std = torch.clamp(target_std, min=eps)
        
        # Ensure positive scaling
        scale = torch.abs(target_std / input_std)
        
        normalized = ((input_img - input_mean) * scale) + target_mean
        
        return normalized

    ## Vis

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
