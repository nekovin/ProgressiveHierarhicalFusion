import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

from modelling.utils import save_checkpoint, normalize_to_target
from modelling.loss import *
from modelling.visualisation import *

class trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir=f'checkpoints',
        img_size=512,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.img_size = img_size

        self.vis_dir = Path('visualizations')
        self.vis_dir.mkdir(exist_ok=True)
        
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        #self.mse_loss = MSELoss()
        self.l1_loss = L1Loss()
        
        self.history = {'train_loss': [], 'val_loss': []}

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
                save_checkpoint(epoch, val_loss, self.model, self.optimizer, self.checkpoint_dir, self.img_size)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            plot_losses(self.history, self.vis_dir)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                desc=f'Epoch {epoch}')
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.device)
            num_levels = data.shape[1] - 1
            batch_loss = 0
            
            self.optimizer.zero_grad() 
            
            for level in range(num_levels):
                input_img = data[:, level, :, :, :]
                target_images = data[:, level+1, :, :, :]
                
                output = self.model(input_img)
                output = normalize_to_target(output, target_images)
                
                level_losses = compute_dynamic_loss(output, target_images, level, num_levels)

                level_weight = (level + 1) / num_levels
                batch_loss += level_losses['total_loss'] * level_weight

            input_img = data[:, 0, :, :, :]
            target_images = data[:, -1, :, :, :]
            output = self.model(input_img)
            output = normalize_to_target(output, target_images)
            final_losses = compute_loss(output, target_images)

            final_weight = 1.0
            batch_loss += final_losses['total_loss'] * final_weight

            
            batch_loss.backward()

            self.optimizer.step()
            
            epoch_losses.append(batch_loss.item())
            pbar.set_postfix({'loss': f"{sum(epoch_losses)/len(epoch_losses):.4f}"})
            
            if batch_idx % 50 == 0:
                visualize_batch(epoch, batch_idx, input_img, output, target_images, self.vis_dir)
        
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

                output = normalize_to_target(output, target_images)
                
                losses = compute_loss(output, target_images)
                
                val_losses.append(losses['total_loss'].item())
        
        return sum(val_losses) / len(val_losses)