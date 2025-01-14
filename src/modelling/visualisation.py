import matplotlib.pyplot as plt

def visualize_batch(epoch, batch_idx, input_images, output_images, target_images, vis_dir):
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
        plt.savefig(vis_dir / f'epoch_{epoch}_batch_{batch_idx}.png')
        plt.close()

def visualize_fusion_levels(epoch, batch_idx, data, vis_dir):
    """Visualize all fusion levels for a single image"""
    n_levels = data.shape[1]
    fig, axes = plt.subplots(1, n_levels, figsize=(20, 4))
    
    for i in range(n_levels):
        img = data[0, i].cpu().squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Level {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / f'fusion_levels_epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

def plot_losses(history, vis_dir):
    """Plot training history"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(vis_dir / 'loss_history.png')
    plt.close()
