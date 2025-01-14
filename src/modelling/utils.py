import torch

def save_checkpoint(epoch, val_loss, model, optimizer, checkpoint_dir, img_size):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, checkpoint_dir / f'{img_size}_checkpoint_epoch_{epoch}.pt')

def normalize_to_target(input_img, target_img):
    target_mean = target_img.mean()
    target_std = target_img.std()
    input_mean = input_img.mean()
    input_std = input_img.std()

    eps = 1e-8
    input_std = torch.clamp(input_std, min=eps)
    target_std = torch.clamp(target_std, min=eps)
    
    scale = torch.abs(target_std / input_std)
    
    normalized = ((input_img - input_mean) * scale) + target_mean
    
    return normalized

def compute_low_signal_mask(output, threshold_factor=0.5):
        """Compute mask for low-signal areas based on intensity."""
        mean_intensity = output.mean()
        threshold = threshold_factor * mean_intensity  # Define threshold as a fraction of the mean
        low_signal_mask = (output < threshold).float()
        return low_signal_mask