def divide_into_patches(image, patch_size, stride):
    """
    Divide a full image into overlapping patches.
    
    Args:
        image (torch.Tensor): Input image of shape [1, height, width].
        patch_size (int): Size of each patch (patch_size x patch_size).
        stride (int): Stride for overlapping patches.
    
    Returns:
        patches (torch.Tensor): Tensor of shape [num_patches, 1, patch_size, patch_size].
        patch_locations (list): List of (top, left) tuples for each patch.
    """
    _, height, width = image.shape
    patches = []
    patch_locations = []

    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            patch = image[:, top:top+patch_size, left:left+patch_size]
            patches.append(patch)
            patch_locations.append((top, left))

    return torch.stack(patches), patch_locations

def recombine_patches(patches, patch_locations, image_size, patch_size, stride):
    """
    Recombine patches into a full image with proper normalisation, including edge handling.
    
    Args:
        patches (torch.Tensor): Tensor of shape [num_patches, 1, patch_size, patch_size].
        patch_locations (list): List of (top, left) tuples for each patch.
        image_size (tuple): Size of the full image (height, width).
        patch_size (int): Size of each patch (patch_size x patch_size).
        stride (int): Stride used for overlapping patches.
    
    Returns:
        full_image (torch.Tensor): Reconstructed full image of shape [1, height, width].
    """
    height, width = image_size
    full_image = torch.zeros((1, height, width), dtype=patches.dtype, device=patches.device)
    weight_map = torch.zeros((1, height, width), dtype=patches.dtype, device=patches.device)

    for i, (top, left) in enumerate(patch_locations):
        # Add patch values to the full image
        full_image[:, top:top+patch_size, left:left+patch_size] += patches[i]
        # Add contributions to the weight map
        weight_map[:, top:top+patch_size, left:left+patch_size] += 1

    # Ensure proper normalisation by avoiding division by zero
    weight_map[weight_map == 0] = 1

    # Normalize overlapping areas
    full_image /= weight_map
    return full_image

def visualize_patch_grid(image, patch_locations, patch_size):
    """
    Visualize the grid of patches on the original image.
    
    Args:
        image (torch.Tensor): Input full image of shape [1, height, width].
        patch_locations (list): List of (top, left) tuples for each patch.
        patch_size (int): Size of each patch (patch_size x patch_size).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = image[0].cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    
    for (top, left) in patch_locations:
        rect = Rectangle((left, top), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.title("Patch Grid")
    plt.axis("off")
    plt.show()

def normalize_output(x, target_min=None, target_max=None):
    """
    Normalize output to match the target range [target_min, target_max].
    
    Args:
        x (torch.Tensor): Input tensor to be normalized.
        target_min (float): Minimum value of the target range.
        target_max (float): Maximum value of the target range.
        
    Returns:
        torch.Tensor: Normalized tensor with values in the range [target_min, target_max].
    """
    if target_min is None or target_max is None:
        return x
    
    x_min = x.min()
    x_max = x.max()
    
    # Avoid division by zero
    if x_max - x_min == 0:
        return torch.full_like(x, target_min)
    
    x_normalized = (x - x_min) / (x_max - x_min)  # Scale to [0, 1]
    x_scaled = x_normalized * (target_max - target_min) + target_min  # Scale to [target_min, target_max]
    
    return x_scaled

def denoise_full_image(image, model, patch_size, stride):
    """
    Denoise a full image using the model with patch-based processing.
    
    Args:
        image (torch.Tensor): Input full image of shape [1, height, width].
        model (torch.nn.Module): Trained denoising model.
        patch_size (int): Size of patches for processing.
        stride (int): Stride for overlapping patches.
    
    Returns:
        denoised_image (torch.Tensor): Denoised full image of shape [1, height, width].
    """

    model = model.to(image.device)
    # Divide the image into patches
    patches, patch_locations = divide_into_patches(image, patch_size, stride)

    print(f"Processing {len(patches)} patches")
    print(f"Patch Shape: {patches.shape}")
    
    # Denoise each patch
    with torch.no_grad():
        denoised_patches = model(patches)
    
    # Recombine the patches into a full image
    denoised_image = recombine_patches(denoised_patches, patch_locations, 
                                       image.shape[1:], patch_size, stride)
    return denoised_image

import cv2
import torch
import matplotlib.pyplot as plt

def denoise(model):
    # Load the original image
    img_path = r'C:\Users\CL-11\OneDrive\Repos\phf\FusedDataset\RawDataQA (10)\FusedImages_Level_0\Fused_Image_Level_0_0.tif'
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to a PyTorch tensor and add a batch dimension
    original_image = torch.from_numpy(image).float().unsqueeze(0)  # Shape: [1, 512, 512]

    # Ensure the model is in evaluation mode
    model = model.eval()

    # Perform denoising using your denoise_full_image function
    patch_size = 64
    stride = 32  # Overlapping patches for smooth reconstruction
    denoised_image = denoise_full_image(original_image, model, patch_size, stride)

    # Normalize the denoised image to match the range of the original image
    def normalize_output(x, target_min, target_max):
        """Normalize output to match the range of the original image."""
        x_min, x_max = x.min(), x.max()
        if x_max - x_min == 0:
            return torch.full_like(x, target_min)  # Handle case where input has no variation
        x_normalized = (x - x_min) / (x_max - x_min)  # Scale to [0, 1]
        x_scaled = x_normalized * (target_max - target_min) + target_min  # Scale to [target_min, target_max]
        return x_scaled

    # Apply normalisation
    original_min, original_max = original_image.min(), original_image.max()
    denoised_normalized = normalize_output(denoised_image, target_min=original_min, target_max=original_max)
    denoised_image = denoised_image.squeeze(0).unsqueeze(-1)

    # Visualise the results
    plt.figure(figsize=(15, 5))  # Adjust the figure size

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image[0].cpu().numpy(), cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image.cpu().numpy(), cmap='gray')  # Ensure indexing is consistent
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Denoised and Normalized Image")
    plt.imshow(denoised_normalized[0].cpu().numpy(), cmap='gray')  # Ensure indexing is consistent
    plt.axis("off")

    plt.tight_layout()  # Ensure subplots do not overlap
    plt.show()

