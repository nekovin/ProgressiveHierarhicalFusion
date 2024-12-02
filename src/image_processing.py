import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import argparse

class OCTImageProcessor:
    def __init__(self, patient: str, icip_data_path: str):
        self.patient = patient
        self.patient_path = os.path.join(icip_data_path, '0', patient)
        self.output_base_path = Path('../FusedDataset')

    def extract_number(self, filename: str) -> int:
        """Extract number from filename pattern (number)."""
        match = re.search(r'\((\d+)\)', filename)
        return int(match.group(1)) if match else -1

    def get_initial_fusion(self) -> Image.Image:
        """Create initial fusion from all images in patient directory."""
        images = sorted(
            [os.path.join(self.patient_path, image) for image in os.listdir(self.patient_path)],
            key=lambda x: self.extract_number(os.path.basename(x))
        )
        
        loaded_images = [Image.open(image) for image in images]
        image_arrays = [np.array(image) for image in loaded_images]
        stacked_images = np.stack(image_arrays, axis=0)
        fused_image_array = np.mean(stacked_images, axis=0).astype(np.uint8)
        
        return Image.fromarray(fused_image_array)

    @staticmethod
    def create_oct_mask(image: np.ndarray, threshold_factor: float = 0.3) -> np.ndarray:
        """Create binary mask for OCT image focusing on tissue regions."""
        if torch.is_tensor(image):
            image = image.cpu().detach().numpy()
        
        if image.ndim == 3:
            image = image.squeeze()
        
        image = (image - image.min()) / (image.max() - image.min())
        
        mean_val = np.mean(image)
        std_val = np.std(image)
        threshold = mean_val + threshold_factor * std_val
        
        mask = (image > threshold).astype(np.float32)
        
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return cv2.medianBlur(mask.astype(np.uint8), 5)

    def get_reference_mask(self, fused_image: np.ndarray, threshold_factor: float = 0.3) -> np.ndarray:
        """Generate reference mask from fused image."""
        return self.create_oct_mask(fused_image, threshold_factor)

    @staticmethod
    def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Dice coefficient between two masks."""
        intersection = np.sum(mask1 * mask2)
        sum_masks = np.sum(mask1) + np.sum(mask2)
        return 1.0 if sum_masks == 0 else (2. * intersection) / sum_masks

    @staticmethod
    def compute_jaccard_index(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Jaccard index between two masks."""
        intersection = np.sum(mask1 * mask2)
        union = np.sum(mask1) + np.sum(mask2) - intersection
        return 1.0 if union == 0 else intersection / union

    def process_folder(self) -> Dict[str, np.ndarray]:
        """Process all images in the patient folder and create masks."""
        masks = {}
        image_files = [f for f in os.listdir(self.patient_path) if f.endswith(('.png', '.jpg', '.tiff'))]
        
        for filename in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(self.patient_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            masks[filename] = self.create_oct_mask(image)
            
        return masks

    def get_scores(self, reference_mask: np.ndarray) -> Tuple[List[str], List[float], List[float]]:
        """Calculate similarity scores between masks and reference mask."""
        masks = self.process_folder()
        
        dice_scores = []
        jaccard_scores = []
        image_paths = []
        
        for mask_name, mask in masks.items():
            mask_binary = (mask > 0).astype(np.uint8)
            
            if reference_mask.shape != mask.shape:
                reference_resized = cv2.resize(reference_mask, (mask.shape[1], mask.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
            else:
                reference_resized = reference_mask
                
            reference_binary = (reference_resized > 0).astype(np.uint8)
            
            dice = self.compute_dice_coefficient(mask_binary, reference_binary)
            jaccard = self.compute_jaccard_index(mask_binary, reference_binary)
            
            dice_scores.append(dice)
            jaccard_scores.append(jaccard)
            image_paths.append(mask_name)
            
            print(f"Mask: {mask_name}, Dice: {dice:.4f}, Jaccard: {jaccard:.4f}")
            
        return image_paths, dice_scores, jaccard_scores

    def align_images(self, reference_mask: np.ndarray, dice_threshold: float = 0.5) -> List[np.ndarray]:
        """Select good quality images based on similarity scores."""
        image_paths, dice_scores, _ = self.get_scores(reference_mask)
        
        good_images = [img for i, img in enumerate(image_paths) if dice_scores[i] >= dice_threshold]
        
        return [cv2.imread(os.path.join(self.patient_path, img)) for img in good_images]

    def recompute_fusion(self, good_images: List[np.ndarray]):
        """Recursively fuse images and save results."""
        def save_fused_images(fused_images: List[np.ndarray], level: int):
            output_dir = self.output_base_path / self.patient / f'FusedImages_Level_{level}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, image_array in enumerate(fused_images):
                image = Image.fromarray(image_array.astype(np.uint8)).convert('L')
                image.save(output_dir / f'Fused_Image_Level_{level}_{idx}.tif')
            
            print(f"Saved {len(fused_images)} fused images at Level {level} in {output_dir}")

        def fuse_images(image_list: List[np.ndarray], level: int = 0):
            if len(image_list) == 1:
                return
            
            fused_images = []
            for i in range(0, len(image_list), 2):
                if i + 1 < len(image_list):
                    fused = np.mean([image_list[i], image_list[i + 1]], axis=0)
                else:
                    fused = image_list[i]
                fused_images.append(fused)
                
            save_fused_images(fused_images, level)
            fuse_images(fused_images, level + 1)

        fuse_images(good_images)

    def visualize_fusion_strip(self, strip_number: int = 0):
        """
        Visualize a vertical strip of fused images across all fusion levels.
        
        Args:
            strip_number (int): The horizontal position of the strip to visualize
        """
        # Find all fusion level directories
        fusion_dirs = sorted([d for d in os.listdir(self.output_base_path / self.patient) 
                            if d.startswith('FusedImages_Level_')],
                            key=lambda x: int(x.split('_')[-1]))
        
        if not fusion_dirs:
            print("No fusion levels found.")
            return
        
        # Create a figure
        fig, axes = plt.subplots(1, len(fusion_dirs), figsize=(15, 5))
        if len(fusion_dirs) == 1:
            axes = [axes]
        
        # Process each fusion level
        for idx, fusion_dir in enumerate(fusion_dirs):
            level_path = self.output_base_path / self.patient / fusion_dir
            
            # Get the first image that contains the specified strip
            try:
                image_path = next(level_path.glob(f'Fused_Image_Level_*_{strip_number}.tif'))
                image = np.array(Image.open(image_path))
                
                # Plot the image
                axes[idx].imshow(image, cmap='gray')
                axes[idx].set_title(f'Level {idx}')
                axes[idx].axis('off')
            except StopIteration:
                print(f"No image found for strip {strip_number} in {fusion_dir}")
                axes[idx].text(0.5, 0.5, 'No Image', ha='center', va='center')
                axes[idx].axis('off')
        
        plt.suptitle(f'Fusion Levels for Strip {strip_number}')
        plt.tight_layout()
        plt.show()

def parse_arguments():
    """Parse command line arguments with defaults."""
    parser = argparse.ArgumentParser(description='Process OCT images for a given patient.')
    
    parser.add_argument(
        '--input_folder',
        type=str,
        default=r'C:\Datasets\OCTData\ICIP training data',
        help='Input folder containing OCT data'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        default=r'C:\Datasets\OCTData\FusedDataset',
        help='Output folder for fused images'
    )
    
    parser.add_argument(
        '--patient',
        type=str,
        default='RawDataQA (10)',
        help='Patient number to process'
    )
    
    return parser.parse_args()

def main():

    args = parse_arguments()
    
    print(f"Input Folder: {args.input_folder}")
    print(f"Output Folder: {args.output_folder}")
    print(f"Processing Patient: {args.patient}")

    if not os.path.exists(args.input_folder):
        print(f"Input folder {args.input_folder} does not exist.")
        sys.exit(1)

    # default to 0 for data for now
    if not os.path.exists(os.path.join(args.input_folder, '0', args.patient)):
        print(f"Error: Patient folder not found: {os.path.join(args.input_folder, '0', args.patient)}")
        sys.exit(1)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder {args.output_folder}")
    
    processor = OCTImageProcessor(
        patient=args.patient,
        icip_data_path=args.input_folder
    )
    
    initial_fusion = processor.get_initial_fusion()
    reference_mask = processor.get_reference_mask(np.array(initial_fusion))
    
    good_images = processor.align_images(reference_mask)
    processor.recompute_fusion(good_images)
    
    processor.visualize_fusion_strip(0)

if __name__ == "__main__":
    main()