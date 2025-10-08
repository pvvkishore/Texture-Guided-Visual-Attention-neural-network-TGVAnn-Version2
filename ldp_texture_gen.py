"""
LDP (Local Directional Pattern) Texture Generator
Generates texture features from RGB images using Kirsch gradients
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


def compute_kirsch_gradients(image):
    """
    Compute Kirsch gradients for 8 directions
    
    Args:
        image: Grayscale image array
    
    Returns:
        gradients: List of 8 gradient images
    """
    kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),    # North
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),    # Northeast
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),    # East
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),    # Southeast
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),    # South
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),    # Southwest
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),    # West
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])     # Northwest
    ]
    
    gradients = [cv2.filter2D(image, cv2.CV_32F, kernel) for kernel in kernels]
    return gradients


def compute_ldp(image, top_k=3):
    """
    Compute Local Directional Pattern (LDP)
    
    Args:
        image: Grayscale image array
        top_k: Number of top gradients to consider (default: 3)
    
    Returns:
        ldp_pattern: LDP texture image
    """
    gradients = compute_kirsch_gradients(image)
    gradients = np.stack(gradients, axis=-1)
    
    # Get top-k indices
    top_k_indices = np.argsort(-gradients, axis=-1)[..., :top_k]
    
    # Create LDP pattern
    ldp_pattern = np.zeros_like(image, dtype=np.int32)
    for k in range(top_k):
        ldp_pattern += (1 << k) * (top_k_indices[..., k] + 1)
    
    return np.clip(ldp_pattern, 0, 255).astype(np.uint8)


def enhance_image(image):
    """
    Enhance brightness and contrast using CLAHE
    
    Args:
        image: Grayscale image
    
    Returns:
        enhanced_image: Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image


def generate_texture_single_channel(image):
    """
    Generate single-channel texture from RGB image
    
    Args:
        image: RGB image array
    
    Returns:
        texture: Single-channel texture image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute LDP texture
    ldp_texture = compute_ldp(gray, top_k=3)
    
    # Enhance texture
    enhanced_texture = enhance_image(ldp_texture)
    
    return enhanced_texture


def generate_texture_color(image):
    """
    Generate color texture from RGB image
    
    Args:
        image: RGB image array
    
    Returns:
        color_texture: 3-channel color texture image
    """
    # Split into color channels
    b_channel, g_channel, r_channel = cv2.split(image)
    
    # Compute LDP for each channel
    ldp_b = enhance_image(compute_ldp(b_channel))
    ldp_g = enhance_image(compute_ldp(g_channel))
    ldp_r = enhance_image(compute_ldp(r_channel))
    
    # Merge LDP channels into color texture
    color_texture = cv2.merge((ldp_b, ldp_g, ldp_r))
    
    return color_texture


def process_dataset(input_folder, output_folder, texture_type='single', resize_shape=None):
    """
    Process entire dataset to generate texture features
    
    Args:
        input_folder: Path to input RGB dataset
        output_folder: Path to save texture dataset
        texture_type: 'single' for grayscale or 'color' for RGB texture
        resize_shape: Tuple (H, W) to resize images, None to keep original
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all class folders
    class_folders = [f for f in os.listdir(input_folder) 
                     if os.path.isdir(os.path.join(input_folder, f))]
    
    print(f"Processing {len(class_folders)} classes...")
    
    for class_folder in tqdm(class_folders, desc="Processing Classes"):
        class_path = os.path.join(input_folder, class_folder)
        output_class_path = os.path.join(output_folder, class_folder)
        
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
        
        # Process all images in class folder
        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        for image_name in tqdm(image_files, desc=f"Class: {class_folder}", leave=False):
            image_path = os.path.join(class_path, image_name)
            
            try:
                # Read RGB image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read: {image_path}")
                    continue
                
                # Resize if specified
                if resize_shape is not None:
                    image = cv2.resize(image, (resize_shape[1], resize_shape[0]))
                
                # Generate texture
                if texture_type == 'single':
                    texture = generate_texture_single_channel(image)
                elif texture_type == 'color':
                    texture = generate_texture_color(image)
                else:
                    raise ValueError(f"Invalid texture_type: {texture_type}")
                
                # Save texture image
                output_image_path = os.path.join(output_class_path, image_name)
                cv2.imwrite(output_image_path, texture)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    print(f"\nTexture generation complete! Saved to: {output_folder}")


def verify_dataset_structure(dataset_path):
    """
    Verify dataset structure and print statistics
    
    Args:
        dataset_path: Path to dataset
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    print(f"\nDataset: {dataset_path}")
    print("=" * 60)
    
    class_folders = [f for f in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, f))]
    
    total_images = 0
    for class_folder in sorted(class_folders):
        class_path = os.path.join(dataset_path, class_folder)
        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        num_images = len(image_files)
        total_images += num_images
        print(f"{class_folder:30s}: {num_images:5d} images")
    
    print("=" * 60)
    print(f"Total Classes: {len(class_folders)}")
    print(f"Total Images: {total_images}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    input_folder = "Maize_Dataset"
    output_folder = "Maize_Texture"
    
    # Generate single-channel texture
    print("Generating single-channel LDP textures...")
    process_dataset(input_folder, output_folder, texture_type='single', resize_shape=(256, 256))
    
    # Verify generated dataset
    verify_dataset_structure(output_folder)