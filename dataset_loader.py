"""
Dataset loaders for TGVAnn
Handles RGB and Texture image pairs
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFile
from pathlib import Path

# Fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def remove_corrupted_images(directory):
    """
    Remove corrupted images from dataset
    
    Args:
        directory: Path to image directory
    """
    corrupted_count = 0
    for img_path in Path(directory).rglob("*.*"):
        if not img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            continue
        try:
            img = Image.open(img_path)
            img.verify()
        except (Image.UnidentifiedImageError, OSError):
            print(f"Removing corrupted image: {img_path}")
            img_path.unlink()
            corrupted_count += 1
    
    if corrupted_count > 0:
        print(f"Removed {corrupted_count} corrupted images")


class RGBTextureDataset(Dataset):
    """
    Dataset for RGB-Texture image pairs
    
    Args:
        rgb_dir: Path to RGB images
        texture_dir: Path to texture images
        transform_rgb: Transforms for RGB images
        transform_texture: Transforms for texture images
    """
    
    def __init__(self, rgb_dir, texture_dir, transform_rgb=None, transform_texture=None):
        self.rgb_dataset = datasets.ImageFolder(rgb_dir, transform=transform_rgb)
        self.texture_dataset = datasets.ImageFolder(texture_dir, transform=transform_texture)
        
        # Verify both datasets have same structure
        assert len(self.rgb_dataset.classes) == len(self.texture_dataset.classes), \
            "RGB and Texture datasets must have same number of classes"
        assert len(self.rgb_dataset) == len(self.texture_dataset), \
            "RGB and Texture datasets must have same number of images"
        
        self.classes = self.rgb_dataset.classes
        self.class_to_idx = self.rgb_dataset.class_to_idx
        
    def __len__(self):
        return len(self.rgb_dataset)
    
    def __getitem__(self, idx):
        rgb_img, label = self.rgb_dataset[idx]
        texture_img, _ = self.texture_dataset[idx]
        
        return rgb_img, texture_img, label


def get_transforms(input_size=256, augment=True):
    """
    Get data transforms for RGB and Texture images
    
    Args:
        input_size: Size to resize images (default: 256)
        augment: Whether to apply data augmentation (default: True)
    
    Returns:
        transform_rgb: RGB transforms
        transform_texture: Texture transforms
    """
    if augment:
        transform_rgb = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_texture = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        transform_rgb = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_texture = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    return transform_rgb, transform_texture


def prepare_dataloaders(rgb_dir, texture_dir, batch_size=32, input_size=256,
                        num_workers=4, augment=True, val_split=0.2):
    """
    Prepare train and validation dataloaders
    
    Args:
        rgb_dir: Path to RGB dataset
        texture_dir: Path to texture dataset
        batch_size: Batch size
        input_size: Input image size
        num_workers: Number of workers for data loading
        augment: Apply data augmentation
        val_split: Validation split ratio
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_classes: Number of classes
        class_names: List of class names
    """
    # Remove corrupted images
    print("Checking for corrupted images...")
    remove_corrupted_images(rgb_dir)
    remove_corrupted_images(texture_dir)
    
    # Get transforms
    transform_rgb_train, transform_texture_train = get_transforms(input_size, augment=augment)
    transform_rgb_val, transform_texture_val = get_transforms(input_size, augment=False)
    
    # Create full dataset
    full_dataset = RGBTextureDataset(rgb_dir, texture_dir,
                                     transform_rgb_train, transform_texture_train)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset with different transforms
    val_dataset_clean = RGBTextureDataset(rgb_dir, texture_dir,
                                          transform_rgb_val, transform_texture_val)
    
    # Use only validation indices
    val_indices = val_dataset.indices
    val_dataset_final = torch.utils.data.Subset(val_dataset_clean, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset_final, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {dataset_size}")
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, num_classes, class_names


def prepare_test_dataloader(rgb_dir, texture_dir, batch_size=32, input_size=256, num_workers=4):
    """
    Prepare test dataloader (no augmentation)
    
    Args:
        rgb_dir: Path to RGB test dataset
        texture_dir: Path to texture test dataset
        batch_size: Batch size
        input_size: Input image size
        num_workers: Number of workers
    
    Returns:
        test_loader: Test dataloader
        num_classes: Number of classes
        class_names: List of class names
    """
    # Remove corrupted images
    print("Checking for corrupted images...")
    remove_corrupted_images(rgb_dir)
    remove_corrupted_images(texture_dir)
    
    # Get transforms (no augmentation)
    transform_rgb, transform_texture = get_transforms(input_size, augment=False)
    
    # Create dataset
    test_dataset = RGBTextureDataset(rgb_dir, texture_dir, transform_rgb, transform_texture)
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    
    print(f"\nTest Dataset Statistics:")
    print(f"Total samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    return test_loader, num_classes, class_names
