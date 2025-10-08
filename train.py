"""
Training script for TGVAnn
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tgvann import TGVAnn
from models.utils import count_parameters, model_summary, save_model_checkpoint
from data.dataset import prepare_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (rgb_imgs, texture_imgs, labels) in enumerate(pbar):
        rgb_imgs = rgb_imgs.to(device)
        texture_imgs = texture_imgs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs, _ = model(rgb_imgs, texture_imgs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for rgb_imgs, texture_imgs, labels in tqdm(val_loader, desc='Validation'):
            rgb_imgs = rgb_imgs.to(device)
            texture_imgs = texture_imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, _ = model(rgb_imgs, texture_imgs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    print("\nPreparing datasets...")
    train_loader, val_loader, num_classes, class_names = prepare_dataloaders(
        rgb_dir=args.rgb_dir,
        texture_dir=args.texture_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        augment=args.augment,
        val_split=args.val_split
    )
    
    # Create model
    print("\nCreating TGVAnn model...")
    model = TGVAnn(num_classes=num_classes, input_channels_rgb=3, input_channels_texture=1)
    model = model.to(device)
    
    # Print model summary
    if args.print_model:
        model_summary(model, input_size=(1, 3, args.input_size, args.input_size),
                     texture_size=(1, 1, args.input_size, args.input_size), device=device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            save_model_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
            print(f"✓ New best model saved! Accuracy: {val_acc:.2f}%")
        
        # Save checkpoint every N epochs
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_model_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    save_model_checkpoint(model, optimizer, args.epochs, val_loss, val_acc, final_model_path)
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TGVAnn for crop disease detection')
    
    # Data parameters
    parser.add_argument('--rgb_dir', type=str, required=True, help='Path to RGB dataset')
    parser.add_argument('--texture_dir', type=str, required=True, help='Path to texture dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'],
                       help='Learning rate scheduler')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--print_model', action='store_true', help='Print model summary')
    
    args = parser.parse_args()
    
    # Set augmentation to True by default
    args.augment = True
    
    main(args)