"""
Grad-CAM Visualization for TGVAnn
Generates attention heatmaps for model interpretability
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tgvann import TGVAnn
from data.dataset import prepare_test_dataloader


def generate_gradcam(model, rgb_input, texture_input, target_layer, class_idx=None):
    """
    Generate Grad-CAM for given inputs
    
    Args:
        model: TGVAnn model
        rgb_input: RGB images [B, 3, H, W]
        texture_input: Texture images [B, 1, H, W]
        target_layer: Layer to compute Grad-CAM on
        class_idx: Target class indices (if None, use predicted class)
    
    Returns:
        cams: Grad-CAM heatmaps [B, H, W]
        attention_weights: Attention weights
    """
    model.eval()
    gradients = []
    activations = []
    
    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def save_activations(module, input, output):
        activations.append(output)
    
    # Register hooks
    hook_activation = target_layer.register_forward_hook(save_activations)
    hook_gradient = target_layer.register_backward_hook(save_gradients)
    
    # Forward pass
    outputs, attn_weights = model(rgb_input, texture_input)
    
    # Use predicted class if not specified
    if class_idx is None:
        class_idx = outputs.argmax(dim=1)
    
    # One-hot encoding for target class
    one_hot = torch.zeros_like(outputs)
    for i, idx in enumerate(class_idx):
        one_hot[i, idx] = 1
    
    # Backward pass
    model.zero_grad()
    outputs.backward(gradient=one_hot, retain_graph=True)
    
    # Remove hooks
    hook_activation.remove()
    hook_gradient.remove()
    
    # Calculate Grad-CAM
    grads = gradients[0]  # [B, C, H, W]
    acts = activations[0]  # [B, C, H, W]
    
    # Channel-wise mean of gradients
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
    
    # Weighted sum of activations
    cam = torch.sum(weights * acts, dim=1)  # [B, H, W]
    
    # Apply ReLU and normalize
    cam = F.relu(cam)
    
    # Normalize per image
    batch_cams = []
    for i in range(cam.shape[0]):
        cam_i = cam[i]
        if torch.max(cam_i) > 0:
            cam_i = cam_i / torch.max(cam_i)
        batch_cams.append(cam_i)
    
    batch_cams = torch.stack(batch_cams)
    
    return batch_cams.cpu().detach(), attn_weights


def visualize_gradcam_batch(model, dataloader, device, num_samples=16, save_dir='gradcam_results'):
    """
    Visualize Grad-CAM for a batch of samples
    
    Args:
        model: TGVAnn model
        dataloader: Test dataloader
        device: Device to run on
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    model.eval()
    
    # Get class names
    class_names = dataloader.dataset.dataset.classes if hasattr(dataloader.dataset, 'dataset') \
                  else dataloader.dataset.classes
    
    # Get a batch
    data_iter = iter(dataloader)
    rgb_inputs, texture_inputs, labels = next(data_iter)
    
    rgb_inputs = rgb_inputs.to(device)
    texture_inputs = texture_inputs.to(device)
    labels = labels.to(device)
    
    # Limit to num_samples
    rgb_inputs = rgb_inputs[:num_samples]
    texture_inputs = texture_inputs[:num_samples]
    labels = labels[:num_samples]
    
    # Generate Grad-CAM (using layer after TGVA fusion)
    target_layer = model.layer3  # After TGVA fusion
    cams, _ = generate_gradcam(model, rgb_inputs, texture_inputs, target_layer)
    
    # Get predictions
    with torch.no_grad():
        outputs, _ = model(rgb_inputs, texture_inputs)
        _, preds = torch.max(outputs, 1)
    
    # Visualize
    n_rows = int(np.ceil(num_samples / 4))
    fig, axes = plt.subplots(n_rows, 4 * 3, figsize=(20, 5 * n_rows))
    fig.suptitle("Grad-CAM Visualizations", fontsize=20, y=0.995)
    
    for i in range(num_samples):
        row = i // 4
        col = (i % 4) * 3
        
        # Original image
        img = rgb_inputs[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        # Resize CAM to match image size
        cam = cams[i].numpy()
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend original and heatmap
        blended = 0.6 * img + 0.4 * heatmap
        blended = np.clip(blended, 0, 1)
        
        # Plot original
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"True: {class_names[labels[i]]}", fontsize=10)
        axes[row, col].axis('off')
        
        # Plot heatmap
        axes[row, col + 1].imshow(heatmap)
        axes[row, col + 1].set_title("Grad-CAM", fontsize=10)
        axes[row, col + 1].axis('off')
        
        # Plot blended
        axes[row, col + 2].imshow(blended)
        pred_label = class_names[preds[i].item()]
        correct = "✓" if preds[i] == labels[i] else "✗"
        axes[row, col + 2].set_title(f"Pred: {pred_label} {correct}", fontsize=10)
        axes[row, col + 2].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, n_rows * 4):
        row = i // 4
        for col_offset in range(3):
            col = (i % 4) * 3 + col_offset
            axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grad-CAM visualization saved to {save_path}")
    plt.close()


def visualize_per_class(model, dataloader, device, save_dir='gradcam_per_class'):
    """
    Visualize one sample per class
    
    Args:
        model: TGVAnn model
        dataloader: Test dataloader
        device: Device to run on
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    model.eval()
    
    # Get class names
    class_names = dataloader.dataset.dataset.classes if hasattr(dataloader.dataset, 'dataset') \
                  else dataloader.dataset.classes
    num_classes = len(class_names)
    
    # Collect one sample per class
    selected_samples = {i: None for i in range(num_classes)}
    
    for rgb_inputs, texture_inputs, labels in dataloader:
        for i in range(len(labels)):
            label = labels[i].item()
            if selected_samples[label] is None:
                selected_samples[label] = (rgb_inputs[i], texture_inputs[i], labels[i])
            
            # Break if all classes have samples
            if all(v is not None for v in selected_samples.values()):
                break
        
        if all(v is not None for v in selected_samples.values()):
            break
    
    # Prepare batch
    rgb_batch = torch.stack([sample[0] for sample in selected_samples.values()]).to(device)
    texture_batch = torch.stack([sample[1] for sample in selected_samples.values()]).to(device)
    label_batch = torch.stack([sample[2] for sample in selected_samples.values()]).to(device)
    
    # Generate Grad-CAM
    target_layer = model.layer3
    cams, _ = generate_gradcam(model, rgb_batch, texture_batch, target_layer)
    
    # Get predictions
    with torch.no_grad():
        outputs, _ = model(rgb_batch, texture_batch)
        _, preds = torch.max(outputs, 1)
    
    # Visualize
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))
    fig.suptitle("Grad-CAM Per Class (One Sample Each)", fontsize=16)
    
    for i in range(num_classes):
        # Original image
        img = rgb_batch[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        # Resize CAM
        cam = cams[i].numpy()
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend
        blended = 0.6 * img + 0.4 * heatmap
        blended = np.clip(blended, 0, 1)
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {class_names[i]}", fontsize=12)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(blended)
        pred_label = class_names[preds[i].item()]
        correct = "✓" if preds[i] == label_batch[i] else "✗"
        axes[i, 2].set_title(f"Prediction: {pred_label} {correct}", fontsize=12)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_per_class.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class Grad-CAM saved to {save_path}")
    plt.close()


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("\nLoading test dataset...")
    test_loader, num_classes, class_names = prepare_test_dataloader(
        rgb_dir=args.rgb_dir,
        texture_dir=args.texture_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\nLoading TGVAnn model...")
    model = TGVAnn(num_classes=num_classes, input_channels_rgb=3, input_channels_texture=1)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Generate visualizations
    print("\nGenerating Grad-CAM visualizations...")
    
    if args.mode == 'batch':
        visualize_gradcam_batch(model, test_loader, device, args.num_samples, args.output_dir)
    elif args.mode == 'per_class':
        visualize_per_class(model, test_loader, device, args.output_dir)
    else:
        print("Generating both batch and per-class visualizations...")
        visualize_gradcam_batch(model, test_loader, device, args.num_samples, 
                               os.path.join(args.output_dir, 'batch'))
        visualize_per_class(model, test_loader, device, 
                           os.path.join(args.output_dir, 'per_class'))
    
    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Grad-CAM for TGVAnn')
    
    parser.add_argument('--rgb_dir', type=str, required=True, help='Path to RGB test dataset')
    parser.add_argument('--texture_dir', type=str, required=True, help='Path to texture test dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./gradcam_results', help='Output directory')
    parser.add_argument('--mode', type=str, default='both', choices=['batch', 'per_class', 'both'],
                       help='Visualization mode')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples for batch mode')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    main(args)