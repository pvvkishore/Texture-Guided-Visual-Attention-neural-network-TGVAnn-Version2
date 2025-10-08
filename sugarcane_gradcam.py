"""
Grad-CAM Visualization for TGVAnn on KSCI Sugarcane Dataset
5 classes: Healthy, Mosaic, RedRot, Rust, Yellow
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
    """Generate Grad-CAM for given inputs"""
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
    grads = gradients[0]
    acts = activations[0]
    
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * acts, dim=1)
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


def visualize_gradcam_per_class(model, dataloader, device, save_dir='gradcam_sugarcane'):
    """Visualize one sample per class for all 5 sugarcane diseases"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    model.eval()
    
    # Get class names
    class_names = dataloader.dataset.dataset.classes if hasattr(dataloader.dataset, 'dataset') \
                  else dataloader.dataset.classes
    num_classes = len(class_names)
    
    print(f"\nGenerating Grad-CAM for {num_classes} Sugarcane disease classes...")
    print(f"Classes: {', '.join(class_names)}")
    
    # Collect one sample per class
    selected_samples = {i: None for i in range(num_classes)}
    
    for rgb_inputs, texture_inputs, labels in dataloader:
        for i in range(len(labels)):
            label = labels[i].item()
            if selected_samples[label] is None:
                selected_samples[label] = (rgb_inputs[i], texture_inputs[i], labels[i])
            
            if all(v is not None for v in selected_samples.values()):
                break
        
        if all(v is not None for v in selected_samples.values()):
            break
    
    # Prepare batch
    rgb_batch = torch.stack([sample[0] for sample in selected_samples.values()]).to(device)
    texture_batch = torch.stack([sample[1] for sample in selected_samples.values()]).to(device)
    label_batch = torch.stack([sample[2] for sample in selected_samples.values()]).to(device)
    
    # Generate Grad-CAM
    target_layer = model.layer3  # After TGVA fusion
    cams, _ = generate_gradcam(model, rgb_batch, texture_batch, target_layer)
    
    # Get predictions
    with torch.no_grad():
        outputs, _ = model(rgb_batch, texture_batch)
        _, preds = torch.max(outputs, 1)
        probs = F.softmax(outputs, dim=1)
    
    # Visualize - 5 rows (one per class), 3 columns (original, heatmap, overlay)
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))
    fig.suptitle("KSCI Sugarcane - Grad-CAM Visualization (5 Disease Classes)", 
                 fontsize=18, fontweight='bold', y=0.995)
    
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
        
        # Get prediction confidence
        confidence = probs[i][preds[i]].item() * 100
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {class_names[i]}", fontsize=14, fontweight='bold')
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=14, fontweight='bold')
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(blended)
        pred_label = class_names[preds[i].item()]
        correct = "✓" if preds[i] == label_batch[i] else "✗"
        axes[i, 2].set_title(f"Prediction: {pred_label} ({confidence:.1f}%) {correct}", 
                            fontsize=14, fontweight='bold',
                            color='green' if correct == "✓" else 'red')
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_per_class_sugarcane.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Per-class Grad-CAM saved to {save_path}")
    plt.close()


def visualize_gradcam_batch(model, dataloader, device, num_samples=15, save_dir='gradcam_sugarcane'):
    """Visualize Grad-CAM for multiple samples"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    model.eval()
    
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
    
    # Generate Grad-CAM
    target_layer = model.layer3
    cams, _ = generate_gradcam(model, rgb_inputs, texture_inputs, target_layer)
    
    # Get predictions
    with torch.no_grad():
        outputs, _ = model(rgb_inputs, texture_inputs)
        _, preds = torch.max(outputs, 1)
        probs = F.softmax(outputs, dim=1)
    
    # Visualize - 5 samples per row, 3 columns per sample
    n_rows = int(np.ceil(num_samples / 5))
    fig, axes = plt.subplots(n_rows, 5 * 3, figsize=(25, 5 * n_rows))
    fig.suptitle("KSCI Sugarcane - Batch Grad-CAM Visualizations", 
                 fontsize=20, fontweight='bold', y=0.998)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        row = i // 5
        col = (i % 5) * 3
        
        # Original image
        img = rgb_inputs[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        cam = cams[i].numpy()
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        blended = 0.6 * img + 0.4 * heatmap
        blended = np.clip(blended, 0, 1)
        
        confidence = probs[i][preds[i]].item() * 100
        
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
        axes[row, col + 2].set_title(f"{pred_label} ({confidence:.0f}%) {correct}", fontsize=10)
        axes[row, col + 2].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, n_rows * 5):
        row = i // 5
        for col_offset in range(3):
            col = (i % 5) * 3 + col_offset
            axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_batch_sugarcane.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Batch Grad-CAM saved to {save_path}")
    plt.close()


def compare_layer_gradcams(model, dataloader, device, save_dir='gradcam_sugarcane'):
    """Compare Grad-CAM from different layers"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    model.eval()
    
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
            if all(v is not None for v in selected_samples.values()):
                break
        if all(v is not None for v in selected_samples.values()):
            break
    
    rgb_batch = torch.stack([sample[0] for sample in selected_samples.values()]).to(device)
    texture_batch = torch.stack([sample[1] for sample in selected_samples.values()]).to(device)
    label_batch = torch.stack([sample[2] for sample in selected_samples.values()]).to(device)
    
    # Generate Grad-CAM from layer3 and layer4
    layer3_cams, _ = generate_gradcam(model, rgb_batch, texture_batch, model.layer3)
    layer4_cams, _ = generate_gradcam(model, rgb_batch, texture_batch, model.layer4)
    
    with torch.no_grad():
        outputs, _ = model(rgb_batch, texture_batch)
        _, preds = torch.max(outputs, 1)
    
    # Visualize comparison
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))
    fig.suptitle("KSCI Sugarcane - Layer Comparison (Layer3 vs Layer4)", 
                 fontsize=18, fontweight='bold', y=0.995)
    
    for i in range(num_classes):
        img = rgb_batch[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        # Layer 3 CAM
        cam3 = layer3_cams[i].numpy()
        cam3_resized = cv2.resize(cam3, (img.shape[1], img.shape[0]))
        heatmap3 = cv2.applyColorMap(np.uint8(255 * cam3_resized), cv2.COLORMAP_JET)
        heatmap3 = cv2.cvtColor(heatmap3, cv2.COLOR_BGR2RGB) / 255.0
        blended3 = 0.6 * img + 0.4 * heatmap3
        blended3 = np.clip(blended3, 0, 1)
        
        # Layer 4 CAM
        cam4 = layer4_cams[i].numpy()
        cam4_resized = cv2.resize(cam4, (img.shape[1], img.shape[0]))
        heatmap4 = cv2.applyColorMap(np.uint8(255 * cam4_resized), cv2.COLORMAP_JET)
        heatmap4 = cv2.cvtColor(heatmap4, cv2.COLOR_BGR2RGB) / 255.0
        blended4 = 0.6 * img + 0.4 * heatmap4
        blended4 = np.clip(blended4, 0, 1)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {class_names[i]}", fontsize=14, fontweight='bold')
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(blended3)
        axes[i, 1].set_title("Layer3 Grad-CAM (After TGVA)", fontsize=14, fontweight='bold')
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(blended4)
        axes[i, 2].set_title("Layer4 Grad-CAM (High-level)", fontsize=14, fontweight='bold')
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_layer_comparison_sugarcane.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Layer comparison saved to {save_path}")
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("TGVAnn Grad-CAM Visualization - KSCI Sugarcane Dataset")
    print("5 Classes: Healthy, Mosaic, RedRot, Rust, Yellow")
    print("="*80)
    
    # Load test data
    print("\nLoading KSCI Sugarcane test dataset...")
    test_loader, num_classes, class_names = prepare_test_dataloader(
        rgb_dir=args.rgb_dir,
        texture_dir=args.texture_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers
    )
    
    print(f"Dataset Classes: {class_names}")
    assert num_classes == 5, f"Expected 5 classes, got {num_classes}"
    
    # Create model
    print("\nLoading TGVAnn model...")
    model = TGVAnn(num_classes=5, input_channels_rgb=3, input_channels_texture=1)
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
        visualize_gradcam_per_class(model, test_loader, device, args.output_dir)
    elif args.mode == 'compare':
        compare_layer_gradcams(model, test_loader, device, args.output_dir)
    else:  # 'all'
        print("\nGenerating all visualization types...")
        visualize_gradcam_per_class(model, test_loader, device, 
                                    os.path.join(args.output_dir, 'per_class'))
        visualize_gradcam_batch(model, test_loader, device, args.num_samples,
                               os.path.join(args.output_dir, 'batch'))
        compare_layer_gradcams(model, test_loader, device,
                              os.path.join(args.output_dir, 'comparison'))
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}")
    print("✓ Grad-CAM generation complete for KSCI Sugarcane dataset!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Grad-CAM for KSCI Sugarcane')
    
    parser.add_argument('--rgb_dir', type=str, default='./Sugarcane_RGB',
                       help='Path to Sugarcane RGB test dataset')
    parser.add_argument('--texture_dir', type=str, default='./Sugarcane_Texture',
                       help='Path to Sugarcane texture test dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./gradcam_results/sugarcane',
                       help='Output directory')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['batch', 'per_class', 'compare', 'all'],
                       help='Visualization mode')
    parser.add_argument('--num_samples', type=int, default=15,
                       help='Number of samples for batch mode')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    main(args)