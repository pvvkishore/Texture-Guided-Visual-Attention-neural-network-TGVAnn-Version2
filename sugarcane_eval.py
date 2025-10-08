"""
Evaluation script for TGVAnn on KSCI Sugarcane Dataset
5 classes: Healthy, Mosaic, RedRot, Rust, Yellow
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tgvann import TGVAnn
from models.utils import count_parameters
from data.dataset import prepare_test_dataloader


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for rgb_imgs, texture_imgs, labels in tqdm(test_loader, desc='Evaluating Sugarcane'):
            rgb_imgs = rgb_imgs.to(device)
            texture_imgs = texture_imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, _ = model(rgb_imgs, texture_imgs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    return accuracy, all_labels, all_preds


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_sugarcane.png'):
    """Plot confusion matrix for Sugarcane dataset"""
    plt.figure(figsize=(12, 10))
    
    # Custom colormap for better visualization
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray')
    
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.title('KSCI Sugarcane - Confusion Matrix\n5-Class Disease Classification', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(labels, preds, class_names, save_path='per_class_metrics_sugarcane.png'):
    """Plot per-class precision, recall, F1-score"""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(len(class_names))
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision * 100, width, label='Precision', 
                   color='#667eea', alpha=0.8)
    bars2 = ax.bar(x, recall * 100, width, label='Recall', 
                   color='#764ba2', alpha=0.8)
    bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', 
                   color='#28a745', alpha=0.8)
    
    ax.set_xlabel('Disease Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('KSCI Sugarcane - Per-Class Performance Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics plot saved to {save_path}")


def print_classification_report(labels, preds, class_names):
    """Print detailed classification report"""
    print("\n" + "="*80)
    print("KSCI Sugarcane Dataset - Classification Report")
    print("="*80)
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(len(class_names))
    )
    
    print("\n" + "="*80)
    print("Per-Class Metrics Summary (5 Disease Classes)")
    print("="*80)
    print(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-"*80)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:>12.4f} {recall[i]:>12.4f} "
              f"{f1[i]:>12.4f} {support[i]:>10.0f}")
    
    # Overall metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    print("-"*80)
    print(f"{'Macro Avg':<15} {macro_precision:>12.4f} {macro_recall:>12.4f} {macro_f1:>12.4f}")
    print(f"{'Weighted Avg':<15} {weighted_precision:>12.4f} {weighted_recall:>12.4f} {weighted_f1:>12.4f}")
    print("="*80)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("TGVAnn Evaluation - KSCI Sugarcane Dataset")
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
    assert num_classes == 5, f"Expected 5 classes for Sugarcane, got {num_classes}"
    
    # Create model
    print("\nLoading TGVAnn model (5-class)...")
    model = TGVAnn(num_classes=5, input_channels_rgb=3, input_channels_texture=1)
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Previous validation accuracy: {checkpoint.get('accuracy', 0):.2f}%")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Evaluate
    print("\nEvaluating model on KSCI Sugarcane test set...")
    accuracy, all_labels, all_preds = evaluate_model(model, test_loader, device, class_names)
    
    print(f"\n{'='*80}")
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix_sugarcane.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Plot per-class metrics
    metrics_path = os.path.join(args.output_dir, 'per_class_metrics_sugarcane.png')
    plot_per_class_metrics(all_labels, all_preds, class_names, metrics_path)
    
    # Print classification report
    print_classification_report(all_labels, all_preds, class_names)
    
    # Save results to file
    results_path = os.path.join(args.output_dir, 'evaluation_results_sugarcane.txt')
    with open(results_path, 'w') as f:
        f.write(f"TGVAnn Evaluation Results - KSCI Sugarcane Dataset\n")
        f.write(f"{'='*80}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Dataset: {args.rgb_dir}\n")
        f.write(f"Number of Classes: 5\n")
        f.write(f"Class Names: {', '.join(class_names)}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"{'='*80}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
        f.write(f"\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"\n✓ Evaluation complete for KSCI Sugarcane dataset!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TGVAnn on KSCI Sugarcane Dataset')
    
    parser.add_argument('--rgb_dir', type=str, default='./Sugarcane_RGB',
                       help='Path to Sugarcane RGB test dataset')
    parser.add_argument('--texture_dir', type=str, default='./Sugarcane_Texture',
                       help='Path to Sugarcane texture test dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./eval_results/sugarcane',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    main(args)