"""
Evaluation script for TGVAnn
Computes accuracy, confusion matrix, and classification metrics
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
    """
    Evaluate model on test set
    
    Returns:
        accuracy, all_labels, all_preds
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for rgb_imgs, texture_imgs, labels in tqdm(test_loader, desc='Evaluating'):
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


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")


def print_classification_report(labels, preds, class_names):
    """Print detailed classification report"""
    print("\n" + "="*80)
    print("Classification Report")
    print("="*80)
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(len(class_names))
    )
    
    print("\n" + "="*80)
    print("Per-Class Metrics Summary")
    print("="*80)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*80)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} "
              f"{f1[i]:>10.4f} {support[i]:>10.0f}")
    
    # Overall metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    print("-"*80)
    print(f"{'Macro Avg':<20} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f}")
    print(f"{'Weighted Avg':<20} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f}")
    print("="*80)


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
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Evaluate
    print("\nEvaluating model...")
    accuracy, all_labels, all_preds = evaluate_model(model, test_loader, device, class_names)
    
    print(f"\n{'='*80}")
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Print classification report
    print_classification_report(all_labels, all_preds, class_names)
    
    # Save results to file
    results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"TGVAnn Evaluation Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Dataset: {args.rgb_dir}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Class Names: {', '.join(class_names)}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"{'='*80}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TGVAnn model')
    
    parser.add_argument('--rgb_dir', type=str, required=True, help='Path to RGB test dataset')
    parser.add_argument('--texture_dir', type=str, required=True, help='Path to texture test dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    main(args)