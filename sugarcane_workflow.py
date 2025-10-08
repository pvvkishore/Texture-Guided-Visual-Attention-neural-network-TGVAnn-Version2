"""
Complete End-to-End Workflow for KSCI Sugarcane Dataset
Demonstrates: Data preparation → Training → Evaluation → Visualization
5 classes: Healthy, Mosaic, RedRot, Rust, Yellow
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TGVAnn, count_parameters
from data import process_dataset, verify_dataset_structure, prepare_dataloaders
import torch.nn as nn
import torch.optim as optim


def step1_prepare_texture_data():
    """Step 1: Generate LDP texture features from RGB images"""
    print("\n" + "="*80)
    print("STEP 1: Preparing Texture Data for KSCI Sugarcane")
    print("="*80)
    
    input_folder = "Sugarcane_RGB"
    output_folder = "Sugarcane_Texture"
    
    if os.path.exists(output_folder):
        print(f"✓ Texture folder already exists: {output_folder}")
    else:
        print(f"Generating LDP texture features...")
        print(f"Input:  {input_folder}")
        print(f"Output: {output_folder}")
        
        process_dataset(
            input_folder=input_folder,
            output_folder=output_folder,
            texture_type='single',
            resize_shape=(256, 256)
        )
        
        print("✓ Texture generation complete!")
    
    # Verify datasets
    print("\nVerifying RGB dataset...")
    verify_dataset_structure(input_folder)
    
    print("\nVerifying Texture dataset...")
    verify_dataset_structure(output_folder)
    
    return input_folder, output_folder


def step2_create_model(num_classes=5):
    """Step 2: Create and analyze TGVAnn model for 5-class classification"""
    print("\n" + "="*80)
    print("STEP 2: Creating TGVAnn Model (5-Class Sugarcane)")
    print("="*80)
    
    model = TGVAnn(
        num_classes=num_classes,
        input_channels_rgb=3,
        input_channels_texture=1
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
    print(f"  Number of Classes: {num_classes}")
    print(f"  Disease Classes: Healthy, Mosaic, RedRot, Rust, Yellow")
    
    return model


def step3_prepare_data(rgb_dir, texture_dir, batch_size=16):
    """Step 3: Prepare data loaders for 5-class dataset"""
    print("\n" + "="*80)
    print("STEP 3: Preparing Data Loaders")
    print("="*80)
    
    train_loader, val_loader, num_classes, class_names = prepare_dataloaders(
        rgb_dir=rgb_dir,
        texture_dir=texture_dir,
        batch_size=batch_size,
        input_size=256,
        num_workers=4,
        augment=True,
        val_split=0.2
    )
    
    print(f"\n✓ Data loaders ready!")
    print(f"  Classes ({num_classes}): {class_names}")
    assert num_classes == 5, f"Expected 5 classes, got {num_classes}"
    
    return train_loader, val_loader, num_classes, class_names


def step4_train_model(model, train_loader, val_loader, num_epochs=10):
    """Step 4: Train the model (simplified quick demo)"""
    print("\n" + "="*80)
    print("STEP 4: Training Model")
    print("="*80)
    print("Note: Using reduced epochs for demo. Use 100+ epochs for best results.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Lower LR for Sugarcane
    
    best_val_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for rgb_imgs, texture_imgs, labels in train_loader:
            rgb_imgs = rgb_imgs.to(device)
            texture_imgs = texture_imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(rgb_imgs, texture_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for rgb_imgs, texture_imgs, labels in val_loader:
                rgb_imgs = rgb_imgs.to(device)
                texture_imgs = texture_imgs.to(device)
                labels = labels.to(device)
                
                outputs, _ = model(rgb_imgs, texture_imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('./outputs/sugarcane', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc
            }, './outputs/sugarcane/best_model.pth')
            print(f"  ✓ Saved best model! Accuracy: {val_acc:.2f}%")
    
    print(f"\n✓ Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    return model


def step5_evaluate_model(model, test_loader, class_names):
    """Step 5: Evaluate the model"""
    print("\n" + "="*80)
    print("STEP 5: Evaluating Model")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for rgb_imgs, texture_imgs, labels in test_loader:
            rgb_imgs = rgb_imgs.to(device)
            texture_imgs = texture_imgs.to(device)
            
            outputs, _ = model(rgb_imgs, texture_imgs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    print(f"\n✓ Test Accuracy: {accuracy:.2f}%")
    print("\nClassification Report (5 Sugarcane Disease Classes):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


def main():
    """Main workflow for KSCI Sugarcane Dataset"""
    print("\n" + "="*80)
    print("TGVAnn Complete Workflow - KSCI Sugarcane Dataset")
    print("5-Class Crop Disease Detection")
    print("Classes: Healthy, Mosaic, RedRot, Rust, Yellow")
    print("="*80)
    
    # Configuration
    NUM_CLASSES = 5
    BATCH_SIZE = 16  # Smaller batch for 5-class
    NUM_EPOCHS = 10  # Increase to 100 for production
    
    try:
        # Step 1: Prepare texture data
        rgb_dir, texture_dir = step1_prepare_texture_data()
        
        # Step 2: Create model
        model = step2_create_model(num_classes=NUM_CLASSES)
        
        # Step 3: Prepare data loaders
        train_loader, val_loader, num_classes, class_names = step3_prepare_data(
            rgb_dir, texture_dir, BATCH_SIZE
        )
        
        # Step 4: Train model
        model = step4_train_model(model, train_loader, val_loader, NUM_EPOCHS)
        
        # Step 5: Evaluate model (using validation set as test for demo)
        step5_evaluate_model(model, val_loader, class_names)
        
        print("\n" + "="*80)
        print("✓ Complete Workflow Finished Successfully!")
        print("="*80)
        print("\nNext Steps for Production:")
        print("1. Train for more epochs (100-150) for better results")
        print("   python train_sugarcane.py --epochs 100")
        print("\n2. Generate Grad-CAM visualizations:")
        print("   python visualize_gradcam_sugarcane.py \\")
        print("       --checkpoint ./outputs/sugarcane/best_model.pth \\")
        print("       --mode all")
        print("\n3. Evaluate on separate test set:")
        print("   python evaluate_sugarcane.py \\")
        print("       --checkpoint ./outputs/sugarcane/best_model.pth")
        print("\n4. Recommended hyperparameters for KSCI Sugarcane:")
        print("   - Batch size: 16")
        print("   - Learning rate: 1e-5")
        print("   - Epochs: 100-150")
        print("   - Scheduler: Cosine Annealing")
        print("   - Data augmentation: Enabled")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
