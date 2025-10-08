# TGVAnn for KSCI Sugarcane Dataset

**5-Class Sugarcane Disease Detection using Texture-Guided Visual Attention Network**

## 🌾 Dataset Information

**KSCI (Kaggle Sugarcane Crop Images) Dataset**
- **Total Classes**: 5
- **Disease Classes**:
  1. **Healthy** - Normal sugarcane leaves
  2. **Mosaic** - Mosaic virus symptoms
  3. **RedRot** - Red rot disease
  4. **Rust** - Rust fungal infection
  5. **Yellow** - Yellow leaf disease

## 📋 Quick Start

### 1. Generate Texture Features (5 minutes)

```bash
# Generate LDP textures from RGB images
python generate_texture_sugarcane.py
```

This will:
- Read RGB images from `Sugarcane_RGB/`
- Generate LDP texture features
- Save to `Sugarcane_Texture/`
- Verify dataset structure

**Expected Output:**
```
KSCI Sugarcane Dataset Structure:
  Healthy: XXX images
  Mosaic: XXX images
  RedRot: XXX images
  Rust: XXX images
  Yellow: XXX images
```

### 2. Train Model (Recommended: 2-4 hours on GPU)

```bash
# Quick training (10 epochs for testing)
python train_sugarcane.py \
    --rgb_dir ./Sugarcane_RGB \
    --texture_dir ./Sugarcane_Texture \
    --output_dir ./outputs/sugarcane \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-5

# Full training (production - recommended)
python train_sugarcane.py \
    --rgb_dir ./Sugarcane_RGB \
    --texture_dir ./Sugarcane_Texture \
    --output_dir ./outputs/sugarcane \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-5 \
    --scheduler cosine \
    --augment
```

### 3. Evaluate Model

```bash
python evaluate_sugarcane.py \
    --rgb_dir ./Sugarcane_RGB \
    --texture_dir ./Sugarcane_Texture \
    --checkpoint ./outputs/sugarcane/best_model_sugarcane.pth \
    --output_dir ./eval_results/sugarcane
```

### 4. Generate Grad-CAM Visualizations

```bash
# Generate all visualization types
python visualize_gradcam_sugarcane.py \
    --rgb_dir ./Sugarcane_RGB \
    --texture_dir ./Sugarcane_Texture \
    --checkpoint ./outputs/sugarcane/best_model_sugarcane.pth \
    --mode all

# Or specific modes
python visualize_gradcam_sugarcane.py \
    --checkpoint ./outputs/sugarcane/best_model_sugarcane.pth \
    --mode per_class  # One sample per disease class

python visualize_gradcam_sugarcane.py \
    --checkpoint ./outputs/sugarcane/best_model_sugarcane.pth \
    --mode batch --num_samples 15  # Multiple samples

python visualize_gradcam_sugarcane.py \
    --checkpoint ./outputs/sugarcane/best_model_sugarcane.pth \
    --mode compare  # Compare different layers
```

## ⚙️ Recommended Hyperparameters

### KSCI Sugarcane Specific Settings

| Parameter | Value | Reason |
|-----------|-------|--------|
| Batch Size | 16 | Optimal for 5-class with limited GPU memory |
| Learning Rate | 1e-5 | Lower LR for better convergence on 5 classes |
| Epochs | 100-150 | More epochs needed for 5-class classification |
| Scheduler | Cosine Annealing | Smooth LR decay |
| Weight Decay | 1e-4 | Standard regularization |
| Input Size | 256×256 | Balanced performance/accuracy |
| Val Split | 0.2 | 20% for validation |
| Augmentation | Enabled | Critical for generalization |

### Why Different from Maize (3-class)?

- **More classes** (5 vs 3) → Need more training time
- **Lower learning rate** → Better fine-grained discrimination
- **Smaller batch size** → Better gradient estimates for 5-way classification
- **More epochs** → Convergence takes longer with more classes

## 📊 Expected Performance

### Typical Results (After 100 epochs)

| Metric | Expected Range |
|--------|----------------|
| Overall Accuracy | 88-94% |
| Macro Precision | 87-93% |
| Macro Recall | 86-92% |
| Macro F1-Score | 87-93% |

### Per-Class Performance

| Disease | Typical Accuracy |
|---------|-----------------|
| Healthy | 92-97% (easiest to detect) |
| Mosaic | 88-93% |
| RedRot | 85-91% |
| Rust | 87-92% |
| Yellow | 84-90% (most challenging) |

*Note: Actual results depend on dataset quality and training duration*

## 🎯 Training Tips for Sugarcane

### 1. Data Augmentation is Critical
```python
# Enabled by default in train_sugarcane.py
--augment  # Includes: flips, rotation, color jitter
```

### 2. Monitor Class Imbalance
```bash
# Check class distribution
python -c "from data.ldp_texture_generator import verify_dataset_structure; verify_dataset_structure('Sugarcane_RGB')"
```

### 3. Learning Rate Schedule
```bash
# Use cosine annealing for smooth convergence
--scheduler cosine

# Or reduce on plateau for adaptive learning
--scheduler plateau
```

### 4. Early Stopping
- Monitor validation accuracy
- Stop if no improvement for 20 epochs
- Best model is automatically saved

## 🔍 Grad-CAM Interpretation Guide

### What to Look For in Visualizations

**Healthy Class:**
- Uniform attention across leaf surface
- Low activation intensity
- Broad, diffuse patterns

**Mosaic:**
- Attention on mosaic patterns
- High activation on leaf veins
- Localized bright spots

**RedRot:**
- Strong activation on reddish areas
- Attention to discolored patches
- Edge focus

**Rust:**
- Concentrated on rust pustules
- Small, scattered activation points
- Orange/brown region focus

**Yellow:**
- Attention on yellowing areas
- Leaf tip/edge activation
- Chlorotic region focus

## 📁 Output Files Structure

After running the complete workflow:

```
outputs/sugarcane/
├── best_model_sugarcane.pth          # Best model checkpoint
├── final_model_sugarcane.pth         # Final epoch model
├── training_history_sugarcane.png    # Loss/accuracy curves
└── sugarcane_checkpoint_epoch_*.pth  # Periodic checkpoints

eval_results/sugarcane/
├── confusion_matrix_sugarcane.png    # 5×5 confusion matrix
├── per_class_metrics_sugarcane.png   # Precision/Recall/F1 chart
└── evaluation_results_sugarcane.txt  # Detailed metrics

gradcam_results/sugarcane/
├── per_class/
│   └── gradcam_per_class_sugarcane.png      # One per disease
├── batch/
│   └── gradcam_batch_sugarcane.png          # Multiple samples
└── comparison/
    └── gradcam_layer_comparison_sugarcane.png  # Layer3 vs Layer4
```

## 🐛 Troubleshooting

### Issue: Out of Memory with 5 Classes

**Solution:**
```bash
# Reduce batch size
python train_sugarcane.py ... --batch_size 8

# Or reduce input size
python train_sugarcane.py ... --input_size 224 --batch_size 16
```

### Issue: Low Accuracy on Specific Class

**Solution:**
1. Check class balance in dataset
2. Increase data augmentation
3. Use class weights in loss function:
```python
# In train_sugarcane.py, modify:
class_weights = torch.FloatTensor([1.0, 1.2, 1.3, 1.1, 1.4]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Issue: Training Plateaus Early

**Solution:**
```bash
# Lower learning rate
python train_sugarcane.py ... --lr 1e-6

# Use plateau scheduler
python train_sugarcane.py ... --scheduler plateau

# Train longer
python train_sugarcane.py ... --epochs 150
```

### Issue: Overfitting (High train acc, low val acc)

**Solution:**
```bash
# Increase validation split
python train_sugarcane.py ... --val_split 0.25

# More aggressive augmentation (modify in dataset.py)
# Increase dropout in model
```

## 📈 Comparison: Maize vs Sugarcane

| Aspect | Maize (TMCI) | Sugarcane (KSCI) |
|--------|--------------|------------------|
| Classes | 3 | 5 |
| Batch Size | 32 | 16 |
| Learning Rate | 1e-4 | 1e-5 |
| Recommended Epochs | 50 | 100-150 |
| Typical Accuracy | 94-98% | 88-94% |
| Training Time | 1-2 hours | 2-4 hours |

## 🔬 Advanced Usage

### Custom Training Script

```python
from models.tgvann import TGVAnn
from data.dataset import prepare_dataloaders

# Custom configuration for Sugarcane
model = TGVAnn(num_classes=5)
train_loader, val_loader, _, _ = prepare_dataloaders(
    rgb_dir='./Sugarcane_RGB',
    texture_dir='./Sugarcane_Texture',
    batch_size=16,
    augment=True
)

# Train with your custom loop
# ...
```

### Ensemble Models

```python
# Train multiple models with different seeds
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    model = TGVAnn(num_classes=5)
    # Train...

# Ensemble predictions for better accuracy
```

### Transfer Learning from Maize

```python
# Load Maize pretrained model
maize_model = TGVAnn(num_classes=3)
maize_checkpoint = torch.load('maize_model.pth')
maize_model.load_state_dict(maize_checkpoint['model_state_dict'])

# Create Sugarcane model and copy weights
sugarcane_model = TGVAnn(num_classes=5)
# Copy all layers except final classifier
# Fine-tune on Sugarcane dataset
```

## 📚 Citation

If you use this code for KSCI Sugarcane research:

```bibtex
@article{kishore2025tgvann_sugarcane,
  title={Texture Feature Guided Attention based Fusion for Sugarcane Disease Detection},
  author={Kishore, P. V. V. and others},
  journal={Conference/Journal Name},
  year={2025},
  note={KSCI Dataset - 5 Disease Classes}
}
```

## 🤝 Dataset Credits

**KSCI Dataset**: Kaggle Sugarcane Crop Images
- Source: [Kaggle Dataset Link]
- Classes: Healthy, Mosaic, RedRot, Rust, Yellow
- License: [Dataset License]

## ✨ Summary

**For KSCI Sugarcane (5-class):**
1. ✅ Use `batch_size=16` and `lr=1e-5`
2. ✅ Train for 100+ epochs
3. ✅ Enable data augmentation
4. ✅ Use Cosine Annealing scheduler
5. ✅ Monitor per-class performance
6. ✅ Generate Grad-CAM for interpretability

**Expected Training Time:**
- Quick test (10 epochs): 20-30 minutes
- Production (100 epochs): 2-4 hours on GPU

---

**Ready to train? Run:**
```bash
python example_workflow_sugarcane.py
```

This will guide you through the complete pipeline! 🌾
