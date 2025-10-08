# TGVAnn: Texture-Guided Visual Attention Network

Official PyTorch implementation of **"Texture Feature Guided Attention based Fusion representations for Crop Leaf Disease Detection"**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Model Performance](#model-performance)
- [Citation](#citation)

## 🔍 Overview

TGVAnn is a dual-stream deep learning architecture for crop leaf disease detection that fuses RGB and texture features using a novel **Texture-Guided Visual Attention (TGVA)** mechanism. The model achieves state-of-the-art performance on crop disease classification tasks.

### Key Features
- ✅ Dual-stream architecture processing RGB and LDP texture features
- ✅ TGVA cross-attention module for effective multimodal fusion
- ✅ Single-head attention with gated fusion mechanism
- ✅ Comprehensive Grad-CAM visualizations for interpretability
- ✅ Reproducible training and evaluation pipeline

## 🏗️ Architecture

TGVAnn consists of three main components:

1. **Dual-Stream Feature Extraction**: Parallel ResNet-based encoders for RGB and texture streams
2. **TGVA Fusion Module**: Cross-attention mechanism at ResBlock-2 (32×32×128 → 32×32×256)
3. **Classification Head**: Global average pooling + fully connected layers

### TGVA Configuration
- **d_model**: 128
- **Heads**: 1 (single-head attention)
- **d_ff**: 512 (FFN hidden dimension)
- **Tokens**: 1024 (32×32 spatial resolution)
- **Dropouts**: p_attn=0.10, p_ff=0.10
- **Fusion**: Gated add-fusion with learnable γ

### Model Statistics
- **Total Parameters**: ~X.XX M
- **Trainable Parameters**: ~X.XX M
- **FLOPs**: X.XX GFLOPs (for 256×256 input)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 8GB+ GPU memory recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/pvvkishore/Texture-Feature-Guided-Attention-based-Fusion-AIA-Mar30-2025.git
cd Texture-Feature-Guided-Attention-based-Fusion-AIA-Mar30-2025

# Create virtual environment
conda create -n tgvann python=3.9
conda activate tgvann

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (choose appropriate CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📁 Dataset Preparation

### Directory Structure
```
Your_Dataset/
├── RGB/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
└── Texture/  # Generated automatically
    ├── class1/
    ├── class2/
    └── ...
```

### Generate LDP Texture Features

```bash
# For Maize dataset (TMCI)
python data/ldp_texture_generator.py

# Or modify the script for custom datasets
```

Edit the script to specify your dataset paths:
```python
input_folder = "path/to/your/RGB/dataset"
output_folder = "path/to/save/texture/dataset"
```

### Example: Maize Dataset (TMCI)
```bash
# Assuming your RGB data is in Maize_RGB/
python data/ldp_texture_generator.py
# This will create Maize_Texture/ with LDP features
```

## ⚡ Quick Start

### Train on Maize Dataset

```bash
python train.py \
    --rgb_dir ./Maize_RGB \
    --texture_dir ./Maize_Texture \
    --output_dir ./outputs/maize_experiment \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --input_size 256 \
    --augment \
    --print_model
```

### Evaluate Trained Model

```bash
python evaluate.py \
    --rgb_dir ./Maize_RGB \
    --texture_dir ./Maize_Texture \
    --checkpoint ./outputs/maize_experiment/best_model.pth \
    --output_dir ./eval_results
```

### Generate Grad-CAM Visualizations

```bash
python visualize_gradcam.py \
    --rgb_dir ./Maize_RGB \
    --texture_dir ./Maize_Texture \
    --checkpoint ./outputs/maize_experiment/best_model.pth \
    --output_dir ./gradcam_results \
    --mode both \
    --num_samples 16
```

## 🔧 Training

### Basic Training
```bash
python train.py \
    --rgb_dir <path_to_rgb> \
    --texture_dir <path_to_texture> \
    --output_dir <output_path>
```

### Advanced Training Options

```bash
python train.py \
    --rgb_dir ./data/RGB \
    --texture_dir ./data/Texture \
    --output_dir ./outputs/experiment_1 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --input_size 256 \
    --val_split 0.2 \
    --num_workers 8 \
    --augment \
    --save_freq 10 \
    --print_model
```

### Training Arguments
- `--rgb_dir`: Path to RGB dataset
- `--texture_dir`: Path to texture dataset
- `--output_dir`: Directory to save outputs
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)
- `--scheduler`: LR scheduler ('cosine', 'plateau', 'none')
- `--input_size`: Input image size (default: 256)
- `--val_split`: Validation split ratio (default: 0.2)
- `--augment`: Enable data augmentation
- `--save_freq`: Save checkpoint every N epochs
- `--print_model`: Print detailed model summary

## 📊 Evaluation

### Evaluate on Test Set

```bash
python evaluate.py \
    --rgb_dir ./test_data/RGB \
    --texture_dir ./test_data/Texture \
    --checkpoint ./outputs/best_model.pth \
    --output_dir ./eval_results \
    --batch_size 32
```

### Evaluation Outputs
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix (saved as PNG)
- Detailed classification report (saved as TXT)

## 🎨 Visualization

### Grad-CAM Attention Maps

```bash
# Visualize batch of samples
python visualize_gradcam.py \
    --rgb_dir ./data/RGB \
    --texture_dir ./data/Texture \
    --checkpoint ./outputs/best_model.pth \
    --mode batch \
    --num_samples 16

# Visualize one sample per class
python visualize_gradcam.py \
    --rgb_dir ./data/RGB \
    --texture_dir ./data/Texture \
    --checkpoint ./outputs/best_model.pth \
    --mode per_class

# Generate both
python visualize_gradcam.py \
    --rgb_dir ./data/RGB \
    --texture_dir ./data/Texture \
    --checkpoint ./outputs/best_model.pth \
    --mode both
```

## 📈 Model Performance

### Maize Dataset (TMCI) - 3 Classes
| Metric | Value |
|--------|-------|
| Overall Accuracy | XX.XX% |
| Macro Precision | XX.XX% |
| Macro Recall | XX.XX% |
| Macro F1-Score | XX.XX% |

### Per-Class Performance (Example)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Healthy | XX.XX% | XX.XX% | XX.XX% |
| Disease 1 | XX.XX% | XX.XX% | XX.XX% |
| Disease 2 | XX.XX% | XX.XX% | XX.XX% |

## 📂 Project Structure

```
TGVAnn/
├── models/
│   ├── __init__.py
│   ├── tgvann.py              # Main TGVAnn architecture
│   ├── tgva_attention.py      # TGVA attention module
│   └── utils.py               # Model utilities
├── data/
│   ├── __init__.py
│   ├── ldp_texture_generator.py   # LDP texture generation
│   └── dataset.py             # Dataset loaders
├── configs/
│   ├── maize.yaml             # Maize dataset config
│   └── sugarcane.yaml         # Sugarcane dataset config
├── scripts/
│   ├── prepare_data.sh        # Data preparation script
│   └── reproduce_results.sh   # Reproduce paper results
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── visualize_gradcam.py       # Grad-CAM visualization
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── ARCHITECTURE.md            # Detailed architecture description
```

## 🔬 Reproducing Paper Results

### Step 1: Prepare Dataset
```bash
# Download Maize dataset (TMCI)
# Place RGB images in Maize_RGB/

# Generate texture features
python data/ldp_texture_generator.py
```

### Step 2: Train Model
```bash
# Train with paper settings
python train.py \
    --rgb_dir ./Maize_RGB \
    --texture_dir ./Maize_Texture \
    --output_dir ./outputs/paper_reproduction \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --input_size 256 \
    --augment
```

### Step 3: Evaluate
```bash
python evaluate.py \
    --rgb_dir ./Maize_RGB \
    --texture_dir ./Maize_Texture \
    --checkpoint ./outputs/paper_reproduction/best_model.pth \
    --output_dir ./paper_results
```

### Step 4: Generate Visualizations
```bash
python visualize_gradcam.py \
    --rgb_dir ./Maize_RGB \
    --texture_dir ./Maize_Texture \
    --checkpoint ./outputs/paper_reproduction/best_model.pth \
    --output_dir ./paper_gradcam \
    --mode both
```

## 🧪 Model Analysis

### Calculate Model Parameters and FLOPs

```python
from models.tgvann import TGVAnn
from models.utils import model_summary

# Create model
model = TGVAnn(num_classes=3)

# Print comprehensive summary
model_summary(model, input_size=(1, 3, 256, 256), 
              texture_size=(1, 1, 256, 256), device='cuda')
```

### Expected Output:
```
================================================================================
TGVAnn Model Summary
================================================================================

Total Parameters: X,XXX,XXX
Trainable Parameters: X,XXX,XXX
Non-trainable Parameters: 0

FLOPs: X.XXX GFLOPs
Params (from THOP): X.XXX M
================================================================================
```

## 💡 Tips for Best Results

### Data Augmentation
- Enable with `--augment` flag
- Includes: random flips, rotation, color jitter
- Significantly improves generalization

### Learning Rate Scheduling
- **Cosine Annealing**: Smooth decay, good for long training
- **Reduce on Plateau**: Adaptive, good for convergence
- Start with `lr=1e-4` for most datasets

### Batch Size Selection
- **Large datasets**: Use batch_size=32 or 64
- **Small datasets**: Use batch_size=16 or 8
- Adjust based on GPU memory (8GB GPU → batch_size=16-32)

### Input Size
- Default: 256×256 (optimal for TGVA architecture)
- Can use 224×224 for faster training
- Higher resolutions (384×384) may improve accuracy but slower

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
```bash
# Reduce batch size
python train.py ... --batch_size 16

# Or reduce input size
python train.py ... --input_size 224 --batch_size 32
```

### Corrupted Image Files
The code automatically detects and removes corrupted images:
```python
from data.dataset import remove_corrupted_images
remove_corrupted_images("path/to/dataset")
```

### Low Accuracy
- Ensure texture features are properly generated
- Enable data augmentation with `--augment`
- Try different learning rates: 1e-3, 1e-4, 1e-5
- Increase number of epochs
- Check dataset class balance

### Installation Issues
```bash
# THOP installation (for FLOPs calculation)
pip install thop

# If OpenCV issues
pip uninstall opencv-python
pip install opencv-python-headless

# CUDA compatibility
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Additional Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture description
- [configs/](configs/) - Configuration file examples
- [scripts/](scripts/) - Helper scripts for data preparation

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**P. V. V. Kishore**
- Email: pvvkishore@example.com
- GitHub: [@pvvkishore](https://github.com/pvvkishore)

## 🙏 Acknowledgments

- ResNet architecture from [torchvision](https://pytorch.org/vision/stable/models.html)
- Grad-CAM implementation inspired by [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- LDP texture descriptor from original paper

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{kishore2025tgvann,
  title={Texture Feature Guided Attention based Fusion representations for Crop Leaf Disease Detection},
  author={Kishore, P. V. V. and others},
  journal={Conference/Journal Name},
  year={2025}
}
```

## 🔄 Updates

### Version 1.0.0 (March 2025)
- Initial release
- TGVAnn architecture implementation
- Training and evaluation scripts
- Grad-CAM visualization
- LDP texture generation
- Comprehensive documentation

---

**Note**: Update the performance metrics, contact information, and citation details with your actual paper information before publication.