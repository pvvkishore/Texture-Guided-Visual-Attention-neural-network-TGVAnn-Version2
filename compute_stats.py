"""
Compute and display TGVAnn model statistics
Calculates parameters, FLOPs, and layer-wise information
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.tgvann import TGVAnn
from models.utils import count_parameters, model_summary, print_layer_shapes


def main():
    print("\n" + "="*80)
    print("TGVAnn Model Statistics Calculator")
    print("="*80)
    
    # Configuration
    num_classes = 3  # Maize dataset (TMCI)
    input_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Input size: {input_size}×{input_size}")
    print(f"  - Device: {device}")
    
    # Create model
    print(f"\nCreating TGVAnn model...")
    model = TGVAnn(num_classes=num_classes, 
                   input_channels_rgb=3, 
                   input_channels_texture=1)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n" + "="*80)
    print("Parameter Statistics")
    print("="*80)
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Non-trainable Params:  {total_params - trainable_params:,}")
    print(f"\nModel Size (MB):       {total_params * 4 / (1024**2):.2f} MB (FP32)")
    print(f"Model Size (MB):       {total_params * 2 / (1024**2):.2f} MB (FP16)")
    
    # Calculate FLOPs
    print(f"\n" + "="*80)
    print("Computing FLOPs...")
    print("="*80)
    
    try:
        from thop import profile, clever_format
        
        model = model.to(device)
        model.eval()
        
        rgb_input = torch.randn(1, 3, input_size, input_size).to(device)
        texture_input = torch.randn(1, 1, input_size, input_size).to(device)
        
        flops, params = profile(model, inputs=(rgb_input, texture_input), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        
        print(f"FLOPs:  {flops_str}")
        print(f"GFLOPs: {flops / 1e9:.3f}")
        print(f"Params: {params_str}")
        
    except ImportError:
        print("⚠ THOP not installed. Install with: pip install thop")
        print("Skipping FLOPS calculation...")
    
    # Layer-wise breakdown
    print(f"\n" + "="*80)
    print("Layer-wise Parameter Count")
    print("="*80)
    
    print(f"\n{'Layer Name':<50} {'Parameters':>15} {'Shape':>20}")
    print("-"*85)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<50} {param.numel():>15,} {str(tuple(param.shape)):>20}")
    
    # Component-wise breakdown
    print(f"\n" + "="*80)
    print("Component-wise Parameter Distribution")
    print("="*80)
    
    components = {
        'RGB Stream Initial': 0,
        'Texture Stream Initial': 0,
        'ResBlock-1 (RGB)': 0,
        'ResBlock-1 (Texture)': 0,
        'ResBlock-2 (RGB)': 0,
        'ResBlock-2 (Texture)': 0,
        'TGVA Attention': 0,
        'ResBlock-3': 0,
        'ResBlock-4': 0,
        'Classification Head': 0
    }
    
    for name, param in model.named_parameters():
        if 'rgb_conv1' in name or 'rgb_bn1' in name:
            components['RGB Stream Initial'] += param.numel()
        elif 'texture_conv1' in name or 'texture_bn1' in name:
            components['Texture Stream Initial'] += param.numel()
        elif 'rgb_layer1' in name:
            components['ResBlock-1 (RGB)'] += param.numel()
        elif 'texture_layer1' in name:
            components['ResBlock-1 (Texture)'] += param.numel()
        elif 'rgb_layer2' in name:
            components['ResBlock-2 (RGB)'] += param.numel()
        elif 'texture_layer2' in name:
            components['ResBlock-2 (Texture)'] += param.numel()
        elif 'tgva_block' in name:
            components['TGVA Attention'] += param.numel()
        elif 'layer3' in name:
            components['ResBlock-3'] += param.numel()
        elif 'layer4' in name:
            components['ResBlock-4'] += param.numel()
        elif 'fc' in name or 'classifier' in name:
            components['Classification Head'] += param.numel()
    
    for component, params in components.items():
        percentage = (params / total_params) * 100
        print(f"{component:<30} {params:>12,} ({percentage:>5.2f}%)")
    
    # Output shapes
    print(f"\n" + "="*80)
    print("Layer Output Shapes (Forward Pass)")
    print("="*80)
    
    try:
        print_layer_shapes(model, 
                          input_size=(1, 3, input_size, input_size),
                          texture_size=(1, 1, input_size, input_size),
                          device=device)
    except Exception as e:
        print(f"Could not compute layer shapes: {e}")
    
    print(f"\n" + "="*80)
    print("Summary Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
