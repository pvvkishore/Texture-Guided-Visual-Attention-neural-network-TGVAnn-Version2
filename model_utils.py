"""
Utility functions for TGVAnn model
Includes parameter counting, FLOPS calculation, and model summary
"""

import torch
import torch.nn as nn
from thop import profile, clever_format


def count_parameters(model):
    """
    Count total and trainable parameters in the model
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def calculate_flops(model, input_size=(1, 3, 256, 256), texture_size=(1, 1, 256, 256), device='cuda'):
    """
    Calculate FLOPs and parameters for TGVAnn model
    
    Args:
        model: TGVAnn model
        input_size: RGB input size (B, C, H, W)
        texture_size: Texture input size (B, C, H, W)
        device: Device to run calculation on
    
    Returns:
        flops: Total FLOPs
        params: Total parameters
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    rgb_input = torch.randn(input_size).to(device)
    texture_input = torch.randn(texture_size).to(device)
    
    # Calculate FLOPs
    flops, params = profile(model, inputs=(rgb_input, texture_input), verbose=False)
    
    # Format numbers
    flops, params = clever_format([flops, params], "%.3f")
    
    return flops, params


def model_summary(model, input_size=(1, 3, 256, 256), texture_size=(1, 1, 256, 256), device='cuda'):
    """
    Print comprehensive model summary
    
    Args:
        model: TGVAnn model
        input_size: RGB input size
        texture_size: Texture input size
        device: Device to run on
    """
    print("=" * 80)
    print("TGVAnn Model Summary")
    print("=" * 80)
    
    # Parameter count
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Calculate FLOPs
    try:
        flops, params = calculate_flops(model, input_size, texture_size, device)
        print(f"\nFLOPs: {flops}")
        print(f"Params (from THOP): {params}")
    except Exception as e:
        print(f"\nFLOPs calculation failed: {e}")
        print("Please install 'thop' package: pip install thop")
    
    # Model structure
    print("\n" + "=" * 80)
    print("Model Architecture:")
    print("=" * 80)
    print(model)
    
    print("\n" + "=" * 80)


def get_layer_output_shapes(model, input_size=(1, 3, 256, 256), texture_size=(1, 1, 256, 256), device='cuda'):
    """
    Get output shapes of each layer in the model
    
    Args:
        model: TGVAnn model
        input_size: RGB input size
        texture_size: Texture input size
        device: Device to run on
    
    Returns:
        layer_shapes: Dictionary of layer names and output shapes
    """
    model = model.to(device)
    model.eval()
    
    layer_shapes = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                layer_shapes[name] = output[0].shape
            else:
                layer_shapes[name] = output.shape
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    rgb_input = torch.randn(input_size).to(device)
    texture_input = torch.randn(texture_size).to(device)
    
    with torch.no_grad():
        _ = model(rgb_input, texture_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return layer_shapes


def print_layer_shapes(model, input_size=(1, 3, 256, 256), texture_size=(1, 1, 256, 256), device='cuda'):
    """
    Print output shapes of each layer
    """
    layer_shapes = get_layer_output_shapes(model, input_size, texture_size, device)
    
    print("\n" + "=" * 80)
    print("Layer Output Shapes:")
    print("=" * 80)
    
    for name, shape in layer_shapes.items():
        print(f"{name:50s} : {str(tuple(shape)):30s}")
    
    print("=" * 80)


def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint
    
    Args:
        model: TGVAnn model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_model_checkpoint(model, optimizer, filepath, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        model: TGVAnn model
        optimizer: Optimizer
        filepath: Path to checkpoint
        device: Device to load on
    
    Returns:
        model, optimizer, epoch, loss, accuracy
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint.get('accuracy', 0.0)
    
    print(f"Checkpoint loaded from epoch {epoch}")
    print(f"Previous loss: {loss:.4f}, accuracy: {accuracy:.4f}")
    
    return model, optimizer, epoch, loss, accuracy
