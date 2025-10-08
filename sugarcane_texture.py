"""
LDP Texture Generation for KSCI Sugarcane Dataset
Generates texture features from RGB images using Local Directional Pattern
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.ldp_texture_generator import process_dataset, verify_dataset_structure


def main():
    print("\n" + "="*80)
    print("LDP Texture Generation - KSCI Sugarcane Dataset")
    print("5 Classes: Healthy, Mosaic, RedRot, Rust, Yellow")
    print("="*80)
    
    # Define paths
    input_folder = "Sugarcane_RGB"
    output_folder = "Sugarcane_Texture"
    
    print(f"\nInput RGB Dataset: {input_folder}")
    print(f"Output Texture Dataset: {output_folder}")
    
    # Verify input dataset
    print("\n" + "-"*80)
    print("Step 1: Verifying Input Dataset")
    print("-"*80)
    verify_dataset_structure(input_folder)
    
    # Generate textures
    print("\n" + "-"*80)
    print("Step 2: Generating LDP Texture Features")
    print("-"*80)
    print("Parameters:")
    print("  - top_k: 3 (top 3 strongest gradients)")
    print("  - CLAHE clip limit: 2.0")
    print("  - Resize: 256×256")
    print("  - Texture type: Single-channel (grayscale)")
    
    process_dataset(
        input_folder=input_folder,
        output_folder=output_folder,
        texture_type='single',
        resize_shape=(256, 256)
    )
    
    # Verify output dataset
    print("\n" + "-"*80)
    print("Step 3: Verifying Generated Texture Dataset")
    print("-"*80)
    verify_dataset_structure(output_folder)
    
    print("\n" + "="*80)
    print("✓ Texture Generation Complete!")
    print("="*80)
    print(f"\nYou can now use the following directories for training:")
    print(f"  RGB Dataset: {input_folder}")
    print(f"  Texture Dataset: {output_folder}")
    print("\nNext steps:")
    print("  1. Train model: python train_sugarcane.py")
    print("  2. Evaluate: python evaluate_sugarcane.py --checkpoint <path>")
    print("  3. Visualize: python visualize_gradcam_sugarcane.py --checkpoint <path>")


if __name__ == "__main__":
    main()
