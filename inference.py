"""
Inference script for UNet Image Inpainting Model

This script loads a trained model and performs inference on new images.
It can process single images or entire directories.

Usage:
    # Single image
    python inference.py --model checkpoints/best_model.pth --input test_image.jpg --output result.png
    
    # Directory of images
    python inference.py --model checkpoints/best_model.pth --input_dir ./test_images/ --output_dir ./results/
    
    # With visualization
    python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --visualize
"""

import argparse
import os
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

import config as cf
from Unet_Architecture.Image_Painting.UnetSkipConnection import Unet


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on
        
    Returns:
        model: Loaded model in evaluation mode
    """
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Initialize model architecture
    in_channels = 1
    num_classes = 1
    model = Unet(in_channels, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_dice = checkpoint.get('val_dice', 'unknown')
        print(f"✅ Loaded checkpoint from epoch {epoch}")
        print(f"   Validation Dice: {val_dice}")
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded model weights")
    
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Load and preprocess image for inference.
    
    Args:
        image_path: Path to input image
        target_size: Target size (width, height) for resizing
        
    Returns:
        image_tensor: Preprocessed image tensor (1, 1, H, W)
        original_image: Original PIL image for reference
    """
    # Load image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    original_image = image.copy()
    
    # Resize to model input size
    image = image.resize(target_size, Image.BILINEAR)
    
    # Convert to tensor and normalize to [-1, 1]
    image_array = np.array(image)
    image_tensor = transforms.functional.to_tensor(image_array)
    image_tensor = image_tensor * 2 - 1  # Scale from [0, 1] to [-1, 1]
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_image


def create_masked_image(image_tensor, mask_type='center', mask_size=(50, 50), mask_position=None):
    """
    Create a masked version of the input image.
    
    Args:
        image_tensor: Input image tensor (1, 1, H, W) in range [-1, 1]
        mask_type: Type of mask ('center', 'random', 'custom')
        mask_size: Size of mask (height, width)
        mask_position: Custom position (x, y) for mask. If None, uses default
        
    Returns:
        masked_tensor: Image with masked region set to 0
        mask_tensor: Binary mask indicating masked region (1 = masked)
    """
    masked_tensor = image_tensor.clone()
    _, _, h, w = image_tensor.shape
    mask_h, mask_w = mask_size
    
    # Create mask tensor
    mask_tensor = torch.zeros_like(image_tensor)
    
    if mask_type == 'center':
        # Center mask
        x = (w - mask_w) // 2
        y = (h - mask_h) // 2
    elif mask_type == 'random':
        # Random position
        x = np.random.randint(0, max(1, w - mask_w))
        y = np.random.randint(0, max(1, h - mask_h))
    elif mask_type == 'custom' and mask_position is not None:
        x, y = mask_position
    else:
        # Default to center
        x = (w - mask_w) // 2
        y = (h - mask_h) // 2
    
    # Apply mask
    masked_tensor[:, :, y:y+mask_h, x:x+mask_w] = -1  # Set to black in [-1, 1] range
    mask_tensor[:, :, y:y+mask_h, x:x+mask_w] = 1
    
    return masked_tensor, mask_tensor


def postprocess_output(output_tensor):
    """
    Convert model output to displayable image.
    
    Args:
        output_tensor: Model output tensor (1, 1, H, W)
        
    Returns:
        output_image: PIL Image
    """
    # Apply sigmoid if not already applied
    output_tensor = torch.sigmoid(output_tensor)
    
    # Convert to numpy and remove batch/channel dimensions
    output_array = output_tensor.squeeze().cpu().detach().numpy()
    
    # Scale to [0, 255]
    output_array = (output_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    output_image = Image.fromarray(output_array, mode='L')
    
    return output_image


def visualize_results(input_tensor, masked_tensor, output_tensor, mask_tensor, save_path=None):
    """
    Create visualization comparing input, masked input, prediction, and mask.
    
    Args:
        input_tensor: Original input image tensor
        masked_tensor: Masked input tensor
        output_tensor: Model prediction tensor
        mask_tensor: Binary mask tensor
        save_path: Path to save visualization (if None, displays instead)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert tensors to numpy for visualization
    # Scale from [-1, 1] to [0, 1] for input/masked
    input_img = ((input_tensor.squeeze().cpu().detach().numpy() + 1) / 2)
    masked_img = ((masked_tensor.squeeze().cpu().detach().numpy() + 1) / 2)
    output_img = torch.sigmoid(output_tensor).squeeze().cpu().detach().numpy()
    mask_img = mask_tensor.squeeze().cpu().detach().numpy()
    
    # Create difference map
    diff = np.abs(input_img - output_img)
    
    # Reconstructed image (masked region from prediction, rest from original)
    reconstructed = input_img.copy()
    reconstructed[mask_img > 0.5] = output_img[mask_img > 0.5]
    
    # Plot images
    axes[0, 0].imshow(input_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(masked_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Masked Input', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Mask Region', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(output_img, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Model Prediction', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(reconstructed, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Reconstructed Image', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title('Difference Map', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def infer_single_image(model, image_path, output_path, device, mask_type='center', 
                       mask_size=(50, 50), visualize=False, mask_position=None):
    """
    Perform inference on a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        output_path: Path to save output
        device: Computation device
        mask_type: Type of mask to apply
        mask_size: Size of mask region
        visualize: Whether to create visualization
        mask_position: Custom mask position (x, y)
    """
    print(f"\n🖼️  Processing: {image_path}")
    
    # Preprocess
    image_tensor, original_image = preprocess_image(image_path, 
                                                     target_size=(cf.IMAGE_WIDTH, cf.IMAGE_HEIGHT))
    
    # Create masked input
    masked_tensor, mask_tensor = create_masked_image(image_tensor, mask_type, mask_size, mask_position)
    
    # Move to device
    masked_tensor = masked_tensor.to(device)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        output_tensor = model(masked_tensor)
    inference_time = time.time() - start_time
    
    print(f"   ⏱️  Inference time: {inference_time*1000:.2f} ms")
    
    # Postprocess
    output_image = postprocess_output(output_tensor)
    
    # Save output
    output_image.save(output_path)
    print(f"   ✅ Saved: {output_path}")
    
    # Create visualization if requested
    if visualize:
        vis_path = str(Path(output_path).parent / f"{Path(output_path).stem}_visualization.png")
        visualize_results(image_tensor, masked_tensor, output_tensor, mask_tensor, vis_path)
    
    return output_image


def infer_directory(model, input_dir, output_dir, device, mask_type='center',
                   mask_size=(50, 50), visualize=False):
    """
    Perform inference on all images in a directory.
    
    Args:
        model: Trained model
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        device: Computation device
        mask_type: Type of mask to apply
        mask_size: Size of mask region
        visualize: Whether to create visualizations
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images in {input_dir}")
    print(f"📁 Outputs will be saved to {output_dir}")
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}]", end=" ")
        output_path = output_dir / f"{image_path.stem}_inpainted{image_path.suffix}"
        
        try:
            infer_single_image(model, str(image_path), str(output_path), device,
                             mask_type, mask_size, visualize)
        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")
            continue
    
    print(f"\n✅ Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(description='Inference for UNet image inpainting')
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    
    # Input/Output
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input image (for single image inference)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image (for single image inference)')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing input images (for batch inference)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save output images (for batch inference)')
    
    # Masking options
    parser.add_argument('--mask_type', type=str, default='center',
                       choices=['center', 'random', 'custom'],
                       help='Type of mask to apply (default: center)')
    parser.add_argument('--mask_size', type=int, nargs=2, default=[50, 50],
                       help='Size of mask as HEIGHT WIDTH (default: 50 50)')
    parser.add_argument('--mask_position', type=int, nargs=2, default=None,
                       help='Custom mask position as X Y (only for --mask_type custom)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization showing all steps')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu). Auto-detected if not specified')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("🧠 UNet Image Inpainting - Inference")
    print("=" * 80)
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load model
    model = load_model(args.model, device)
    
    # Validate arguments
    single_mode = args.input is not None
    batch_mode = args.input_dir is not None
    
    if single_mode and batch_mode:
        print("❌ Error: Cannot specify both --input and --input_dir")
        sys.exit(1)
    
    if not single_mode and not batch_mode:
        print("❌ Error: Must specify either --input or --input_dir")
        sys.exit(1)
    
    if single_mode and args.output is None:
        # Auto-generate output path
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_inpainted{input_path.suffix}")
        print(f"ℹ️  Output path not specified, using: {args.output}")
    
    if batch_mode and args.output_dir is None:
        args.output_dir = './inference_results'
        print(f"ℹ️  Output directory not specified, using: {args.output_dir}")
    
    mask_size = tuple(args.mask_size)
    mask_position = tuple(args.mask_position) if args.mask_position else None
    
    # Run inference
    try:
        if single_mode:
            infer_single_image(model, args.input, args.output, device,
                             args.mask_type, mask_size, args.visualize, mask_position)
        else:
            infer_directory(model, args.input_dir, args.output_dir, device,
                          args.mask_type, mask_size, args.visualize)
        
        print("\n" + "=" * 80)
        print("✅ Inference complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
