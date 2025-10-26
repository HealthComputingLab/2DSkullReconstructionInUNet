# 🎓 Complete Training Guide for UNet Skull Image Inpainting

This comprehensive guide will walk you through the entire process of training and using the UNet model for skull image inpainting and reconstruction.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training the Model](#training-the-model)
5. [Model Evaluation](#model-evaluation)
6. [Inference on New Images](#inference-on-new-images)
7. [Advanced Topics](#advanced-topics)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- **Minimum**: 
  - CPU: Intel i5 or equivalent
  - RAM: 8 GB
  - Disk: 5 GB free space
  
- **Recommended for faster training**:
  - GPU: NVIDIA GPU with CUDA support (4GB+ VRAM)
  - RAM: 16 GB+
  - SSD storage

### Software Requirements

- Windows 10/11
- Anaconda or Miniconda installed
- Git (optional, for version control)

---

## Environment Setup

### Step 1: Activate the Conda Environment

The project uses a dedicated conda environment called `2dskull`. Always activate it before working:

```powershell
conda activate 2dskull
```

### Step 2: Verify Installation

Check that all required packages are installed:

```powershell
# Check Python version (should be 3.8+)
python --version

# Check PyTorch installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Check if CUDA is available for GPU training
python -c "import torch; print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Expected output:
```
PyTorch version: 2.9.0+cpu (or +cu124 if CUDA is available)
CUDA available: True (if you have NVIDIA GPU) or False (CPU only)
```

### Step 3: Install Missing Dependencies (if needed)

If any packages are missing:

```powershell
pip install -r requirements.txt
```

For GPU support with CUDA 12.4 (if you have NVIDIA GPU):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only (if no NVIDIA GPU):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Dataset Preparation

### Current Dataset Structure

Your dataset is already organized correctly:

```
dataset/
├── train/           # 21 images (Topview0.jpg to Topview44.jpg)
│   ├── Topview0.jpg
│   ├── Topview1.jpg
│   └── ...
└── val/             # 12 images (Topview14.jpg to Topview30.jpg)
    ├── Topview14.jpg
    ├── Topview15.jpg
    └── ...
```

**Total**: 21 training images + 12 validation images

### Data Augmentation

The training pipeline automatically applies:
- Random horizontal flips
- Adaptive masking (targets bright regions)
- Normalization to [-1, 1] range

### Adding More Data (Optional)

To improve model performance, consider adding more images:

1. Place additional skull images in `dataset/train/` for training
2. Place validation images in `dataset/val/`
3. Supported formats: `.jpg`, `.jpeg`, `.png`
4. Images will be automatically resized to 256×256 pixels

**Recommended**: 100+ training images for better generalization

---

## Training the Model

### Understanding Training Parameters

The training configuration is defined in `config.py`:

```python
TARGET_DICE_SCORE = 0.974        # Training stops when this Dice score is reached
MAX_EPOCHS_PER_ROUND = 250       # Maximum epochs per training round
IMAGE_WIDTH = 256                # Input image width
IMAGE_HEIGHT = 256               # Input image height
BATCH_SIZE = 8                   # Number of images per batch
LEARNING_RATE = 0.0005           # Initial learning rate
CUT_SIZE = (50, 50)             # Size of masked region (height, width)
perceptual_loss_rate = 0.05     # Weight for perceptual loss
```

### Basic Training Command

Navigate to the project directory and run:

```powershell
cd d:\Github\2DSkullReconstructionInUNet
conda activate 2dskull
python train_model.py
```

### Training with Custom Parameters

```powershell
# Train for 100 epochs with custom learning rate
python train_model.py --epochs 100 --lr 0.001

# Save checkpoints to a specific directory
python train_model.py --save_dir ./my_models

# Save checkpoints more frequently (every 5 epochs)
python train_model.py --save_every 5

# Train with early stopping (patience of 15 epochs)
python train_model.py --patience 15

# Visualize samples every 5 epochs
python train_model.py --visualize_every 5
```

### Full Training Example

Complete training command with all options:

```powershell
python train_model.py `
    --epochs 250 `
    --lr 0.0005 `
    --batch_size 8 `
    --save_dir ./checkpoints `
    --save_every 10 `
    --patience 20 `
    --visualize_every 10
```

### Resuming Training from Checkpoint

If training is interrupted, resume from the last checkpoint:

```powershell
python train_model.py --resume ./checkpoints/checkpoint_epoch_50.pth --epochs 100
```

Or resume from the best model:

```powershell
python train_model.py --resume ./checkpoints/best_model.pth --epochs 100
```

### Understanding Training Output

During training, you'll see output like this:

```
🖥️  Device: cuda:0
   GPU: NVIDIA GeForce RTX 3080
   Memory: 10.0 GB

📂 Loading dataset...
   Train: d:\Github\2DSkullReconstructionInUNet\dataset\train
   Val: d:\Github\2DSkullReconstructionInUNet\dataset\val

✅ Dataset loaded:
   Train images: 21
   Val images: 12
   Batch size: 8

🏗️  Building model...
   Total parameters: 31,037,633
   Trainable parameters: 31,037,633

⚙️  Training configuration:
   Learning rate: 0.0005
   Perceptual loss weight: 0.05
   Max epochs: 250
   Early stopping patience: 20
   Target Dice score: 0.974

🚀 Starting training...

[Train] Epoch   1 | Batch    50/3 | Avg PSNR: 23.45
[Validation] Dice: 0.8234 | Loss: 0.1234 | PSNR: 24.56 dB
================================================================================
📊 Epoch 1/250 Summary:
   Time: 15.23s | LR: 0.000500
   Train - Loss: 0.1345 | PSNR: 23.45 dB
   Val   - Loss: 0.1234 | PSNR: 24.56 dB | Dice: 0.8234
   🌟 New best model! (Previous: 0.0000)
   Patience: 0/20
💾 Checkpoint saved: checkpoints\checkpoint_epoch_1.pth
🏆 Best model saved: checkpoints\best_model.pth (Dice: 0.8234)
================================================================================
```

### Key Metrics Explained

1. **PSNR (Peak Signal-to-Noise Ratio)**: 
   - Measures reconstruction quality
   - Higher is better (typically 20-40 dB)
   - >30 dB indicates good reconstruction

2. **Dice Coefficient**:
   - Measures overlap between prediction and ground truth
   - Range: 0 (no overlap) to 1 (perfect overlap)
   - Target: 0.974 (97.4% overlap)

3. **Loss**:
   - Combined BCE + Perceptual loss
   - Lower is better
   - Should decrease over epochs

### Training Time Estimates

Based on your dataset (21 training + 12 validation images):

- **CPU only**: ~5-10 minutes per epoch
- **GPU (NVIDIA RTX 3080)**: ~10-30 seconds per epoch

Full training (250 epochs):
- **CPU**: 20-40 hours
- **GPU**: 1-2 hours

---

## Model Evaluation

### Saved Outputs

Training automatically saves:

1. **Checkpoints** (`./checkpoints/`):
   - `checkpoint_epoch_N.pth` - Full checkpoint with optimizer state
   - `model_epoch_N.pth` - Model weights only (for inference)
   - `best_model.pth` - Best performing model

2. **Training History** (`./checkpoints/training_history.json`):
   ```json
   {
     "train_loss": [0.1345, 0.1234, ...],
     "train_psnr": [23.45, 24.56, ...],
     "val_loss": [0.1234, 0.1123, ...],
     "val_psnr": [24.56, 25.67, ...],
     "val_dice": [0.8234, 0.8456, ...],
     "learning_rates": [0.0005, 0.0005, ...]
   }
   ```

3. **Sample Predictions** (`./checkpoints/samples/`):
   - Visualization images saved every N epochs
   - Shows: input, prediction, ground truth, mask

### Analyzing Training Results

Create a plotting script to visualize training progress:

```python
import json
import matplotlib.pyplot as plt

# Load training history
with open('./checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss')
axes[0, 0].plot(history['val_loss'], label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: PSNR
axes[0, 1].plot(history['train_psnr'], label='Train PSNR')
axes[0, 1].plot(history['val_psnr'], label='Val PSNR')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('PSNR (dB)')
axes[0, 1].set_title('PSNR over Epochs')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Dice Score
axes[1, 0].plot(history['val_dice'], label='Validation Dice', color='green')
axes[1, 0].axhline(y=0.974, color='r', linestyle='--', label='Target')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Dice Score')
axes[1, 0].set_title('Validation Dice Score')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 4: Learning Rate
axes[1, 1].plot(history['learning_rates'], label='Learning Rate')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('./training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

Save as `analyze_training.py` and run:
```powershell
python analyze_training.py
```

---

## Inference on New Images

### Single Image Inference

Process a single image with the trained model:

```powershell
# Basic inference
python inference.py --model checkpoints/best_model.pth --input test_image.jpg --output result.png

# With visualization
python inference.py --model checkpoints/best_model.pth --input test_image.jpg --output result.png --visualize
```

### Batch Inference (Directory)

Process multiple images at once:

```powershell
# Process all images in a directory
python inference.py --model checkpoints/best_model.pth --input_dir ./test_images/ --output_dir ./results/

# With visualization for each image
python inference.py --model checkpoints/best_model.pth --input_dir ./test_images/ --output_dir ./results/ --visualize
```

### Custom Masking

Control where the mask is applied:

```powershell
# Center mask (default)
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --mask_type center

# Random mask position
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --mask_type random

# Custom mask position (x=100, y=80) and size (60x60)
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --mask_type custom --mask_position 100 80 --mask_size 60 60
```

### Understanding Inference Output

The inference script produces:

1. **Inpainted Image**: Final reconstructed image
2. **Visualization** (if `--visualize` is used):
   - Original image
   - Masked input
   - Mask region
   - Model prediction
   - Reconstructed image (combines original + prediction)
   - Difference map (shows reconstruction quality)

---

## Advanced Topics

### Fine-tuning the Model

Start from a pre-trained model and continue training:

```powershell
# Fine-tune on new data
python train_model.py --resume checkpoints/best_model.pth --epochs 50 --lr 0.0001
```

### Adjusting Hyperparameters

Edit `config.py` to tune performance:

```python
# For larger masked regions
CUT_SIZE = (80, 80)

# For higher quality but slower training
BATCH_SIZE = 4
LEARNING_RATE = 0.0001

# For faster convergence but potentially overfitting
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# To increase perceptual loss influence
perceptual_loss_rate = 0.1
```

### Multi-GPU Training (if available)

Modify `train_model.py` to use DataParallel:

```python
# After model initialization
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

### Export for Production

Convert to ONNX format for deployment:

```python
import torch
from Unet_Architecture.Image_Painting.UnetSkipConnection import Unet

# Load model
model = Unet(1, 1)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 1, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "skull_inpainting.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms**: Training crashes with "CUDA out of memory" or "RuntimeError: out of memory"

**Solutions**:
```powershell
# Reduce batch size
python train_model.py --batch_size 4

# Or edit config.py:
BATCH_SIZE = 4  # or even 2
```

### Issue: Training Loss Not Decreasing

**Possible causes**:
1. Learning rate too high or too low
2. Not enough data
3. Model already converged

**Solutions**:
```powershell
# Try different learning rates
python train_model.py --lr 0.0001  # Lower LR
python train_model.py --lr 0.001   # Higher LR

# Add more training data
# Reduce perceptual loss weight in config.py:
perceptual_loss_rate = 0.01  # instead of 0.05
```

### Issue: Overfitting (High training score, low validation score)

**Solutions**:
1. Add more training data
2. Increase data augmentation
3. Add dropout layers to the model
4. Reduce model capacity

### Issue: "Import torch could not be resolved"

**Solution**:
```powershell
conda activate 2dskull
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Training Too Slow on CPU

**Solutions**:
1. Use a machine with GPU
2. Reduce batch size
3. Reduce image size (edit `config.py`)
4. Use fewer epochs

### Issue: Model Predictions Are All Black or White

**Possible causes**:
1. Learning rate too high
2. Loss function issue
3. Data normalization problem

**Solutions**:
```powershell
# Restart training with lower learning rate
python train_model.py --lr 0.0001

# Check data normalization in Prepare_Dataset.py
```

---

## Quick Reference

### Essential Commands

```powershell
# Activate environment
conda activate 2dskull

# Basic training
python train_model.py

# Training with options
python train_model.py --epochs 100 --lr 0.0005 --save_dir ./models

# Resume training
python train_model.py --resume ./checkpoints/best_model.pth --epochs 50

# Single image inference
python inference.py --model checkpoints/best_model.pth --input image.jpg --output result.png

# Batch inference
python inference.py --model checkpoints/best_model.pth --input_dir ./images/ --output_dir ./results/

# With visualization
python inference.py --model checkpoints/best_model.pth --input image.jpg --output result.png --visualize
```

### File Structure After Training

```
2DSkullReconstructionInUNet/
├── checkpoints/                    # Training outputs
│   ├── best_model.pth             # Best model (use this for inference)
│   ├── checkpoint_epoch_10.pth    # Checkpoint at epoch 10
│   ├── checkpoint_epoch_20.pth    # Checkpoint at epoch 20
│   ├── model_epoch_10.pth         # Model weights only
│   ├── training_history.json      # Training metrics
│   └── samples/                   # Sample predictions
│       ├── epoch_10_sample_0.png
│       └── ...
├── dataset/
│   ├── train/                     # Training images
│   └── val/                       # Validation images
├── config.py                      # Configuration
├── train_model.py                 # Training script
├── inference.py                   # Inference script
└── TRAINING_GUIDE.md             # This file
```

---

## Next Steps

1. ✅ **Start Training**: Run `python train_model.py`
2. ⏳ **Monitor Progress**: Watch the console output and check sample predictions
3. 📊 **Analyze Results**: Use `analyze_training.py` to visualize training curves
4. 🎯 **Test Model**: Run inference on test images
5. 🔧 **Fine-tune**: Adjust hyperparameters if needed
6. 🚀 **Deploy**: Use the best model for your application

---

## Support

For issues or questions:
1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review the console output for error messages
3. Check `log.txt` for detailed logs
4. Verify you're in the correct conda environment: `conda activate 2dskull`

---

**Happy Training! 🎉**
