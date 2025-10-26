# 🧠 2D Skull Reconstruction with UNet - Complete Training & Inference Guide

## 📌 Overview

This document provides a **complete, step-by-step guide** for training a UNet model to reconstruct masked/missing regions in skull images, and then using the trained model for inference on new images.

---

## 🎯 What You'll Learn

1. ✅ How to prepare your environment
2. ✅ How to train the model from scratch
3. ✅ How to monitor training progress
4. ✅ How to evaluate model performance
5. ✅ How to use the trained model for inference
6. ✅ How to interpret and improve results

---

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Windows 10/11
- ✅ Anaconda/Miniconda installed
- ✅ Conda environment `2dskull` created
- ✅ All dependencies installed (see requirements.txt)
- ✅ Dataset images in `dataset/train/` and `dataset/val/`

---

## 🚀 Step-by-Step Training Guide

### Step 1: Activate the Conda Environment

**ALWAYS** activate the `2dskull` environment before any operation:

```powershell
conda activate 2dskull
```

To verify activation:
```powershell
conda info --envs
# Look for an asterisk (*) next to 2dskull
```

### Step 2: Navigate to Project Directory

```powershell
cd d:\Github\2DSkullReconstructionInUNet
```

### Step 3: Verify Environment Setup

Run the smoke test to check everything is configured correctly:

```powershell
python run_smoke.py
```

**Expected output**: All checks should pass ✅

If any check fails, install missing dependencies:
```powershell
pip install -r requirements.txt
```

### Step 4: Understand Your Dataset

Current dataset:
- **Training images**: 21 images in `dataset/train/`
- **Validation images**: 12 images in `dataset/val/`
- **Format**: JPG grayscale skull images
- **Processing**: Automatically resized to 256×256

**Note**: More training data (100+ images) will improve model performance. Add more images to `dataset/train/` if available.

### Step 5: Start Training

#### Option A: Quick Test Training (5-10 epochs)

For testing that everything works:

```powershell
python train_model.py --epochs 10 --save_dir ./test_checkpoints --visualize_every 5
```

**Time**: 
- CPU: ~5-10 minutes per epoch (50-100 minutes total)
- GPU: ~10-30 seconds per epoch (2-5 minutes total)

#### Option B: Full Training (Production)

For training a production-ready model:

```powershell
python train_model.py --epochs 250 --save_dir ./checkpoints --save_every 10 --patience 20 --visualize_every 10
```

**Time**:
- CPU: ~5-10 minutes per epoch (20-40 hours total)
- GPU: ~10-30 seconds per epoch (1-2 hours total)

**Recommendation**: Run full training on a machine with NVIDIA GPU for best results.

### Step 6: Understanding Training Output

During training, you'll see:

```
🖥️  Device: cuda:0 (or cpu)
   GPU: NVIDIA GeForce RTX 3080

📂 Loading dataset...
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
   Target Dice score: 0.974

🚀 Starting training...

[Train] Epoch   1 | Batch    50/3 | Avg PSNR: 23.45
[Validation] Dice: 0.8234 | Loss: 0.1234 | PSNR: 24.56 dB
================================================================================
📊 Epoch 1/250 Summary:
   Time: 15.23s | LR: 0.000500
   Train - Loss: 0.1345 | PSNR: 23.45 dB
   Val   - Loss: 0.1234 | PSNR: 24.56 dB | Dice: 0.8234
   🌟 New best model!
   Patience: 0/20
💾 Checkpoint saved: checkpoints\checkpoint_epoch_1.pth
🏆 Best model saved: checkpoints\best_model.pth (Dice: 0.8234)
================================================================================
```

**Key Metrics to Watch**:

1. **Dice Score** (Most Important)
   - Measures overlap between prediction and ground truth
   - Range: 0 (worst) to 1 (perfect)
   - **Target**: 0.974 (97.4% overlap)
   - **Good**: >0.90
   - **Acceptable**: >0.85

2. **PSNR** (Peak Signal-to-Noise Ratio)
   - Measures reconstruction quality
   - Range: typically 20-40 dB
   - **Good**: >30 dB
   - **Acceptable**: >25 dB

3. **Loss**
   - Combined BCE + Perceptual loss
   - Should **decrease** over epochs
   - Both training and validation loss should go down

### Step 7: Monitor Training Progress

#### Check Sample Predictions

During training, sample predictions are automatically saved every N epochs:

```
checkpoints/samples/
├── epoch_10_sample_0.png
├── epoch_10_sample_1.png
├── epoch_20_sample_0.png
└── ...
```

**Open these images** to visually verify the model is learning to reconstruct masked regions.

#### Check Training History

After training completes (or during), analyze metrics:

```powershell
python analyze_training.py
```

This creates `training_analysis.png` showing:
- Loss curves (train vs validation)
- PSNR progression
- Dice score progression
- Learning rate schedule
- Overfitting indicators
- Statistical summary

### Step 8: Training Complete!

Training stops when:
1. ✅ Target Dice score (0.974) is reached, OR
2. ✅ Early stopping triggers (no improvement for 20 epochs), OR
3. ✅ Maximum epochs (250) is reached

**Final outputs** in `checkpoints/`:
- `best_model.pth` ← **USE THIS for inference!**
- `checkpoint_epoch_N.pth` (every 10 epochs)
- `training_history.json` (all metrics)
- `samples/` (sample predictions)

---

## 🎯 Step-by-Step Inference Guide

### Step 9: Test the Trained Model

#### Test on a Single Image

```powershell
python inference.py --model checkpoints/best_model.pth --input dataset/val/Topview14.jpg --output result.png --visualize
```

**Output**:
- `result.png` - Inpainted image (masked region reconstructed)
- `result_visualization.png` - Comparison showing:
  - Original image
  - Masked input
  - Mask region
  - Model prediction
  - Reconstructed image
  - Difference map

#### Test on Multiple Images (Batch)

Process all validation images:

```powershell
python inference.py --model checkpoints/best_model.pth --input_dir dataset/val/ --output_dir ./inference_results/ --visualize
```

**Output** in `inference_results/`:
- `Topview14_inpainted.jpg`
- `Topview14_visualization.png`
- `Topview15_inpainted.jpg`
- `Topview15_visualization.png`
- ... (for all images)

#### Test with Different Mask Types

**Center mask** (default):
```powershell
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --mask_type center
```

**Random mask position**:
```powershell
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --mask_type random
```

**Custom mask position** (x=100, y=80):
```powershell
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --mask_type custom --mask_position 100 80
```

**Larger mask** (80×80 pixels):
```powershell
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --mask_size 80 80
```

### Step 10: Use Model on New Images

To use the trained model on your own skull images:

1. **Place images** in a directory (e.g., `my_images/`)
2. **Run inference**:
   ```powershell
   python inference.py --model checkpoints/best_model.pth --input_dir ./my_images/ --output_dir ./my_results/ --visualize
   ```
3. **Check results** in `my_results/`

**Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

---

## 📊 Understanding Results

### What to Expect

After successful training (Dice ~0.90-0.97):

✅ **Good reconstruction**: Masked regions are filled with realistic skull texture  
✅ **Smooth transitions**: No visible seams between original and reconstructed  
✅ **Anatomically plausible**: Reconstructed regions follow skull structure  

### Interpreting Visualization

The visualization shows 6 panels:

1. **Original Image**: Ground truth before masking
2. **Masked Input**: Image with black rectangular mask
3. **Mask Region**: Binary mask showing where to reconstruct
4. **Model Prediction**: Full image prediction from the model
5. **Reconstructed Image**: Original with masked region replaced by prediction
6. **Difference Map**: Shows where model differs from ground truth
   - Dark areas: Good match
   - Bright areas: Larger differences

### Quality Indicators

**Good Model**:
- Dice > 0.90
- PSNR > 30 dB
- Difference map is mostly dark
- Reconstructed regions blend seamlessly

**Poor Model** (needs more training):
- Dice < 0.85
- PSNR < 25 dB
- Visible artifacts or blurriness
- Clear boundaries at mask edges

---

## 🔧 Advanced: Improving Model Performance

### If Dice Score is Low (<0.85)

**Solution 1: Train Longer**
```powershell
python train_model.py --resume checkpoints/best_model.pth --epochs 100
```

**Solution 2: Reduce Learning Rate**
```powershell
python train_model.py --resume checkpoints/best_model.pth --epochs 100 --lr 0.0001
```

**Solution 3: Add More Training Data**
- Add more images to `dataset/train/`
- Recommended: 100+ images for best results

**Solution 4: Increase Perceptual Loss**
Edit `config.py`:
```python
perceptual_loss_rate = 0.1  # instead of 0.05
```

### If Training is Too Slow (CPU)

**Option 1: Reduce Batch Size** (faster per epoch)
```powershell
python train_model.py --batch_size 4
```

**Option 2: Use GPU Machine**
- Training is 10-30× faster on GPU
- Recommended for serious training

**Option 3: Reduce Image Size**
Edit `config.py`:
```python
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
```

### If Out of Memory

**Reduce batch size**:
```powershell
python train_model.py --batch_size 2
```

or edit `config.py`:
```python
BATCH_SIZE = 2  # or even 1
```

---

## 📁 File Organization

After training and inference:

```
d:\Github\2DSkullReconstructionInUNet\
│
├── checkpoints/                      # Training outputs
│   ├── best_model.pth               # ⭐ Best model (use for inference)
│   ├── checkpoint_epoch_10.pth      # Checkpoints every 10 epochs
│   ├── checkpoint_epoch_20.pth
│   ├── training_history.json        # All metrics
│   └── samples/                     # Sample predictions during training
│
├── inference_results/               # Inference outputs
│   ├── Topview14_inpainted.jpg
│   ├── Topview14_visualization.png
│   └── ...
│
├── dataset/
│   ├── train/                       # Training images (21 images)
│   └── val/                         # Validation images (12 images)
│
├── train_model.py                   # Training script
├── inference.py                     # Inference script
├── analyze_training.py              # Analysis/visualization script
│
├── TRAINING_GUIDE.md               # Comprehensive guide
├── QUICK_START.md                  # 5-minute quick start
├── INFERENCE_SCRIPTS_README.md     # Scripts documentation
└── COMPLETE_TUTORIAL.md            # This file
```

---

## 🎓 Complete Workflow Summary

### Training Workflow

```powershell
# 1. Activate environment
conda activate 2dskull

# 2. Navigate to project
cd d:\Github\2DSkullReconstructionInUNet

# 3. Verify setup
python run_smoke.py

# 4. Train model
python train_model.py --epochs 250 --save_dir ./checkpoints

# 5. Analyze results
python analyze_training.py

# 6. Test on validation set
python inference.py --model checkpoints/best_model.pth --input_dir dataset/val/ --output_dir ./val_results/ --visualize
```

### Inference Workflow

```powershell
# 1. Activate environment (if not already)
conda activate 2dskull

# 2. Single image inference
python inference.py --model checkpoints/best_model.pth --input my_image.jpg --output result.png --visualize

# 3. Batch inference
python inference.py --model checkpoints/best_model.pth --input_dir ./my_images/ --output_dir ./results/ --visualize
```

---

## ❓ Troubleshooting

### "conda: command not found"
**Solution**: Conda is not in PATH. Use Anaconda Prompt or add conda to PATH.

### "Import torch could not be resolved"
**Solution**:
```powershell
conda activate 2dskull
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "No images found in dataset"
**Solution**: Check paths in `config.py` match your actual dataset location.

### "CUDA out of memory"
**Solution**: Reduce batch size:
```powershell
python train_model.py --batch_size 4  # or 2
```

### Training loss not decreasing
**Solutions**:
1. Check learning rate (try 0.001 or 0.0001)
2. Verify dataset is correct
3. Check data normalization
4. Increase perceptual loss weight

### Model predictions are all black/white
**Solutions**:
1. Restart with lower learning rate: `--lr 0.0001`
2. Check if dataset loaded correctly
3. Verify model checkpoint is not corrupted

---

## 📚 Additional Documentation

- **TRAINING_GUIDE.md**: Comprehensive training documentation
- **QUICK_START.md**: Get started in 5 minutes
- **INFERENCE_SCRIPTS_README.md**: Detailed scripts documentation
- **README.md**: Project overview

---

## ✅ Checklist for Success

### Before Training
- [ ] Activated `2dskull` conda environment
- [ ] Verified all imports work (run `python run_smoke.py`)
- [ ] Dataset in correct directories (`dataset/train/`, `dataset/val/`)
- [ ] Understood key metrics (Dice, PSNR, Loss)

### During Training
- [ ] Monitor console output for progress
- [ ] Check sample predictions in `checkpoints/samples/`
- [ ] Watch for Dice score improvements
- [ ] Ensure loss is decreasing

### After Training
- [ ] Analyze results with `analyze_training.py`
- [ ] Check best Dice score (should be >0.85)
- [ ] Test inference on validation set
- [ ] Verify reconstructions look good

### For Inference
- [ ] Use `best_model.pth` (not random epoch)
- [ ] Test on validation images first
- [ ] Use `--visualize` to verify quality
- [ ] Check difference maps for quality assessment

---

## 🎉 Congratulations!

You now have:
- ✅ A trained UNet model for skull image inpainting
- ✅ Tools for inference on new images
- ✅ Analysis capabilities for model evaluation
- ✅ Complete understanding of the training process

**Ready to train?**

```powershell
conda activate 2dskull
cd d:\Github\2DSkullReconstructionInUNet
python train_model.py
```

---

## 🆘 Need Help?

1. **Check this guide** for step-by-step instructions
2. **Review TRAINING_GUIDE.md** for detailed explanations
3. **Check console output** for error messages
4. **Verify environment**: `conda activate 2dskull`
5. **Test imports**: `python run_smoke.py`

---

**Happy Training! 🚀🧠**
