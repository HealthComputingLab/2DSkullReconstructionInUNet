# 📝 Inference Scripts Documentation

This document provides detailed information about the inference and training scripts added to the project.

---

## 📁 New Files Added

### 1. `train_model.py` - Complete Training Script
Advanced training script with:
- ✅ Automatic checkpointing
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Progress visualization
- ✅ Training history logging
- ✅ Resume training capability

### 2. `inference.py` - Production Inference Script
Flexible inference tool for:
- ✅ Single image inference
- ✅ Batch processing (entire directories)
- ✅ Multiple masking strategies
- ✅ Comprehensive visualization
- ✅ Custom mask positions

### 3. `analyze_training.py` - Training Analysis Tool
Visualization script that creates:
- ✅ Loss curves
- ✅ PSNR plots
- ✅ Dice score tracking
- ✅ Learning rate schedule
- ✅ Overfitting indicators
- ✅ Statistical summary

### 4. `TRAINING_GUIDE.md` - Comprehensive Documentation
Complete guide covering:
- ✅ Environment setup
- ✅ Dataset preparation
- ✅ Training process
- ✅ Model evaluation
- ✅ Inference usage
- ✅ Troubleshooting

### 5. `QUICK_START.md` - 5-Minute Quick Start
Fast-track guide for:
- ✅ Quick setup verification
- ✅ Basic training
- ✅ Simple inference
- ✅ Common issues

---

## 🚀 Usage Examples

### Training Examples

#### Basic Training
```powershell
python train_model.py
```

#### Custom Training
```powershell
# Train for 100 epochs with custom learning rate
python train_model.py --epochs 100 --lr 0.001

# Save to custom directory
python train_model.py --save_dir ./my_models

# Frequent checkpointing
python train_model.py --save_every 5

# Early stopping with custom patience
python train_model.py --patience 15

# Visualize samples more frequently
python train_model.py --visualize_every 5
```

#### Resume Training
```powershell
# Resume from checkpoint
python train_model.py --resume ./checkpoints/checkpoint_epoch_50.pth --epochs 100

# Resume from best model
python train_model.py --resume ./checkpoints/best_model.pth --epochs 50
```

#### Full Example
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

---

### Inference Examples

#### Single Image
```powershell
# Basic inference
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png

# With visualization
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --visualize

# Custom mask position
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --mask_type custom --mask_position 100 80

# Different mask size
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --mask_size 80 80
```

#### Batch Processing
```powershell
# Process directory
python inference.py --model checkpoints/best_model.pth --input_dir ./test_images/ --output_dir ./results/

# With visualizations
python inference.py --model checkpoints/best_model.pth --input_dir ./test_images/ --output_dir ./results/ --visualize

# Random masking
python inference.py --model checkpoints/best_model.pth --input_dir ./images/ --output_dir ./results/ --mask_type random
```

---

### Analysis Examples

```powershell
# Basic analysis
python analyze_training.py

# Custom paths
python analyze_training.py --history_path ./checkpoints/training_history.json --output_path ./my_report.png

# Simple plot
python analyze_training.py --simple
```

---

## 📊 Output Structure

After training and inference, your directory will look like:

```
2DSkullReconstructionInUNet/
├── checkpoints/                         # Training outputs
│   ├── best_model.pth                  # Best model (USE THIS for inference!)
│   ├── checkpoint_epoch_10.pth         # Full checkpoint with optimizer
│   ├── checkpoint_epoch_20.pth
│   ├── model_epoch_10.pth              # Model weights only
│   ├── model_epoch_20.pth
│   ├── training_history.json           # Metrics for all epochs
│   └── samples/                        # Sample predictions
│       ├── epoch_10_sample_0.png
│       ├── epoch_10_sample_1.png
│       └── ...
│
├── inference_results/                   # Inference outputs
│   ├── image1_inpainted.png
│   ├── image1_visualization.png
│   ├── image2_inpainted.png
│   └── ...
│
├── training_analysis.png                # Training visualization
├── train_model.py                       # Training script
├── inference.py                         # Inference script
├── analyze_training.py                  # Analysis script
├── TRAINING_GUIDE.md                    # Detailed guide
├── QUICK_START.md                       # Quick start guide
└── INFERENCE_SCRIPTS_README.md          # This file
```

---

## 🎯 Checkpoint Files Explained

### `best_model.pth`
- **What**: Complete checkpoint of the best performing model
- **Contains**: Model weights, optimizer state, epoch number, validation metrics
- **Use for**: Inference, resuming training
- **Recommended**: ⭐ Use this for production inference

### `checkpoint_epoch_N.pth`
- **What**: Full training checkpoint at epoch N
- **Contains**: Everything needed to resume training
- **Use for**: Resuming interrupted training

### `model_epoch_N.pth`
- **What**: Model weights only (smaller file)
- **Contains**: Just the model state dict
- **Use for**: Inference only (cannot resume training)

### `training_history.json`
- **What**: All training metrics
- **Contains**: Loss, PSNR, Dice, learning rates for each epoch
- **Use for**: Analyzing training progress, creating plots

---

## 🔧 Command-Line Arguments Reference

### train_model.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 250 | Number of training epochs |
| `--lr` | float | 0.0005 | Learning rate |
| `--batch_size` | int | 8 | Batch size |
| `--save_dir` | str | ./checkpoints | Checkpoint save directory |
| `--save_every` | int | 10 | Save checkpoint every N epochs |
| `--resume` | str | None | Path to checkpoint to resume from |
| `--patience` | int | 20 | Early stopping patience |
| `--visualize_every` | int | 10 | Generate samples every N epochs |

### inference.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | **required** | Path to model checkpoint |
| `--input` | str | None | Input image path (single image) |
| `--output` | str | None | Output path (single image) |
| `--input_dir` | str | None | Input directory (batch) |
| `--output_dir` | str | None | Output directory (batch) |
| `--mask_type` | str | center | Mask type: center, random, custom |
| `--mask_size` | int[2] | 50 50 | Mask size (height width) |
| `--mask_position` | int[2] | None | Custom position (x y) |
| `--visualize` | flag | False | Create visualization |
| `--device` | str | auto | Device: cuda or cpu |

### analyze_training.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--history_path` | str | ./checkpoints/training_history.json | Training history file |
| `--output_path` | str | ./training_analysis.png | Output plot path |
| `--simple` | flag | False | Create simple 2x2 plot |

---

## 🎓 Training Workflow

### Complete Training Pipeline

```powershell
# Step 1: Verify environment
conda activate 2dskull
python run_smoke.py

# Step 2: Start training
python train_model.py --epochs 250 --save_dir ./checkpoints

# Step 3: Monitor training (in another terminal)
# Check ./checkpoints/samples/ for sample predictions

# Step 4: Analyze results after training
python analyze_training.py

# Step 5: Test model on validation set
python inference.py --model checkpoints/best_model.pth --input_dir dataset/val/ --output_dir ./val_results/ --visualize

# Step 6: Test on new images
python inference.py --model checkpoints/best_model.pth --input my_image.jpg --output result.png --visualize
```

---

## 💡 Tips and Best Practices

### Training Tips

1. **Monitor the Dice Score**: This is your primary metric
   - Target: 0.974
   - Good: >0.90
   - Acceptable: >0.85

2. **Watch for Overfitting**:
   - Train loss << Val loss = Overfitting
   - Solutions: Add more data, early stopping, reduce model capacity

3. **Use Early Stopping**:
   - Don't let it train forever
   - `--patience 20` is a good default

4. **Save Frequently**:
   - Use `--save_every 5` for experiments
   - Disk space is cheap, lost training time is expensive

5. **Visualize Progress**:
   - Check `./checkpoints/samples/` regularly
   - Use `analyze_training.py` during training

### Inference Tips

1. **Always Use `best_model.pth`**:
   - It has the best validation performance
   - Not necessarily the last epoch

2. **Use `--visualize` for Analysis**:
   - Shows input, prediction, ground truth, mask, difference
   - Helps debug poor predictions

3. **Batch Processing for Efficiency**:
   - Faster than processing images one by one
   - Use `--input_dir` instead of multiple `--input` calls

4. **Experiment with Mask Types**:
   - `center`: Consistent testing
   - `random`: Test generalization
   - `custom`: Target specific regions

### Performance Optimization

1. **GPU Training**:
   - 10-30× faster than CPU
   - Worth using if available

2. **Batch Size**:
   - Larger = faster training (if fits in memory)
   - Smaller = more stable training
   - Default: 8 (good balance)

3. **Learning Rate**:
   - Too high: Unstable, loss oscillates
   - Too low: Slow convergence
   - Default: 0.0005 (good starting point)

---

## 🐛 Common Issues and Solutions

### Issue: Training not improving after many epochs

**Check**:
```powershell
python analyze_training.py
```

**Solutions**:
- Learning rate too low: `--lr 0.001`
- Increase perceptual loss: Edit `config.py`, set `perceptual_loss_rate = 0.1`
- Add more training data

### Issue: Out of memory

**Solutions**:
```powershell
# Reduce batch size
python train_model.py --batch_size 4

# Or even smaller
python train_model.py --batch_size 2
```

### Issue: Inference results are poor

**Check**:
1. Model Dice score: Should be >0.85
2. Input image format: Must be grayscale or convertible to grayscale
3. Model checkpoint: Use `best_model.pth`, not random epoch

**Solutions**:
- Train longer if Dice is low
- Verify input image: `python -c "from PIL import Image; img = Image.open('test.jpg'); print(img.mode, img.size)"`
- Test with validation images first

### Issue: Visualizations not appearing

**Cause**: Running on remote server without display

**Solution**: Visualizations are automatically saved to files, check:
- Training samples: `./checkpoints/samples/`
- Inference results: `./inference_results/` or specified `--output_dir`
- Training analysis: `./training_analysis.png`

---

## 📚 Additional Resources

- **Main README**: [README.md](README.md) - Project overview
- **Training Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Comprehensive training documentation
- **Quick Start**: [QUICK_START.md](QUICK_START.md) - Get started in 5 minutes
- **Config File**: [config.py](config.py) - All hyperparameters
- **Original Training**: [main_run_scan_rebuild.py](main_run_scan_rebuild.py) - Original training script

---

## 🎉 Summary

You now have:
- ✅ Professional training script with checkpointing
- ✅ Flexible inference for single/batch images
- ✅ Comprehensive analysis tools
- ✅ Complete documentation

**Ready to train?**
```powershell
conda activate 2dskull
python train_model.py
```

**Questions?** Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed help!
