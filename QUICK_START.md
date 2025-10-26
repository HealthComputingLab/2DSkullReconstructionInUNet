# 🚀 Quick Start Guide

Get started with training and inference in 5 minutes!

---

## Step 1: Activate Environment

```powershell
conda activate 2dskull
```

## Step 2: Verify Setup

```powershell
cd d:\Github\2DSkullReconstructionInUNet
python run_smoke.py
```

Expected output: All checks should pass ✅

---

## Step 3: Train the Model

### Option A: Quick Training (Test)

```powershell
python train_model.py --epochs 10 --save_dir ./test_checkpoints --visualize_every 5
```

This runs a short 10-epoch training to verify everything works.

### Option B: Full Training

```powershell
python train_model.py --epochs 250 --save_dir ./checkpoints --save_every 10 --patience 20
```

This runs full training until the target Dice score (0.974) is reached or early stopping triggers.

**Training Time**:
- CPU: ~5-10 minutes per epoch (total: 20-40 hours)
- GPU: ~10-30 seconds per epoch (total: 1-2 hours)

---

## Step 4: Monitor Training

Watch the console output for:
- 📊 Epoch summaries (Loss, PSNR, Dice)
- 💾 Checkpoint saves
- 🌟 Best model updates

Sample predictions are saved to: `checkpoints/samples/`

---

## Step 5: Analyze Results

After training completes:

```powershell
python analyze_training.py --history_path ./checkpoints/training_history.json --output_path ./training_report.png
```

This creates a comprehensive visualization of training progress.

---

## Step 6: Run Inference

### Test on a single image:

```powershell
python inference.py --model checkpoints/best_model.pth --input dataset/val/Topview14.jpg --output result.png --visualize
```

### Batch process multiple images:

```powershell
python inference.py --model checkpoints/best_model.pth --input_dir dataset/val/ --output_dir ./inference_results/ --visualize
```

---

## Expected Results

After training, you should see:

✅ **Validation Dice Score**: ~0.85 to 0.97+ (target: 0.974)  
✅ **PSNR**: ~25-35 dB  
✅ **Reconstructed images**: Clear inpainting of masked regions  

---

## File Locations After Training

```
checkpoints/
├── best_model.pth                    ← Use this for inference!
├── checkpoint_epoch_10.pth
├── training_history.json
└── samples/
    ├── epoch_10_sample_0.png
    └── ...

inference_results/
├── Topview14_inpainted.jpg
├── Topview14_visualization.png
└── ...
```

---

## Troubleshooting

### Training is slow
- You're likely on CPU. Consider using a GPU-enabled machine
- Reduce batch size: `python train_model.py --batch_size 4`

### Out of memory
- Reduce batch size: `python train_model.py --batch_size 4` or `--batch_size 2`

### Low Dice score after training
- Train longer: `python train_model.py --epochs 500`
- Reduce learning rate: `python train_model.py --lr 0.0001`
- Add more training data to `dataset/train/`

### Model predictions look wrong
- Check if you loaded the correct model
- Verify the model was trained successfully (check Dice score)
- Ensure input images are grayscale

---

## Next Steps

1. ✅ Train the model
2. 📊 Analyze training results
3. 🎯 Run inference on test images
4. 🔧 Fine-tune hyperparameters if needed
5. 📖 Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed information

---

**Need Help?** Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for comprehensive documentation.
