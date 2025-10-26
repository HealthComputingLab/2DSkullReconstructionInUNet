# 🧠 Skull Reconstruction UNet

**Deep learning meets medical imaging**: A UNet-based framework for skull reconstruction, image inpainting, and super-resolution tasks.

> **Research Project** | This repository contains production-ready model architectures and training utilities. Large datasets and example notebooks are maintained separately.

---

## 🚀 What's Inside

This research codebase provides:

- **🎯 Training Pipeline** — `main_run_scan_rebuild.py` orchestrates the complete training workflow
- **✅ Smoke Testing** — `run_smoke.py` validates your environment and dataset configuration
- **⚙️ Centralized Config** — `config.py` manages paths, hyperparameters, and training criteria
- **🏗️ Modular Architecture** — `Unet_Architecture/` contains specialized implementations:
  - Image inpainting for skull defect reconstruction
  - Super-resolution for enhanced image quality

---

## 📋 Requirements

- **Python 3.8+** (verified compatibility)
- **GPU Recommended** — CUDA-capable hardware dramatically accelerates training
- **PyTorch** — Install from [pytorch.org](https://pytorch.org/get-started/locally/) matching your system

---

## ⚡ Quick Setup

Fire up your environment in three steps:

### 1. Create Virtual Environment


```bash
conda create -n skull2d python=3.9 -y
conda avtivate skull2d
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Install torch to use CUDA for training if "torch.cuda.is_available()" == False
# e.g. Install CUDA for Nvidia Driver 12.8 in Python 3.9
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version (PyTorch): {torch.version.cuda}')"
# Having installed the CUDA correctly, the command above should output something related to "True"

```

### 3. (Optional) Install as Package

Make `Unet_Architecture` importable from anywhere:

```powershell
pip install -e .
```

---

## 🎛️ Configuration

All settings live in `config.py`. Out of the box, it expects:

```
📂 dataset/
   ├─ 📁 train/  → PATH_TRAIN
   └─ 📁 val/    → PATH_VAL
```

**Key parameters you can tune:**
- `BATCH_SIZE` — Trade memory for speed
- `LEARNING_RATE` — Control convergence behavior
- `IMAGE_WIDTH` & `IMAGE_HEIGHT` — Input dimensions
- Early stopping criteria and epoch limits

💡 **Pro tip:** If your data lives elsewhere, just update the path variables at the top of `config.py`.

---

## 📁 Dataset Structure

Organize your images like this:

```
dataset/
  ├─ train/
  │   ├─ skull_001.png
  │   ├─ skull_002.png
  │   └─ ...
  └─ val/
      ├─ skull_val_001.png
      ├─ skull_val_002.png
      └─ ...
```

**Supported formats:** `.png`, `.jpg`, `.jpeg`

⚠️ **Important:** Never commit large datasets! Host them externally (cloud storage, institutional servers) and document download instructions separately.

---

## 🧪 Verify Your Setup

Before training, run the smoke test to catch configuration issues early:

```powershell
python run_smoke.py
```

**What it checks:**
- ✓ Config module imports correctly
- ✓ Dataset paths point to existing directories
- ✓ UNet architecture modules load without errors
- ✓ Python environment has required packages

If everything passes, you're ready to train! 🎉

---

## 🏃 Running Training

Launch training from the repository root:

```powershell
python main_run_scan_rebuild.py
```

The script will:
1. Load and validate your dataset
2. Initialize the UNet architecture
3. Train with progress monitoring
4. Save checkpoints automatically
5. Validate on your validation set

Check the top of `main_run_scan_rebuild.py` for command-line options and advanced configuration.

---

## 🎯 NEW: Production Training & Inference Scripts

We've added comprehensive training and inference scripts for production use:

### 🚀 Quick Start

```powershell
# Activate environment
conda activate 2dskull

# Train the model
python train_model.py --epochs 250 --save_dir ./checkpoints

# Run inference
python inference.py --model checkpoints/best_model.pth --input test.jpg --output result.png --visualize
```

### 📚 New Scripts & Documentation

- **`train_model.py`** - Advanced training with checkpointing, early stopping, and monitoring
- **`inference.py`** - Flexible inference for single images or batch processing
- **`analyze_training.py`** - Comprehensive training analysis and visualization
- **`COMPLETE_TUTORIAL.md`** - Complete step-by-step guide for training and inference
- **`TRAINING_GUIDE.md`** - Comprehensive training documentation
- **`QUICK_START.md`** - Get started in 5 minutes
- **`INFERENCE_SCRIPTS_README.md`** - Detailed scripts documentation

### 🎓 What's New?

✅ **Automatic Checkpointing**: Save best models automatically  
✅ **Early Stopping**: Prevent overfitting with patience-based stopping  
✅ **Learning Rate Scheduling**: Adaptive learning rate based on validation performance  
✅ **Training Visualization**: Sample predictions saved during training  
✅ **Comprehensive Metrics**: Track Dice score, PSNR, and losses  
✅ **Flexible Inference**: Single image or batch processing with visualization  
✅ **Multiple Masking Strategies**: Center, random, or custom mask positions  
✅ **Training Analysis**: Detailed plots and statistics  

### 📖 Getting Started

**For a complete walkthrough**, see:
- **[COMPLETE_TUTORIAL.md](COMPLETE_TUTORIAL.md)** - Full step-by-step guide
- **[QUICK_START.md](QUICK_START.md)** - Fast 5-minute start
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detailed training documentation


## 🗂️ Project Structure

```
skull-reconstruction-unet/
├─ 📄 config.py                    # Central configuration hub
├─ 🎯 main_run_scan_rebuild.py    # Training orchestration
├─ ✅ run_smoke.py                 # Environment validation
├─ 📦 Unet_Architecture/           # Model implementations
│   ├─ Image_Painting/             # Inpainting models
│   └─ Super_Resolution/           # SR models
├─ 📂 dataset/                     # Your data goes here
│   ├─ train/
│   └─ val/
└─ 📋 requirements.txt             # Python dependencies
```

---

## 📝 Development Notes

**Best Practices:**
- 🔒 Keep datasets external — use `.gitignore` to exclude `dataset/` contents
- 📊 Track experiments with clear naming conventions
- 💾 Regularly backup trained model checkpoints
- 🧬 Document any architectural modifications

**Next Steps for Publication:**
- Add `LICENSE` file (consider MIT, Apache 2.0, or GPL)
- Include `CITATION.cff` for academic attribution
- Create `CONTRIBUTING.md` for community guidelines
- Expand documentation with dataset preparation guide

---

## 🤝 Contributing

This is research code under active development. Contributions, bug reports, and feature requests are welcome through issues and pull requests.

---

**Built with PyTorch** | **Powered by UNet** | **Advancing Medical AI**