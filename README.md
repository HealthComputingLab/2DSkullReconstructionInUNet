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

**PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Bash/Zsh:**
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
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