# Skull Reconstruction UNet

A UNet-based deep learning framework for skull reconstruction, image inpainting, and super-resolution tasks.

## Table of Contents

- [Overview](#overview)
- [Repository Status](#repository-status)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dataset Management](#dataset-management)
- [Troubleshooting](#troubleshooting)

---

## Overview

This repository implements UNet architectures for medical imaging tasks, specifically:
- **Skull reconstruction** from incomplete scan data
- **Image inpainting** for filling missing regions
- **Super-resolution** enhancement

---

## Repository Status

**Recent Changes:** All example datasets and Jupyter notebooks have been removed. The repository now contains only production model code and training scripts.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (optional, but recommended for training)

### Step 1: Create Virtual Environment

**Option A: Conda (Recommended)**

```bash
conda create -n unet_skull python=3.10 -y
conda activate unet_skull
```

**Option B: venv (Windows PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Step 2: Install PyTorch

Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) to get the command matching your CUDA version.

**Example (CPU-only):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Example (CUDA 11.8):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Project in Editable Mode

```bash
pip install -e .
```

This enables imports like `Unet_Architecture.Image_Painting` without manual PYTHONPATH configuration.

### Step 5: Verify Installation

Run the smoke test to verify your environment:

```bash
python run_smoke.py
```

This checks for:
- Core packages (torch, torchvision, PIL, numpy, matplotlib)
- Correct project module imports

---

## Configuration

All configuration parameters are defined in `config.py`. Review and modify these before training:

### Dataset Paths

| Parameter | Description | Default |
|-----------|-------------|---------|
| `PATH_TRAIN` | Training dataset directory | _(must be set)_ |
| `PATH_VAL` | Validation dataset directory | _(must be set)_ |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TARGET_DICE_SCORE` | Early stopping threshold | `0.974` |
| `MAX_EPOCHS_PER_ROUND` | Maximum epochs per training round | `250` |
| `MAX_RANDOM_ROUNDS` | Number of random restarts | `10` |
| `BATCH_SIZE` | Training batch size | `8` |
| `LEARNING_RATE` | Optimizer learning rate | `0.0005` |

### Image Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `IMAGE_WIDTH` | Input image width | `256` |
| `IMAGE_HEIGHT` | Input image height | `256` |
| `CUT_SIZE` | Patch/crop size for preprocessing | _(tuple)_ |

### Loss Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `perceptual_loss_rate` | Blending factor for perceptual loss | `0.05` |

**Important:** Edit `config.py` to point `PATH_TRAIN` and `PATH_VAL` to your local dataset folders before starting training.

---

## Usage

### Running Training

Execute the main training script:

```bash
python main_run_scan_rebuild.py
```

This will:
1. Load training and validation datasets from configured paths
2. Initialize the UNet model
3. Train with early stopping based on Dice score
4. Save checkpoints automatically

---

## Dataset Management

### Expected Dataset Structure

Organize your datasets as follows:

```
Unet_Architecture/
└── Image_Painting/
    └── dataset/
        └── dataset/
            ├── train/
            │   ├── image1.png
            │   ├── image2.png
            │   └── ...
            └── val/
                ├── image1.png
                ├── image2.png
                └── ...
```

### Downloading Datasets

For large datasets, use the provided download helper script:

```bash
python scripts/download_data.py \
  --url <DATASET_URL> \
  --out-dir Unet_Architecture/Image_Painting/dataset/dataset
```

**Supported formats:**
- `.zip`
- `.tar`, `.tar.gz`, `.tgz`

**Notes:**
- For private datasets, use signed URLs or add authentication wrappers
- Never commit large dataset files to the repository
- Host datasets externally (cloud storage, GitHub releases, etc.)

---

## Troubleshooting

### Import Errors for `Unet_Architecture.*`

**Problem:** Module import fails after installing dependencies.

**Solution:** Reinstall the project in editable mode:

```bash
pip install -e .
```

### PyTorch Installation Issues

**Problem:** Wrong PyTorch wheel or CUDA mismatch.

**Solution:**
1. Uninstall existing PyTorch: `pip uninstall torch torchvision`
2. Visit [PyTorch installation page](https://pytorch.org/get-started/locally/)
3. Install the correct wheel for your CUDA version

### Missing Dataset Folders

**Problem:** Training fails because dataset paths don't exist.

**Solution:** Create the required directory structure:

```bash
mkdir -p Unet_Architecture/Image_Painting/dataset/dataset/train
mkdir -p Unet_Architecture/Image_Painting/dataset/dataset/val
```

Then populate with your images or use the download script.

---

## Future Enhancements

Potential improvements for this project:

- [ ] Enhanced `download_data.py` script with credential support
- [ ] Unit tests and integration tests
- [ ] GitHub Actions CI/CD workflow
- [ ] Command-line arguments for `main_run_scan_rebuild.py` (seed, epochs, checkpoint path)
- [ ] Docker containerization for reproducible environments
- [ ] Model evaluation scripts with visualization
- [ ] Pre-trained model weights

---

## License

_(Add your license information here)_

## Citation

_(Add citation information if this is research work)_

## Contact

_(Add contact information or contribution guidelines)_
