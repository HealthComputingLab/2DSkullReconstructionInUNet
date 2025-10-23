Project: Skull_Reconstruction_Unet

This repository contains UNet-based architectures for skull reconstruction, image painting and super-resolution.

Status after cleanup

All example dataset images and Jupyter notebooks have been removed from this repository (they were local test artifacts). The repository now contains the model code and scripts only.

Getting started (recommended: create a new virtual environment)

1) Create and activate an environment

   - Conda (recommended):

     ```powershell
     conda create -n unet_skull python=3.10 -y
     conda activate unet_skull
     ```

   - venv (Windows PowerShell):

     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     python -m pip install --upgrade pip
     ```

2) Install PyTorch (pick the right CUDA / CPU variant)

   - PyTorch wheels depend on your CUDA and Python version. See the official page for the correct command: [PyTorch Get Started](https://pytorch.org/get-started/locally/).

   - Example (CPU-only):

     ```powershell
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     ```

3) Install the remaining Python packages

   ```powershell
   pip install -r requirements.txt
   ```

4) Install the project in editable mode (optional but recommended)

   ```powershell
   python -m pip install -e .
   ```

   This makes imports like `Unet_Architecture.Image_Painting` work without setting PYTHONPATH.

Running the smoke test (quick environment check)

```powershell
python run_smoke.py
```

The script verifies presence of core packages (torch, torchvision, PIL, numpy, matplotlib) and attempts to import the project modules.

Running training / main runner

Example (PowerShell):

```powershell
python main_run_scan_rebuild.py
```

This will load the training and validation datasets from the paths configured in `config.py`.

Configuration (constants in `config.py`)

The repository exposes a small set of configuration constants you should be aware of. They are defined in `config.py` and used by the training scripts:

- `PATH_TRAIN` — default dataset training path (string)
- `PATH_VAL` — default dataset validation path (string)
- `TARGET_DICE_SCORE` — early stopping target Dice score (float, default 0.974)
- `MAX_EPOCHS_PER_ROUND` — maximum epochs per training round (int, default 250)
- `MAX_RANDOM_ROUNDS` — number of random restarts (int, default 10)
- `IMAGE_WIDTH`, `IMAGE_HEIGHT` — input image size (default 256 × 256)
- `BATCH_SIZE` — training batch size (default 8)
- `LEARNING_RATE` — optimizer LR (default 0.0005)
- `CUT_SIZE` — patch/crop size used in preprocessing (tuple)
- `perceptual_loss_rate` — blending factor for perceptual loss (float, default 0.05)

Edit `config.py` to point `PATH_TRAIN` and `PATH_VAL` at your local dataset folders before running training.

Notes and troubleshooting

- If imports fail for `Unet_Architecture.*` after installing packages, re-run:

  ```powershell
  python -m pip install -e .
  ```

- If torch is missing or the wrong wheel is installed, uninstall and re-install the correct PyTorch wheel according to your CUDA version.

- If you removed datasets locally but need to train, re-create the dataset folders and place images under:

  - `Unet_Architecture/Image_Painting/dataset/dataset/train`
  - `Unet_Architecture/Image_Painting/dataset/dataset/val`

  Downloading datasets

  If your dataset is large, we recommend hosting it externally (cloud storage, an HTTP server, or a release artifact).
  This repository contains a small helper script to download and extract dataset archives into the project:

  ```powershell
  python scripts/download_data.py --url <DATASET_URL> --out-dir Unet_Architecture/Image_Painting/dataset/dataset
  ```

  Notes:
  - The script supports .zip and tar(.gz/.tgz) archives.
  - For private datasets, use signed URLs or add an authentication wrapper before calling the script.
  - Do not commit large dataset files to the repository. Instead, add them to a release or cloud storage and download them at setup time.

Optional next tasks I can help with

- Add a small `scripts/download_data.py` that downloads datasets from a provided URL (add credentials if needed).
- Add lightweight unit/smoke tests and a GitHub Actions workflow to run them on every push.
- Harden `main_run_scan_rebuild.py` with more command-line flags (seed, epochs, checkpoint path, quick-mode).

If you want a short CHANGELOG entry listing the files removed and renamed, I can add it now.
