import config as cf
import argparse
from pathlib import Path
import sys

def _import_ml_stack():
    """Import ML dependencies when needed. Raises informative error if missing."""
    try:
        import torch
        import torch.optim as optim
        import torch.nn as nn
        import lpips
    except ModuleNotFoundError as e:
        missing = e.name
        raise ModuleNotFoundError(
            f"Missing required package '{missing}'. Install project requirements: see requirements.txt and run `pip install -r requirements.txt`."
        ) from e

    # Support both lowercase and legacy CamelCase module names
    data = None
    training_1pos_2crit = None
    predict_and_display = None
    Unet = None

    # try prepare_dataset / Prepare_Dataset
    try:
        from Unet_Architecture.Image_Painting.prepare_dataset import data
    except Exception:
        try:
            from Unet_Architecture.Image_Painting.Prepare_Dataset import data
        except Exception:
            pass

    # try train_val / Train_Val
    try:
        from Unet_Architecture.Image_Painting.train_val import training_1pos_2crit, predict_and_display
    except Exception:
        try:
            from Unet_Architecture.Image_Painting.Train_Val import training_1pos_2crit, predict_and_display
        except Exception:
            pass

    # try unet_skip_connection / UnetSkipConnection
    try:
        from Unet_Architecture.Image_Painting.unet_skip_connection import Unet
    except Exception:
        try:
            from Unet_Architecture.Image_Painting.UnetSkipConnection import Unet
        except Exception:
            pass

    if data is None or training_1pos_2crit is None or predict_and_display is None or Unet is None:
        print("Failed to import training modules. Make sure the package path is correct and module names match (prepare_dataset/Prepare_Dataset, train_val/Train_Val, unet_skip_connection/UnetSkipConnection).")
        raise ImportError("Could not import required training modules from Image_Painting subpackage")

    return torch, optim, nn, lpips, data, training_1pos_2crit, predict_and_display, Unet


def main(args):
    # Import ML stack lazily to allow running lightweight checks without ML deps
    torch, optim, nn, lpips, data, training_1pos_2crit, predict_and_display, Unet = _import_ml_stack()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    path_train = Path(cf.PATH_TRAIN)
    path_val = Path(cf.PATH_VAL)

    # Quick mode: create tiny synthetic dataset if requested
    if getattr(args, 'quick', False):
        import shutil
        from PIL import Image
        import numpy as np
        # Use a separate folder to avoid overwriting real dataset
        tmp_dir = Path.cwd() / "dataset_quick"
        # Clean up any previous quick dataset only
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        train_dir = tmp_dir / "train"
        val_dir = tmp_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        def _make_image(path, w, h, pattern=0):
            arr = (np.random.rand(h, w) * 255).astype('uint8')
            img = Image.fromarray(arr, mode='L')
            img.save(path)

        # create a few tiny images
        for i in range(8):
            _make_image(train_dir / f"train_{i:03d}.png", cf.IMAGE_WIDTH, cf.IMAGE_HEIGHT)
        for i in range(2):
            _make_image(val_dir / f"val_{i:03d}.png", cf.IMAGE_WIDTH, cf.IMAGE_HEIGHT)

        path_train = train_dir
        path_val = val_dir
    print(f"Quick mode: created synthetic dataset at {tmp_dir} (non-destructive)")

    if not path_train.exists():
        raise FileNotFoundError(f"Training path not found: {path_train}")
    if not path_val.exists():
        raise FileNotFoundError(f"Validation path not found: {path_val}")

    train_dataset, val_dataset, train_loader, val_loader = data(str(path_train), str(path_val), cf.IMAGE_WIDTH, cf.IMAGE_HEIGHT, cf.BATCH_SIZE)

    in_channels = 1
    num_classes = 1
    model = Unet(in_channels, num_classes)
    model.to(device)

    criterion_bce = nn.BCEWithLogitsLoss().to(device)
    criterion_lpips = lpips.LPIPS(net='vgg').to(device)

    optimizer = optim.Adam(model.parameters(), lr=cf.LEARNING_RATE, betas=[0.5, 0.999])

    # If quick mode, run a single epoch for a fast smoke-test
    epochs = 1 if getattr(args, 'quick', False) else cf.MAX_EPOCHS_PER_ROUND
    model, metrics = training_1pos_2crit(model, optimizer, criterion_bce, criterion_lpips, train_loader, val_loader, epochs, device, cf.perceptual_loss_rate)

    predict_and_display(model, val_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run a short quick test instead of full training (not implemented)')
    args = parser.parse_args()
    main(args)
