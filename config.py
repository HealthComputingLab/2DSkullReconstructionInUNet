from pathlib import Path
import os
from pathlib import Path

# Use project root (this file's parent) as base to build dataset paths
CURRENT_DIR = Path(__file__).resolve().parent

# Set Training path using pathlib for cross-platform compatibility
PATH_TRAIN = str((CURRENT_DIR / "dataset" / "train").resolve())
PATH_VAL = str((CURRENT_DIR / "dataset" / "val").resolve())

# Stop training when Dice reaches above threshold
from pathlib import Path

# config.py - project constants and dataset paths
# Keep this module lightweight: do not import heavy ML libraries here.

PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset paths (constructed relative to project root). Convert to str for libraries that expect strings.
PATH_TRAIN = str((PROJECT_ROOT / "dataset" / "train"))
PATH_VAL = str((PROJECT_ROOT / "dataset" / "val"))

# Training hyperparameters and constants
TARGET_DICE_SCORE = 0.974
MAX_EPOCHS_PER_ROUND = 250
MAX_RANDOM_ROUNDS = 10
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 8
LEARNING_RATE = 0.0005
CUT_SIZE = (50, 50)
perceptual_loss_rate = 0.05

__all__ = [
	"PATH_TRAIN",
	"PATH_VAL",
	"TARGET_DICE_SCORE",
	"MAX_EPOCHS_PER_ROUND",
	"MAX_RANDOM_ROUNDS",
	"IMAGE_WIDTH",
	"IMAGE_HEIGHT",
	"BATCH_SIZE",
	"LEARNING_RATE",
	"CUT_SIZE",
	"perceptual_loss_rate",
]