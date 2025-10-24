import matplotlib.pyplot as plt
import os
import numpy as np
import random
import time
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchsummary import summary
from torcheval.metrics.functional import peak_signal_noise_ratio

# Custom libraries
import tkinter as tk
import lpips

# Hyperparameters and configuration
SEED = 1  # Random seed for reproducibility
BATCH_SIZE = 8  # Number of samples per batch
IMG_HEIGHT = 256  # Image height in pixels
IMG_WIDTH = 256  # Image width in pixels

# Set random seeds for reproducibility across all libraries
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CUDA operations

# Set computation device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8):
	"""
	Calculate Dice coefficient between predicted and ground truth images.
	
	The Dice coefficient measures the overlap between two binary masks, commonly used
	in image segmentation tasks. It ranges from 0 (no overlap) to 1 (perfect overlap).
	
	Args:
		pred: Model predictions after sigmoid activation, shape (N, 1, H, W)
		      where N is batch size, H is height, W is width
		target: Ground truth labels, shape (N, 1, H, W)
		threshold: Threshold value to binarize predictions (default: 0.5)
		eps: Small epsilon value to prevent division by zero (default: 1e-8)
	
	Returns:
		Mean Dice coefficient across the batch as a scalar tensor
	
	Formula:
		Dice = 2 * |pred ∩ target| / (|pred| + |target|)
	"""
	# Binarize predictions using threshold
	pred_bin = (pred > threshold).float()
	target_bin = (target > threshold).float()
	
	# Calculate intersection: element-wise multiplication and sum over spatial dimensions
	intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
	
	# Calculate union: sum of both binary masks
	union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
	
	# Compute Dice coefficient with epsilon to avoid division by zero
	dice = (2 * intersection + eps) / (union + eps)
	
	# Return mean Dice score across the batch
	return dice.mean()