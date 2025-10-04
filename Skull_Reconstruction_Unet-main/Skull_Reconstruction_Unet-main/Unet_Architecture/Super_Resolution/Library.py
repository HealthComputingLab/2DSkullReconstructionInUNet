import matplotlib.pyplot as plt
import os
import numpy as np
import random
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchsummary import summary
from torcheval.metrics.functional import peak_signal_noise_ratio

SEED = 1
BATCH_SIZE = 8
IMG_HEIGHT = 64
IMG_WIDTH = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8):
    """
    Tính Dice coefficient giữa ảnh dự đoán và ground truth.
    - pred: đầu ra sau sigmoid, có shape (N, 1, H, W)
    - target: ground truth (N, 1, H, W)
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()

