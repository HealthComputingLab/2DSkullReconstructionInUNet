from Unet_Architecture.Image_Painting.Library import *

cut_size = (50, 50)  # Size of the rectangular mask region

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    """
    Dataset for grayscale image inpainting with fixed or brightness-based masking.
    
    Returns (input_tensor, target_tensor, mask_tensor) where:
    - input_tensor: Masked image scaled to [-1, 1]
    - target_tensor: Original image scaled to [-1, 1]
    - mask_tensor: Binary mask where 1 indicates the region to learn/reconstruct
    """
    def __init__(self, data_dir, img_width, img_height, is_train=True, cut=cut_size):
        self.data_dir = data_dir
        self.is_train = is_train
        self.img_width = img_width
        self.img_height = img_height
        self.cut_size = cut
        self.images = sorted(os.listdir(self.data_dir))

    def normalize(self, tensor):
        """Scale tensor from [0, 1] to [-1, 1]."""
        return tensor * 2 - 1

    def random_transform(self, input_image, target_image, mask_learn):
        """Apply random horizontal flip to all tensors simultaneously."""
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
            mask_learn = transforms.functional.hflip(mask_learn)
        return input_image, target_image, mask_learn

    def _create_mask_fixed(self, image):
        """
        Create a fixed rectangular mask at position (42, 113).
        
        If the mask region exceeds image boundaries, centers it instead.
        """
        mask = image.copy()
        height, width = mask.shape
        box_h, box_w = self.cut_size
        x, y = 42, 113
        
        # Check if mask region is within image bounds
        if x + box_w > width or y + box_h > height:
            x = max(0, (width - box_w) // 2)
            y = max(0, (height - box_h) // 2)
        
        # Create learning mask (region to reconstruct)
        mask_learn = np.zeros_like(mask)
        mask_learn[y:y + box_h, x:x + box_w] = 1.0
        
        # Black out the masked region
        mask[y:y + box_h, x:x + box_w] = 0
        return mask, mask_learn

    def _create_mask_by_brightness(self, image, attempts=100):
        """
        Create mask in a bright region of the image (adaptive masking).
        
        Searches for a region with >5% white pixels (brightness > 200).
        Falls back to zero mask if no suitable region found after max attempts.
        
        Args:
            image: Grayscale input image
            attempts: Maximum number of random placement attempts
            
        Returns:
            mask: Image with masked region set to black
            mask_learn: Binary mask indicating the learning region
        """
        mask = image.copy()
        gray = mask.copy()
        
        # Threshold to find bright regions
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        height, width = gray.shape
        box_h, box_w = self.cut_size
        
        # Try random positions until finding a bright enough region
        for _ in range(attempts):
            x = np.random.randint(0, max(1, width - box_w))
            y = np.random.randint(0, max(1, height - box_h))
            region = thresh[y:y + box_h, x:x + box_w]
            white_ratio = np.sum(region == 255) / (box_h * box_w)
            
            if white_ratio > 0.05:  # At least 5% bright pixels
                mask_learn = np.zeros_like(mask)
                mask_learn[y:y + box_h, x:x + box_w] = 1.0
                mask[y:y + box_h, x:x + box_w] = 0
                return mask, mask_learn
        
        # No suitable region found
        return mask, np.zeros_like(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("L").resize((self.img_width, self.img_height)))
        
        # Try brightness-based masking first, fall back to fixed if needed
        mask, mask_learn = self._create_mask_by_brightness(image)
        if mask_learn.sum() == 0:
            mask, mask_learn = self._create_mask_fixed(image)

        input_tensor = transforms.functional.to_tensor(mask)
        target_tensor = transforms.functional.to_tensor(image)
        mask_tensor = torch.from_numpy(mask_learn).unsqueeze(0).float()

        input_tensor = self.normalize(input_tensor)
        target_tensor = self.normalize(target_tensor)

        if self.is_train:
            input_tensor, target_tensor, mask_tensor = self.random_transform(
                input_tensor, target_tensor, mask_tensor
            )

        return input_tensor, target_tensor, mask_tensor


def visualize_data(train_loader):
    """Display input, target, and mask for the first sample in the batch."""
    input_batch, target_batch, mask_batch = next(iter(train_loader))
    # Rescale from [-1, 1] to [0, 1] for display
    input_batch = (input_batch + 1) / 2
    target_batch = (target_batch + 1) / 2

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title("Input")
    plt.imshow(input_batch[0].squeeze().numpy(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title("Target")
    plt.imshow(target_batch[0].squeeze().numpy(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.title("Mask")
    plt.imshow(mask_batch[0].squeeze().numpy(), cmap='gray')
    plt.axis('off')


def data(path_train, path_val, width_size, height_size, batch_size):
    """Create training and validation datasets with dataloaders."""
    from torch.utils.data import DataLoader

    train_dataset = ImageDataset(path_train, width_size, height_size, True)
    val_dataset = ImageDataset(path_val, width_size, height_size, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader


if __name__ == "__main__":
    width_size = 256
    height_size = 256
    batch_size = 8
    path_train = "./dataset/dataset/train"
    path_val = "./dataset/dataset/val"

    train_dataset, val_dataset, train_loader, val_loader = data(
        path_train, path_val, width_size, height_size, batch_size
    )
    print(f"Number of image train: {len(train_dataset)} || Number of image val: {len(val_dataset)}")
    visualize_data(train_loader)
    plt.show()
