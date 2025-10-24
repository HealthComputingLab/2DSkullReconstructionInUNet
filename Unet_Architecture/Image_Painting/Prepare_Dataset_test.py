from Unet_Architecture.Image_Painting import library as lib
# NOTE: This test module uses the Image_Painting.library helper. Keep imports minimal.

# from datetime import datetime

class ImageDataset(Dataset):
    """
    Dataset for image inpainting task with random line masking.
    
    Creates random line masks on RGB images to simulate damaged/corrupted regions
    that the model needs to reconstruct.
    """
    def __init__(self, data_dir, img_width, img_height, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.img_width = img_width
        self.img_height = img_height
        self.images = os.listdir(self.data_dir)

    def normalize(self, input_img, target_img):
        """
        Normalize images from [0, 1] to [-1, 1] range.
        
        PyTorch's to_tensor() automatically scales images to [0, 1],
        so we scale to [-1, 1] for better training stability.
        """
        input_img = input_img * 2 - 1
        target_img = target_img * 2 - 1
        return input_img, target_img

    def random_transform(self, input_image, target_image):
        """Apply random horizontal flip augmentation with 50% probability."""
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
        return input_image, target_image

    def create_mask(self, image, img_height, img_width):
        """
        Create random line masks to simulate image corruption.
        
        Draws 1-4 random lines with varying thickness on the image,
        replacing masked regions with white (255).
        
        Args:
            image: Original image array
            img_height: Image height
            img_width: Image width
            
        Returns:
            mask_image: Image with random lines masked as white
        """
        mask_img = image.copy()
        mask = np.full((img_height, img_width, 3), 0, dtype=np.uint8)
        
        # Draw 1-4 random lines
        for _ in range(np.random.randint(1, 5)):
            # Generate random coordinates for two points
            x1, y1 = np.random.randint(1, img_width), np.random.randint(1, img_height)
            x2, y2 = np.random.randint(1, img_width), np.random.randint(1, img_height)
            thick = np.random.randint(1, 15)

            cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness=thick)
        
        # Where mask is True (1), set pixel to white (255); otherwise keep original
        mask_image = np.where(mask, 255 * np.ones_like(mask), mask_img)
        return mask_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.images[item])
        image = np.array(Image.open(img_path).convert("RGB"))
        input_image = self.create_mask(image, self.img_height, self.img_width)
        input_image = transforms.functional.to_tensor(input_image)
        target_image = transforms.functional.to_tensor(image)

        input_image, target_image = self.normalize(input_image, target_image)

        if self.is_train:
            input_image, target_image = self.random_transform(input_image, target_image)

        return input_image, target_image


def visualize_data(train_loader):
    """
    Visualize input and target image pairs from the dataset.
    
    Displays the first two samples showing masked inputs and ground truth targets.
    """
    input_batch, target_batch = next(iter(train_loader))
    # Rescale from [-1, 1] to [0, 1] for visualization
    input_batch = (input_batch + 1) / 2
    target_batch = (target_batch + 1) / 2

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 2, 1)
    plt.title("Input")
    plt.imshow(input_batch[0].numpy().transpose(1, 2, 0))
    plt.axis('off')

    ax = plt.subplot(2, 2, 2)
    plt.title("Target")
    plt.imshow(target_batch[0].numpy().transpose(1, 2, 0))
    plt.axis('off')

    ax = plt.subplot(2, 2, 3)
    plt.title("Input")
    plt.imshow(input_batch[1].numpy().transpose(1, 2, 0))
    plt.axis('off')

    ax = plt.subplot(2, 2, 4)
    plt.title("Target")
    plt.imshow(target_batch[1].numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.show()
    # plt.savefig(f"visualized_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")


def data(path_train, path_val, width_size, height_size, batch_size):
    """Create training and validation datasets and dataloaders."""
    train_dataset = ImageDataset(path_train, width_size, height_size, True)
    val_dataset = ImageDataset(path_val, width_size, height_size, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader


print(__name__)
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
    print(f"Number of train batch: {len(train_loader)} || Number of val batch: {len(val_loader)}")

    visualize_data(train_loader)