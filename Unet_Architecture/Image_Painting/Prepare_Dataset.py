from Unet_Architecture.Image_Painting.Library import *
#from Library import *
cut_size=(50,50)
'''
class ImageDataset(Dataset):
    def __init__(self, data_dir, img_width, img_height, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.img_width = img_width
        self.img_height = img_height
        self.images = os.listdir(self.data_dir)

    def normalize(self, input_img, target_img):
        # do pytorch đã scale ảnh về (0, 1), ta cần scale về (-1, 1)
        input_img = input_img * 2 - 1
        target_img = target_img * 2 - 1

        return input_img, target_img

    def random_transform(self, input_image, target_image):
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
        return input_image, target_image
    
    def create_mask(self, image, img_height, img_width, cut_size,image_id=None, debug = False):

        mask = image.copy()

        # Convert sang grayscale nếu ảnh có 3 kênh

        if len(mask.shape) == 3 and mask.shape[2] == 3:
            gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            gray = mask.copy()

        # Threshold ảnh để tìm vùng trắng (giá trị lớn hơn 200)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        height, width = gray.shape
        box_h, box_w = cut_size

        max_attempts = 100
        found = False
        for _ in range(max_attempts):
            x = np.random.randint(0, width - box_w)
            y = np.random.randint(0, height - box_h)
            region = thresh[y:y + box_h, x:x + box_w]
            white_ratio = np.sum(region == 255) / (box_h * box_w)
            #print(white_ratio)
            if white_ratio > 0.05:
                found = True
                self.mask_coords[image_id] = (x, y)
                break

        if not found:
            # fallback: không tìm được vùng trắng → trả lại ảnh gốc
            return mask
        
        # Vẽ hình vuông màu đen vào vùng đã chọn
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask[y:y + box_h, x:x + box_w] = (0, 0, 0)
            mask_region = np.any(mask != image, axis=2)
        else:
            mask[y:y + box_h, x:x + box_w] = 0
            mask_region = (mask != image)
        if debug:
                # Numeric check
                num_masked = mask_region.sum()
                total_px = img_height * img_width
                print(f"[DEBUG] Masked pixels: {num_masked}/{total_px} ({100*num_masked/total_px:.2f}%)")

                # Visual check: overlay đỏ lên ảnh gốc
                ys, xs = np.where(mask_region)
                plt.figure(figsize=(8,4))
                plt.subplot(1,2,1)
                plt.title("Masked Input")
                plt.imshow(mask, cmap='gray')
                plt.axis('off')
                plt.subplot(1,2,2)
                plt.title("Overlay Mask")
                plt.imshow(image, cmap='gray')
                plt.scatter(xs, ys, s=1, c='red', alpha=0.5)
                plt.axis('off')
                plt.show()
        #print(mask.shape)
        #cv2.waitKey(0)
        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.images[item])
        #image = np.array(Image.open(img_path).convert("RGB"))
        image = np.array(Image.open(img_path).convert("L")) #'L '= grayscale
        image_id = self.images[item]
        input_image = self.create_mask(image, self.img_height, self.img_width, cut_size, image_id=image_id)
        #input_image = self.create_mask(image, self.img_height, self.img_width)
        #cv2.imshow("Input",input_image)
        #cv2.imshow("Target",image)
        #cv2.waitKey(0)
        input_image = transforms.functional.to_tensor(input_image)
        target_image = transforms.functional.to_tensor(image)

        #input_image, target_image = self.normalize(input_image, target_image)

        if self.is_train:
            input_image, target_image = self.random_transform(input_image, target_image)

        return input_image, target_image
'''
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, data_dir, img_width, img_height, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.img_width = img_width
        self.img_height = img_height
        self.cut_size = cut_size
        self.images = os.listdir(self.data_dir)
        self.mask_coords = {}
        self.selected_image = None # Image File Name

    def normalize(self, input_img, target_img):
        # Scale về (-1, 1)
        input_img = input_img * 2 - 1
        target_img = target_img * 2 - 1
        return input_img, target_img

    def random_transform(self, input_image, target_image, mask_learn):
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
            mask_learn = transforms.functional.hflip(mask_learn)
        return input_image, target_image, mask_learn

    def create_mask(self, image, cut_size, image_id=None, debug=False):
        mask = image.copy()
        gray = mask.copy()

        height, width = gray.shape
        box_h, box_w = cut_size


        x, y = 42, 113

        # ✅ Kiểm tra nếu vùng cắt vượt khỏi ảnh
        if x + box_w > width or y + box_h > height:
            raise ValueError(f"Vùng cắt ({x}, {y}, {box_w}, {box_h}) vượt ngoài kích thước ảnh ({width}, {height})")

        if image_id:
            self.mask_coords[image_id] = (x, y)

        # Tạo mask học (vùng cần học)
        mask_learn = np.zeros_like(mask)
        mask_learn[y:y + box_h, x:x + box_w] = 1.0

        # Vẽ đè lên vùng trắng bằng màu đen
        mask[y:y + box_h, x:x + box_w] = 0

        if debug:
            ys, xs = np.where(mask_learn)
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.title("Masked Input")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title("Overlay Mask")
            plt.imshow(image, cmap='gray')
            plt.scatter(xs, ys, s=1, c='red', alpha=0.5)
            plt.axis('off')
            plt.show()

        return mask, mask_learn

    def create_mask_mul_pos(self, image, cut_size, image_id=None, debug=False):
        mask = image.copy()
        gray = mask.copy()

        # Threshold để tìm vùng trắng
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        height, width = gray.shape
        box_h, box_w = cut_size
        max_attempts = 100
        found = False

        for _ in range(max_attempts):
            x = np.random.randint(0, width - box_w)
            y = np.random.randint(0, height - box_h)
            region = thresh[y:y + box_h, x:x + box_w]
            white_ratio = np.sum(region == 255) / (box_h * box_w)
            if white_ratio > 0.05:
                found = True
                if image_id:
                    self.mask_coords[image_id] = (x, y)
                break

        if not found:
            return mask, np.zeros_like(mask)

        # Tạo mask học (vùng cần học)
        mask_learn = np.zeros_like(mask)
        mask_learn[y:y + box_h, x:x + box_w] = 1.0

        # Vẽ đè lên vùng trắng bằng màu đen
        mask[y:y + box_h, x:x + box_w] = 0

        if debug:
            ys, xs = np.where(mask_learn)
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.title("Masked Input")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title("Overlay Mask")
            plt.imshow(image, cmap='gray')
            plt.scatter(xs, ys, s=1, c='red', alpha=0.5)
            plt.axis('off')
            plt.show()

        return mask, mask_learn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.images[item])
        image_id = self.images[item]

        image = np.array(Image.open(img_path).convert("L").resize((self.img_width, self.img_height)))

        input_img_np, mask_learn_np = self.create_mask(image, self.cut_size, image_id=image_id)
        input_tensor = transforms.functional.to_tensor(input_img_np)  # [1, H, W]
        target_tensor = transforms.functional.to_tensor(image)        # [1, H, W]
        mask_learn_tensor = torch.from_numpy(mask_learn_np).unsqueeze(0).float()  # [1, H, W]

        # input_tensor, target_tensor = self.normalize(input_tensor, target_tensor)

        if self.is_train:
            input_tensor, target_tensor, mask_learn_tensor = self.random_transform(
                input_tensor, target_tensor, mask_learn_tensor
            )

        return input_tensor, target_tensor, mask_learn_tensor

def visualize_data(train_loader):
    input_batch, target_batch = next(iter(train_loader))
    # đưa ảnh về (0, 1) để visualize
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

def data(path_train, path_val, width_size, height_size, batch_size):
    train_dataset = ImageDataset(path_train, width_size, height_size, True)
    val_dataset = ImageDataset(path_val, width_size, height_size, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader

#print(__name__)
if __name__ == "__main__":
    width_size = 256
    height_size = 256
    batch_size = 8
    #path_train = "./dataset/dataset/train"
    #path_val = "./dataset/dataset/val"
    path_train = "D:\\ToanLD_20231033M\\1_MainProject\\Unet_Architecture\\Image_Painting\\dataset\\dataset\\train"
    path_val = "D:\\ToanLD_20231033M\\1_MainProject\\Unet_Architecture\\Image_Painting\\dataset\\dataset\\val"

    # images = os.listdir(path_train)

    # train_dataset = ImageDataset(path_train, width_size, height_size, True)
    # val_dataset = ImageDataset(path_val, width_size, height_size, False)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_dataset, val_dataset, train_loader, val_loader = data(path_train, path_val, width_size, height_size, batch_size)
    print(f"Number of image train: {len(train_dataset)} ||Number of image val: {len(val_dataset)}")
    print(f"Number of train batch: {len(train_loader)} || Number of val batch: {len(val_loader)}")

    visualize_data(train_loader)
    """Dataset returning (input_tensor, target_tensor, mask_tensor).

    - input_tensor: masked image scaled to [-1, 1]
    - target_tensor: original image scaled to [-1, 1]
    - mask_tensor: single-channel mask where 1 indicates the region to learn
    """

    def __init__(self, data_dir, img_width, img_height, is_train=True, cut=cut_size):
        self.data_dir = data_dir
        self.is_train = is_train
        self.img_width = img_width
        self.img_height = img_height
        self.cut_size = cut
        self.images = sorted(os.listdir(self.data_dir))

    def normalize(self, tensor):
        return tensor * 2 - 1

    def random_transform(self, input_image, target_image, mask_learn):
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
            mask_learn = transforms.functional.hflip(mask_learn)
        return input_image, target_image, mask_learn

    def _create_mask_fixed(self, image):
        mask = image.copy()
        height, width = mask.shape
        box_h, box_w = self.cut_size
        x, y = 42, 113
        if x + box_w > width or y + box_h > height:
            x = max(0, (width - box_w) // 2)
            y = max(0, (height - box_h) // 2)
        mask_learn = np.zeros_like(mask)
        mask_learn[y:y + box_h, x:x + box_w] = 1.0
        mask[y:y + box_h, x:x + box_w] = 0
        return mask, mask_learn

    def _create_mask_by_brightness(self, image, attempts=100):
        mask = image.copy()
        gray = mask.copy()
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        height, width = gray.shape
        box_h, box_w = self.cut_size
        for _ in range(attempts):
            x = np.random.randint(0, max(1, width - box_w))
            y = np.random.randint(0, max(1, height - box_h))
            region = thresh[y:y + box_h, x:x + box_w]
            white_ratio = np.sum(region == 255) / (box_h * box_w)
            if white_ratio > 0.05:
                mask_learn = np.zeros_like(mask)
                mask_learn[y:y + box_h, x:x + box_w] = 1.0
                mask[y:y + box_h, x:x + box_w] = 0
                return mask, mask_learn
        return mask, np.zeros_like(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("L").resize((self.img_width, self.img_height)))
        mask, mask_learn = self._create_mask_by_brightness(image)
        if mask_learn.sum() == 0:
            mask, mask_learn = self._create_mask_fixed(image)

        input_tensor = transforms.functional.to_tensor(mask)
        target_tensor = transforms.functional.to_tensor(image)
        mask_tensor = torch.from_numpy(mask_learn).unsqueeze(0).float()

        input_tensor = self.normalize(input_tensor)
        target_tensor = self.normalize(target_tensor)

        if self.is_train:
            input_tensor, target_tensor, mask_tensor = self.random_transform(input_tensor, target_tensor, mask_tensor)

        return input_tensor, target_tensor, mask_tensor


def visualize_data(train_loader):
    input_batch, target_batch, mask_batch = next(iter(train_loader))
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

    train_dataset, val_dataset, train_loader, val_loader = data(path_train, path_val, width_size, height_size, batch_size)
    print(f"Number of image train: {len(train_dataset)} ||Number of image val: {len(val_dataset)}")
    visualize_data(train_loader)
    plt.show()
