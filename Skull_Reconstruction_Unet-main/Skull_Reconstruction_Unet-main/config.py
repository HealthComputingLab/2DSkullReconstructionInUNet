from Unet_Architecture.Image_Painting.Library import *

# config.py
currentDir = os.getcwd() # Lấy thư mục hiện tại)
#print(os.getcwd())
  
#Set Training path
PATH_TRAIN = currentDir +"/Unet_Architecture/Image_Painting/dataset/dataset/train"
PATH_VAL = currentDir+"/Unet_Architecture/Image_Painting/dataset/dataset/val"

# Dừng huấn luyện khi Dice đạt trên ngưỡng
TARGET_DICE_SCORE = 0.974

# Epoch tối đa mỗi lần huấn luyện với 1 random mask
MAX_EPOCHS_PER_ROUND = 250

# Số lần random lại mask mới (số vòng huấn luyện độc lập)
MAX_RANDOM_ROUNDS = 10

# Kích thước ảnh
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# Batch size
BATCH_SIZE = 8

# Learning rate
LEARNING_RATE = 0.0005

# Cut size (vùng che khuyết)
CUT_SIZE = (50, 50)

#Tính loss
perceptual_loss_rate = 0.05