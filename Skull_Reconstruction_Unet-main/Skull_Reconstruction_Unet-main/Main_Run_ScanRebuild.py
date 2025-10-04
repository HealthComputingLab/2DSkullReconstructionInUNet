from Unet_Architecture.Image_Painting.Library import *
from Unet_Architecture.Image_Painting.Prepare_Dataset import data
#from Unet_Architecture.Image_Painting.Train_Val import training, valid_epoch, plot_result, predict_and_display
from Unet_Architecture.Image_Painting.Train_Val import *
from Unet_Architecture.Image_Painting.UnetSkipConnection import Unet
import config as cf

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
#print(os.getcwd())
  
#Set Training path
path_train = cf.PATH_TRAIN
path_val = cf.PATH_VAL
print(f"[TRAIN_PATH] = [{path_train}]" )
print(f"[VALIDATION_PATH] = [{path_val}]")
train_dataset, val_dataset, train_loader, val_loader = data(path_train, path_val, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE)

#print(train_dataset.data_dir + "\n"+val_dataset.data_dir+"\n"+str(train_loader.__len__()) +"\n"+str(val_loader.__len__()))
#Set parameters
perceptual_loss_rate = cf.perceptual_loss_rate
in_channels = 1
num_classes = 1
max_epochs = cf.MAX_EPOCHS_PER_ROUND
num_epochs = max_epochs
max_random_round= cf.MAX_RANDOM_ROUNDS
model = Unet(in_channels, num_classes)
model.to(device)
#Default Loss
#criterion = nn.L1Loss() #-- Fail
#1st try: Loss theo pixel (Binary Cross Entropy)
# [RESULT]: Nối lại được nhưng bị mất cấu trúc xương
#criterion = nn.BCEWithLogitsLoss() 

#2nd try: Perceptual Loss (LPIPS với backbone VGG) kết hợp cùng với BCEWithLogitsLoss
# Loss theo pixel (Binary Cross Entropy)
criterion_bce = nn.BCEWithLogitsLoss().to(device)
# Perceptual Loss với mạng VGG (LPIPS)
criterion_lpips = lpips.LPIPS(net='vgg').to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.0005, betas=[0.5, 0.999])

'''Run model_with 1 criterion
#model, metrics = training(model, optimizer, criterion, train_loader, val_loader, num_epochs, device)
'''
#Run model with 1 position 2 criterion
model, metrics = training_1pos_2crit(model, optimizer, criterion_bce, criterion_lpips, train_loader, val_loader, num_epochs, device, perceptual_loss_rate)

'''#Run model with random position_2 criterion
model, metrics = training_mulPos_2crit(model, optimizer, criterion_bce, criterion_lpips, train_loader, val_loader, max_epochs, max_random_round, device,perceptual_loss_rate)
'''
#Display 
predict_and_display(model, val_loader, device)