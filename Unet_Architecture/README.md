## Introduction.

1. Introduction to Unet:
UNET is a deep neural network model developed by Olaf Ronneberger, Philipp Fischer
and Thomas Brox in 2015, mainly used in the field of image processing and artificial intelligence (AI). This model has gained wide popularity in applications related to image segmentation, such as cell segmentation in medical images or object segmentation in high-
quality images.

UNET uses an Encoder-Decoder network architecture, with the main purpose of the skip connection layer being
to create a short path from input to output. The activation function commonly used in UNET is
the Rectified Linear Unit (ReLU).

2. In this task, we will do 2 applications of the Unet network: increasing the resolution of the image (Super Resolution) and reconstructing a missing or obscured part of the image (Image Painting).

3. UNet network structure in this task:

FirstFeature Class:
• Purpose: Create an initial feature map from the input
• Components: A single convolution layer followed by a LeakyReLU function. This convolution
uses a kernel size of 1, a stride of 1, and no padding. This is a simple layer
designed to expand the number of channels for the feature map.

ConvBlock Class:

• Purpose: The basic convolution block for feature extraction
• Components: Two continuous Conv-BatchNorm-LeakyReLU groups. This block is a basic
block in U-Net, used for both down-sampling and up-sampling.

Encoder Class:

• Purpose: To reduce the size of the feature map and extract high-level features.
• Components: A Max Pooling layer followed by a ConvBlock. Max Pooling reduces the size
by half, while the ConvBlock processes the features.

Decoder Class:

• Purpose: To increase the size of the feature map and combine it with the corresponding feature map from the Encoder
(skip connection).

• Components: Upsampling (using bilinear interpolation) to increase the spatial size. A
convolution layer to reduce the number of channels. A ConvBlock to process the combined features
(from the upsampling and skip connection layer).

FinalOutput Class:

• Purpose: Generate the final output from the final feature map.

• Components: A convolution layer with the Tanh function. This reduces the number of output channels
to the same number of channels as the color image.

Unet Class:

• Purpose: Combine all the above components into a complete U-Net architecture.
• Components: Process the input image using FirstFeature and ConvBlock. Four Encoder layers with increasing number of channels, each layer continues to downsample and process the feature map. Four Decoder layers with decreasing number of channels, each layer increases in size, combining features from encoder (skip connection).

A FinalOutput layer to create the processed image.

• Forward: The input is processed through the initial convolutions. Then it is downsampled 4 times,
then upsampled 4 times each time combined with features from encoder. Finally, it goes through the final convolution layer to create the processed image.

## I. Task Super Resolution

1. Model requirements: Unet network using Skip Connection and without Skip Connection, then compare the results between the 2 models.

2. Implementation steps:
• Build a dataset from the original image (256x256x3). When loading the image, each sample needs 2 input and target images.

Input is: original image resized 4 times (64x64x3), target is the original image (256x256x3).

• Divide data into training and validation sets.
• Normalize data to match the activation of the last layer in the model to ensure the output has a value within the range of normal image values.
• Choose appropriate loss and metrics for the problem.
• Configure hyperparameters.
• Train and test results.
• We will build each .py file including:
- Library.py (file containing necessary libraries).
- Prepare_Dataset.py (file preparing dataset for model).
- Train_Val.py (file containing functions used for training and evaluating results)
- UnetNoSkipConnection.py (Unet model does not use SkipConnection).
- UnetSkipConnection.py (Unet model uses SkipConnection).
- 2 ipynb files to train the above 2 models using colab gpu.

3. Results after training.

For Unet model using Skip Connection:

![image](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/8f614901-dd5e-4e58-a967-2c89e24f0ac0)

![Screenshot 2024-02-19 203758](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/6968020c-3ed0-4fbb-beb0-569675bf3595)

For Unet model not using Skip Connection:


![image](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/127647215/59c3c58d-6aaa-4b9b-9819-3ce29874192e)


![Screenshot 2024-02-19 204058](https://github.com/PhamTrinhDuc/Unet-Architecture/assets/1