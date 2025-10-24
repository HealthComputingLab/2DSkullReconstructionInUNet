"""UNet skip-connection variant for super-resolution.

This file avoids importing torch at module import time. Constructing model
objects will import torch lazily to allow code inspection without PyTorch installed.

Architecture:
- Input: Low-resolution images (downsampled 4x)
- Process: Upsample to 4x size, then reconstruct through UNet with skip connections
- Output: High-resolution reconstructed images
- Skip connections preserve fine-grained spatial details during upsampling
"""

from Unet_Architecture.Super_Resolution import library as lib
from torchvision import transforms


class FirstFeatures:
    """
    Initial feature extraction layer using 1x1 convolution.
    
    Purpose: Create lightweight initial feature representation from input.
    Uses small kernel size for efficient computation.
    """
    def __init__(self, in_channels, out_channels):
        import torch
        import torch.nn as nn

        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.module(x)


class ConvBlock:
    """
    Fundamental convolutional block for feature extraction.
    
    Purpose: Extract hierarchical features through double convolution.
    Architecture: Conv-BN-LeakyReLU repeated twice
    
    This is the core building block used throughout the encoder and decoder
    to progressively extract and refine features.
    """
    def __init__(self, in_channels, out_channels):
        import torch
        import torch.nn as nn

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder:
    """
    Encoder block for downsampling and feature extraction.
    
    Purpose: Reduce spatial dimensions while increasing feature depth.
    Components:
    - MaxPool2d: Reduces height and width by half (downsampling)
    - ConvBlock: Extracts features at the new resolution
    
    This creates a hierarchical feature representation where deeper layers
    capture more abstract, high-level features.
    """
    def __init__(self, in_channels, out_channels):
        import torch.nn as nn

        self.encoder = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, x):
        return self.encoder(x)


class Decoder:
    """
    Decoder block for upsampling with skip connections.
    
    Purpose: Increase spatial dimensions while incorporating fine details
    from corresponding encoder layers via skip connections.
    
    Components:
    - Bilinear upsampling: Doubles spatial dimensions
    - 1x1 convolution: Adjusts channel count
    - ConvBlock: Processes concatenated features from upsampling + skip connection
    
    Skip connections are crucial for super-resolution as they preserve
    high-frequency details lost during downsampling.
    """
    def __init__(self, in_channels, out_channels):
        import torch
        import torch.nn as nn

        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        # in_channels + out_channels: accounts for concatenation with skip connection
        self.conv_block = ConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        """
        Forward pass with skip connection integration.
        
        Args:
            x: Upsampled features from previous decoder layer
            skip: Features from corresponding encoder layer (skip connection)
            
        The skip connection concatenates encoder features with decoder features,
        allowing the network to combine high-level semantic information with
        low-level spatial details for better reconstruction quality.
        """
        import torch

        x = self.decoder(x)
        x = self.conv_block(torch.cat([x, skip], dim=1))
        return x


class FinalFeatures:
    """
    Final output layer producing the super-resolved image.
    
    Purpose: Generate final high-resolution output from feature maps.
    Components:
    - 1x1 convolution: Reduces channels to match output image channels
    - Tanh activation: Outputs values in [-1, 1] range (matches normalized targets)
    """
    def __init__(self, in_channels, out_channels):
        import torch.nn as nn

        self.end_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.end_conv(x)


class Unet:
    """
    UNet architecture for image super-resolution with skip connections.
    
    Pipeline:
    1. Input: Low-resolution image (downsampled 4x from original)
    2. Upsample input to target size (4x upsampling)
    3. Encoder path: Extract hierarchical features while downsampling
    4. Decoder path: Reconstruct high-resolution image with skip connections
    5. Output: High-resolution image matching original resolution
    
    Architecture details:
    - Encoder: 4 stages (64 -> 128 -> 256 -> 512 -> 1024 channels)
    - Decoder: 4 stages (1024 -> 512 -> 256 -> 128 -> 64 channels)
    - Skip connections at each resolution level preserve spatial information
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output channels (typically same as input)
        img_height: Height of low-resolution input
        img_width: Width of low-resolution input
    """
    def __init__(self, in_channels, num_classes, img_height, img_width):
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Resize to expected output scale (4x upsampling)
        # Input is 4x smaller than target, so we upsample before processing
        # Final output will be compared with original high-res image
        self.resize_trueform = transforms.Resize((img_height * 4, img_width * 4))
        
        # Initial feature extraction
        self.first_features = FirstFeatures(in_channels, 64)
        self.conv1 = ConvBlock(64, 64)
        
        # Encoder path: progressive downsampling
        self.encoder1 = Encoder(64, 128)    # 1/2 resolution
        self.encoder2 = Encoder(128, 256)   # 1/4 resolution
        self.encoder3 = Encoder(256, 512)   # 1/8 resolution
        self.encoder4 = Encoder(512, 1024)  # 1/16 resolution (bottleneck)

        # Decoder path: progressive upsampling with skip connections
        self.decoder1 = Decoder(1024, 512)  # 1/8 resolution + skip from encoder3
        self.decoder2 = Decoder(512, 256)   # 1/4 resolution + skip from encoder2
        self.decoder3 = Decoder(256, 128)   # 1/2 resolution + skip from encoder1
        self.decoder4 = Decoder(128, 64)    # Full resolution + skip from conv1
        
        # Final output layer
        self.end_features = FinalFeatures(64, num_classes)

    def forward(self, x):
        """
        Forward pass through the super-resolution UNet.
        
        Process:
        1. Upsample low-res input to target resolution
        2. Extract features through encoder (save intermediate features)
        3. Reconstruct through decoder (merge with saved encoder features)
        4. Generate final high-resolution output
        
        The skip connections help preserve fine details that would otherwise
        be lost during the downsampling operations in the encoder.
        """
        # Upsample input to target resolution
        x = self.resize_trueform(x)
        
        # Initial feature extraction
        x = self.first_features(x)
        x1 = self.conv1(x)
        
        # Encoder path: downsample and extract features
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)  # Bottleneck (deepest features)

        # Decoder path: upsample and merge with skip connections
        x = self.decoder1(x5, x4)  # Merge with encoder3 output
        x = self.decoder2(x, x3)   # Merge with encoder2 output
        x = self.decoder3(x, x2)   # Merge with encoder1 output
        x = self.decoder4(x, x1)   # Merge with initial conv output
        
        # Generate final super-resolved output
        x = self.end_features(x)
        return x


if __name__ == "__main__":
    # Run a small smoke test if torch is available
    try:
        import torch

        # Test with 3-channel RGB images at 64x64 input resolution
        model = Unet(3, 3, 64, 64)
        input = torch.ones(2, 3, 64, 64)  # Batch of 2 low-res images
        output = model.forward(input)
        print(f"Output shape: {output.shape}")  # Should be (2, 3, 256, 256) - 4x upscaled
    except Exception as e:
        print(f"Torch not available — skipping smoke test: {e}")