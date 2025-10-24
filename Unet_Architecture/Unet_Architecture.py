"""UNet model implementation.

This module defines a lightweight UNet architecture. It avoids importing heavy
dependencies (torch) at module import time so tools can inspect the code without
having torch installed. Runtime functions import torch lazily when needed.

UNet is a convolutional neural network architecture designed for image-to-image tasks,
particularly effective for image segmentation, inpainting, and restoration. The U-shaped
architecture consists of a contracting path (encoder) and an expansive path (decoder)
connected by skip connections.
"""

from typing import Optional


def _lazy_torch():
    """
    Lazy import of PyTorch to avoid import-time dependencies.
    
    This allows tools to inspect the code structure without requiring
    PyTorch to be installed. PyTorch is only imported when actually
    constructing or running the model.
    
    Returns:
        Tuple of (torch module, torch.nn module)
        
    Raises:
        RuntimeError: If PyTorch is not installed or cannot be imported
    """
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required to construct or run the UNet model") from exc


class FirstFeature:
    """
    Initial feature extraction layer using lightweight 1x1 convolution.
    
    Purpose: Create a small convolutional block to generate initial feature
    representation from the input image with minimal computational cost.
    
    Components:
    - 1x1 convolution: Adjusts channel dimensions without changing spatial size
    - LeakyReLU: Non-linear activation for feature learning
    """
    def __init__(self, in_channels: int, out_channels: int):
        _, nn = _lazy_torch()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.module(x)


class ConvBlock:
    """
    Fundamental convolutional block for hierarchical feature extraction.
    
    Purpose: Extract and refine features through double convolution sequences.
    This is the basic building block used throughout both the encoder and decoder
    paths of the UNet architecture.
    
    Components: Two Conv-BatchNorm-LeakyReLU sequences
    - First conv: Extracts initial features from input
    - BatchNorm: Normalizes activations for stable training
    - LeakyReLU: Non-linear activation allowing gradient flow
    - Second conv: Refines extracted features
    - Second BatchNorm + LeakyReLU: Further normalization and activation
    
    This double convolution pattern allows the network to learn complex
    hierarchical features at each resolution level.
    """
    def __init__(self, in_channels: int, out_channels: int):
        _, nn = _lazy_torch()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.module(x)


class Encoder:
    """
    Encoder block for downsampling and high-level feature extraction.
    
    Purpose: Reduce the spatial dimensions of feature maps while extracting
    progressively more abstract, high-level features. This creates a hierarchical
    feature representation where deeper layers capture semantic information.
    
    Components:
    - MaxPool2d: Downsamples by factor of 2 (reduces both height and width by half)
      * Selects maximum value in each 2x2 window
      * Provides translation invariance
      * Reduces computational cost in deeper layers
    - ConvBlock: Processes features at the new, reduced resolution
      * Extracts features appropriate for this scale
    
    The downsampling creates a pyramid of features at different scales,
    essential for understanding both local details and global context.
    """
    def __init__(self, in_channels: int, out_channels: int):
        _, nn = _lazy_torch()
        self.module = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, x):
        return self.module(x)


class Decoder:
    """
    Decoder block for upsampling and feature reconstruction with skip connections.
    
    Purpose: Increase spatial dimensions of feature maps and combine with
    corresponding encoder features (skip connections) to reconstruct fine-grained
    spatial details while maintaining semantic understanding.
    
    Components:
    - UpsamplingBilinear2d: Increases spatial dimensions by factor of 2
      * Uses bilinear interpolation for smooth upsampling
      * Doubles both height and width
    - 1x1 Convolution: Reduces channel count to prepare for concatenation
      * Adjusts feature dimensionality
      * Lightweight operation (1x1 kernel)
    - BatchNorm + LeakyReLU: Normalizes and activates upsampled features
    - ConvBlock: Processes concatenated features from upsampling and skip connection
      * Integrates high-level semantic information (from decoder path)
      * With low-level spatial details (from encoder skip connection)
    
    Skip Connections:
    The skip connections are crucial - they allow the network to combine:
    1. Semantic information from the decoder (what is in the image)
    2. Spatial details from the encoder (where things are located)
    This enables precise localization while maintaining context understanding.
    """
    def __init__(self, in_channels: int, out_channels: int):
        import torch as _torch
        _, nn = _lazy_torch()
        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        # Input channels = in_channels because we concatenate decoder and skip features
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        """
        Forward pass integrating skip connection from encoder.
        
        Args:
            x: Upsampled features from previous decoder layer (semantic information)
            skip: Features from corresponding encoder layer (spatial details)
            
        Process:
        1. Upsample the decoder features to match encoder resolution
        2. Concatenate with skip connection features along channel dimension
        3. Process combined features through ConvBlock
        
        The concatenation merges complementary information:
        - Decoder features contain high-level semantic understanding
        - Skip features contain fine-grained spatial details
        This combination enables accurate reconstruction with precise localization.
        """
        import torch as _torch

        x = self.decoder(x)
        x = _torch.concat([x, skip], dim=1)  # Concatenate along channel dimension
        x = self.conv_block(x)
        return x


class FinalOutput:
    """
    Final output layer for generating the reconstructed image.
    
    Purpose: Convert the final feature maps into the output image with the
    desired number of channels (e.g., 3 for RGB, 1 for grayscale).
    
    Components:
    - 1x1 Convolution: Reduces feature channels to match output image channels
      * Maps learned features to pixel values
      * Lightweight operation for final prediction
    
    Note: No activation function is applied here. The activation (if needed)
    should be applied externally (e.g., Sigmoid for [0,1] range, Tanh for [-1,1]).
    This provides flexibility for different tasks and loss functions.
    """
    def __init__(self, in_channels: int, out_channels: int):
        _, nn = _lazy_torch()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.module(x)


class Unet:
    """
    Standard UNet architecture with skip connections.
    
    UNet is a U-shaped encoder-decoder architecture designed for dense prediction
    tasks (pixel-level output for each input pixel). It's particularly effective for:
    - Image segmentation
    - Image inpainting
    - Image restoration
    - Medical image analysis
    
    Architecture Overview:
    
    Encoder Path (Contracting Path):
    - Progressively downsamples input through 4 encoder stages
    - Channel progression: 64 -> 128 -> 256 -> 512 -> 1024
    - Each stage halves spatial dimensions and doubles feature channels
    - Captures hierarchical features from low-level to high-level
    
    Bottleneck:
    - Deepest layer with smallest spatial size and most channels (1024)
    - Contains the most abstract, semantic feature representation
    
    Decoder Path (Expansive Path):
    - Progressively upsamples features through 4 decoder stages
    - Channel progression: 1024 -> 512 -> 256 -> 128 -> 64
    - Each stage doubles spatial dimensions and halves feature channels
    - Reconstructs spatial details while maintaining semantic understanding
    
    Skip Connections:
    - Connect each encoder stage to its corresponding decoder stage
    - Preserve fine-grained spatial information lost during downsampling
    - Enable precise localization in the output
    - Critical for maintaining both "what" (semantics) and "where" (localization)
    
    Args:
        n_channels: Number of input image channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output channels (typically same as input for reconstruction)
    """
    def __init__(self, n_channels: int = 1, num_classes: int = 3):
        # Instances allocate modules lazily via the small wrapper classes above
        self.n_channels = n_channels
        self.num_classes = num_classes

        # Initial feature extraction
        self.conv_start = FirstFeature(self.n_channels, 64)
        self.conv = ConvBlock(64, 64)
        
        # Encoder path: progressive downsampling with increasing feature depth
        self.encoder1 = Encoder(64, 128)    # 1/2 resolution, 128 channels
        self.encoder2 = Encoder(128, 256)   # 1/4 resolution, 256 channels
        self.encoder3 = Encoder(256, 512)   # 1/8 resolution, 512 channels
        self.encoder4 = Encoder(512, 1024)  # 1/16 resolution, 1024 channels (bottleneck)

        # Decoder path: progressive upsampling with decreasing feature depth
        self.decoder1 = Decoder(1024, 512)  # 1/8 resolution + skip from encoder3
        self.decoder2 = Decoder(512, 256)   # 1/4 resolution + skip from encoder2
        self.decoder3 = Decoder(256, 128)   # 1/2 resolution + skip from encoder1
        self.decoder4 = Decoder(128, 64)    # Full resolution + skip from conv

        # Final output generation
        self.conv_end = FinalOutput(64, self.num_classes)

    def forward(self, x):
        """
        Forward pass through the complete UNet architecture.
        
        Process:
        1. Initial feature extraction (conv_start, conv)
        2. Encoder path: Downsample 4 times, saving intermediate features
        3. Bottleneck: Deepest features (x5) at smallest spatial resolution
        4. Decoder path: Upsample 4 times, merging with saved encoder features
        5. Final output: Generate reconstructed image
        
        The U-shape is evident in the data flow:
        - Spatial dimensions: decrease in encoder, increase in decoder
        - Feature channels: increase in encoder, decrease in decoder
        - Skip connections: bridge the two paths at each resolution level
        
        Skip connection routing:
        - x1 (full res, 64ch) -> decoder4
        - x2 (1/2 res, 128ch) -> decoder3
        - x3 (1/4 res, 256ch) -> decoder2
        - x4 (1/8 res, 512ch) -> decoder1
        - x5 (1/16 res, 1024ch) -> bottleneck, fed to decoder1
        """
        # Initial convolutions
        x = self.conv_start.forward(x)
        x1 = self.conv.forward(x)

        # Encoder path: downsample and extract hierarchical features
        x2 = self.encoder1.forward(x1)
        x3 = self.encoder2.forward(x2)
        x4 = self.encoder3.forward(x3)
        x5 = self.encoder4.forward(x4)  # Bottleneck with deepest features

        # Decoder path: upsample and merge with skip connections
        x = self.decoder1.forward(x5, x4)  # Merge with encoder3 output
        x = self.decoder2.forward(x, x3)   # Merge with encoder2 output
        x = self.decoder3.forward(x, x2)   # Merge with encoder1 output
        x = self.decoder4.forward(x, x1)   # Merge with initial conv output
        
        # Generate final output
        x = self.conv_end.forward(x)
        return x


if __name__ == "__main__":
    # Quick smoke test that raises a clear error if torch is missing
    torch, nn = _lazy_torch()
    
    # Test with 3-channel RGB input at 128x128 resolution
    model = Unet(3, 3)
    input = torch.ones(2, 3, 128, 128)  # Batch of 2 images
    out = model.forward(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {out.shape}")  # Should match input shape: (2, 3, 128, 128)