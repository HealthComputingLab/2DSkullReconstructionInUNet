import torch
import torch.nn as nn
from torchvision import transforms


class FirstFeatureNoSkip(nn.Module):
    """Initial feature extraction layer using 1x1 convolution."""
    def __init__(self, in_channels, out_channels):
        super(FirstFeatureNoSkip, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlockNoSkip(nn.Module):
    """
    Basic convolutional block for feature extraction.
    
    Two Conv-BatchNorm-LeakyReLU sequences for processing features.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlockNoSkip, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderNoSkip(nn.Module):
    """
    Encoder block for downsampling and extracting high-level features.
    
    Uses MaxPooling to reduce spatial dimensions by half, followed by
    feature extraction via ConvBlock.
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderNoSkip, self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlockNoSkip(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class DecoderNoSkip(nn.Module):
    """
    Decoder block for upsampling feature maps.
    
    Note: Uses out_channels * 2 to balance weights as if skip connections
    were present (for architectural consistency with skip-connection variants).
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderNoSkip, self).__init__()
        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            # out_channels * 2: to balance weights with skip connection architecture
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_block = ConvBlockNoSkip(out_channels * 2, out_channels)

    def forward(self, x):
        x = self.decoder(x)
        x = self.conv_block(x)
        return x


class FinalFeatureNoSkip(nn.Module):
    """
    Final output layer producing reconstructed image.
    
    Uses Tanh activation to output values in [-1, 1] range.
    """
    def __init__(self, in_channels, out_channels):
        super(FinalFeatureNoSkip, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)


class UnetNoSkip(nn.Module):
    """
    UNet architecture without skip connections for super-resolution.
    
    The input image is downsampled 4x, so we upsample 4x during forward pass
    to match the original resolution for training. Final output is compared
    against the original high-resolution image.
    """
    def __init__(self, in_channels, num_classes, img_height, img_width):
        super(UnetNoSkip, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Upsample input 4x since it was downsampled for low-resolution input
        # After reconstruction, compare with original unresized image
        self.resize_trueForm = transforms.Resize(
            (img_height * 4, img_width * 4), antialias=True
        )
        
        self.conv1 = ConvBlockNoSkip(in_channels, 64)
        self.conv2 = ConvBlockNoSkip(64, 64)

        self.encoder1 = EncoderNoSkip(64, 128)
        self.encoder2 = EncoderNoSkip(128, 256)
        self.encoder3 = EncoderNoSkip(256, 512)
        self.encoder4 = EncoderNoSkip(512, 1024)

        self.decoder1 = DecoderNoSkip(1024, 512)
        self.decoder2 = DecoderNoSkip(512, 256)
        self.decoder3 = DecoderNoSkip(256, 128)
        self.decoder4 = DecoderNoSkip(128, 64)

        self.out_conv = FinalFeatureNoSkip(64, num_classes)

    def forward(self, x):
        x = self.resize_trueForm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.out_conv(x)
        return x


if __name__ == "__main__":
    in_channels, num_classes = 3, 3
    model = UnetNoSkip(in_channels, num_classes, 64, 64)
    input = torch.ones(2, 3, 64, 64)
    output = model(input)
    print(output.shape)
