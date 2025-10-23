"""UNet model implementation.

This module defines a lightweight UNet architecture. It avoids importing heavy
dependencies (torch) at module import time so tools can inspect the code without
having torch installed. Runtime functions import torch lazily when needed.
"""

from typing import Optional


def _lazy_torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required to construct or run the UNet model") from exc


# Purpose: create a small conv block used across the network
class FirstFeature:
    def __init__(self, in_channels: int, out_channels: int):
        _, nn = _lazy_torch()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.module(x)


# Mục đích: trích xuất đặc trưng
# Thành phần: tạo hai nhóm Conv-BatchNorm-LeakyReLU,
# khối này là khối cơ bản trong Unet được sử dụng trong encoder và decoder
class ConvBlock:
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


# Mục đích: Để giảm size của feature map và trích xuất các high-level feature.
# Thành phần: Một lớp Max Pooling theo sau là ConvBlock. Max Pooling giảm kích thước
# xuống một nửa, trong khi ConvBlock xử lý các đặc trưng.

class Encoder:
    def __init__(self, in_channels: int, out_channels: int):
        _, nn = _lazy_torch()
        self.module = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, x):
        return self.module(x)


# Mục đích: Để tăng kích thước feature map và kết hợp với feature map tương ứng từ Encoder
# (skip connection).
# Thành phần: Upsampling (sử dụng nội suy bilinear) để tăng kích thước không gian. Một
# lớp convolution để giảm số lượng channel. Một ConvBlock để xử lý các feature được ghép
# (từ lớp upsampling và skip connection).
class Decoder:
    def __init__(self, in_channels: int, out_channels: int):
        import torch as _torch
        _, nn = _lazy_torch()
        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        import torch as _torch

        x = self.decoder(x)
        x = _torch.concat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


# Mục đích: Tạo ra đầu ra cuối cùng từ feature map cuối cùng.
# Thành phần: Một lớp convolution với hàm Tanh. Điều này giảm số lượng channel đầu ra
# xuống bằn số lượng channel của ảnh màu
class FinalOutput:
    def __init__(self, in_channels: int, out_channels: int):
        _, nn = _lazy_torch()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.module(x)


class Unet:
    def __init__(self, n_channels: int = 1, num_classes: int = 3):
        # Instances allocate modules lazily via the small wrapper classes above.
        self.n_channels = n_channels
        self.num_classes = num_classes

        self.conv_start = FirstFeature(self.n_channels, 64)
        self.conv = ConvBlock(64, 64)
        self.encoder1 = Encoder(64, 128)
        self.encoder2 = Encoder(128, 256)
        self.encoder3 = Encoder(256, 512)
        self.encoder4 = Encoder(512, 1024)

        self.decoder1 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder4 = Decoder(128, 64)

        self.conv_end = FinalOutput(64, self.num_classes)

    def forward(self, x):
        x = self.conv_start.forward(x)
        x1 = self.conv.forward(x)

        x2 = self.encoder1.forward(x1)
        x3 = self.encoder2.forward(x2)
        x4 = self.encoder3.forward(x3)
        x5 = self.encoder4.forward(x4)

        x = self.decoder1.forward(x5, x4)
        x = self.decoder2.forward(x, x3)
        x = self.decoder3.forward(x, x2)
        x = self.decoder4.forward(x, x1)
        x = self.conv_end.forward(x)
        return x


if __name__ == "__main__":
    # Quick smoke test that raises a clear error if torch is missing.
    torch, nn = _lazy_torch()
    model = Unet(3, 3)
    input = torch.ones(2, 3, 128, 128)
    out = model.forward(input)
    print(out.shape)
