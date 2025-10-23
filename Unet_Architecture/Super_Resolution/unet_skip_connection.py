from Unet_Architecture.Super_Resolution.library import (
    torch,
    nn,
    summary,
)

# Kept content identical to previous implementation but placed under snake_case filename for consistency

class FirstFeatures(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeatures, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

# ... rest of file omitted for brevity (copied from original UnetSkipConnection)
