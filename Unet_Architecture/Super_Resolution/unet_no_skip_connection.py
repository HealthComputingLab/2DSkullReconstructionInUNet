from Unet_Architecture.Super_Resolution.library import (
    torch,
    nn,
    summary,
)

# Placeholder for the skipless UNet implementation in snake_case filename

class UnetNoSkip(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UnetNoSkip, self).__init__()
        # implement encoder/decoder without skip connections as originally present
        pass

if __name__ == '__main__':
    print('unet_no_skip_connection module ready')
