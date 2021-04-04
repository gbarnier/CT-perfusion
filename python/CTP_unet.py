import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Implementation of the UNet paper
# https://www.youtube.com/watch?v=IHq1t7NxS8k&t=211s

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        # Call parent class constructor
        super(DoubleConv, self).__init__()

        # Define one conv block
        f = 3 # Filter size
        s = 1
        p = 1 # => 'same' convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

    # Define forward pass
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):

        # Call parent class constructor
        super(UNET, self).__init__()

        # List for encoding part
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck
        print("features[:-1]: ", features[:-1])
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final convolutional layer
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward():
        skip_connections = []

        # Encoding part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse the skip connection list
        skip_connections = skip_connections[::-1]

        # Decoding part
        for idx in range(0, len(self.ups), 2):

            # Apply transpose
            x = self.ups[idx](x)

            # Concatenate along the channel dimension
            # The input shape is (batch, channel, height, width)
            assert(len(self.ups)%2 == 0)
            skip_connection = skip_connections[idx/2]

            # Check case where the upsampling has a different shape
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)

            # Apply double conv
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():

    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    print("model: ", model)
    # preds = model(x)
    # print(preds.shape)
    # print(x.shape)
    # assert preds.shape == x.shape

if __name__ == "__main__":
    test()
