import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self,in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2
                    ),
                    DoubleConv(in_channels=feature*2, out_channels=feature)
                )
            )

        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1]*2)
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            # print(x.shape)
            x = self.pool(x)

        x = self.bottleneck(x)
        # print(x.shape)
        skip_connections = skip_connections[::-1]
        # print(len(skip_connections))
        # print(len(self.ups))
        for up, skip_connection in zip(self.ups, skip_connections):
            x2 = skip_connection
            x1 = up[0](x)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY //2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            # print(x.shape)
            x = up[1](x)

        x = self.final_conv(x)
        # print(x.shape)
        return x


# if __name__ == '__main__':
#     model = UNet(in_channels=1, out_channels=1)
#     image = torch.rand((3, 1, 161, 161))
#     x = model(image)
#     assert x.shape == image.shape