# Model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

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

class UNET(nn.Module):
    def __init__(
            self, in_channels=18, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)
        # outputs = torch.sigmoid(x)
        outputs = x
        return outputs

class FCN(nn.Module):
    def __init__(self, in_channel = 18):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=9, stride=1, padding=(4,4), bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=(3,3), bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=9, stride=1, padding=(4,4), bias=True)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=(3,3), bias=True)
        self.bn4 = nn.BatchNorm2d(32)
        
        # self.tran5 = nn.ConvTranspose2d(32,32, kernel_size=9, stride=2, padding=(4,4), bias=True)
        self.tran5 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=2, padding=4, output_padding=(1, 1), bias=True)
        self.bn5 = nn.BatchNorm2d(32)
        
        self.conv6 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=(2,2), bias=True)
        self.bn6 = nn.BatchNorm2d(16)

        # self.tran7 = nn.ConvTranspose2d(16,4, kernel_size=5, stride=2, padding=(2,2), bias=True)
        self.tran7 = nn.ConvTranspose2d(16, 4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        self.bn7 = nn.BatchNorm2d(4)
        
        # self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=(1,1), bias=True)

    def forward(self, x):
        if len(x.size()) > 3:
            output_size = (x.size(0), 1, x.size(2), x.size(3))
        else:
            output_size = (1, x.size(1), x.size(2))
        # print(x.size())
        out = self.conv1(x)
        out = self.bn1(out)
        
        residual = self.pool1(out)
        
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.pool2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        
        # out = self.tran5(out, output_size=residual.size())
        out = self.tran5(out)
        out = self.bn5(out)
        
        if out.size() != residual.size():
            out = TF.resize(out, size=residual.size()[2:], antialias=True)
        out += residual
        
        out = self.conv6(out)
        out = self.bn6(out)
        
        # out = self.tran7(out, output_size=output_size)
        out = self.tran7(out)
        out = self.bn7(out)
        
        out = self.conv8(out)
        ## Ensure width and height of out is same as input
        ## First check if width and height are same
        if out.size() != output_size:
            out = TF.resize(out, size=output_size[2:], antialias=True)
        
        return out

def test():
    x = torch.randn((3, 1, 100, 267))
    model = FCN(in_channel=1)
    preds = model(x)
    # print(preds.shape)
    # print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()