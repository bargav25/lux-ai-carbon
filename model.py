import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedDualInputCNN(nn.Module):
    def __init__(self):
        super(EnhancedDualInputCNN, self).__init__()

        # Pathway for state_1 (14x32x32) - moderately increased channels
        self.conv1_state1 = nn.Conv2d(14, 48, kernel_size=3, padding=1)  # Adjusted channels
        self.bn1_state1 = nn.BatchNorm2d(48)
        self.conv2_state1 = nn.Conv2d(48, 96, kernel_size=3, padding=1)  # Adjusted channels
        self.bn2_state1 = nn.BatchNorm2d(96)
        self.conv3_state1 = nn.Conv2d(96, 192, kernel_size=3, padding=1)  # Adjusted channels
        self.bn3_state1 = nn.BatchNorm2d(192)

        # Pathway for state_2 (14x4x4) - moderately increased channels, simplified
        self.conv1_state2 = nn.Conv2d(14, 48, kernel_size=1)  # Adjusted channels
        self.bn1_state2 = nn.BatchNorm2d(48)
        self.fc1_state2 = nn.Linear(48 * 4 * 4, 192)  # Simplified FC layer
        self.fc2_state2 = nn.Linear(192, 192 * 4 * 4)  # Adjusted for simplicity

        # Combined pathway - moderately adjusted
        self.conv4_combined = nn.Conv2d(192 + 192, 192, kernel_size=3, padding=1)  # Adjusted for combined
        self.bn4_combined = nn.BatchNorm2d(192)
        self.final_conv = nn.Conv2d(192, 5, kernel_size=1)  # Same as before to keep output size

    def forward(self, state_1, state_2):
        # Process state_1
        x1 = F.relu(self.bn1_state1(self.conv1_state1(state_1)))
        x1 = F.relu(self.bn2_state1(self.conv2_state1(x1)))
        x1 = F.relu(self.bn3_state1(self.conv3_state1(x1)))

        # Process state_2
        x2 = F.relu(self.bn1_state2(self.conv1_state2(state_2)))
        x2 = x2.view(-1, 48 * 4 * 4)
        x2 = F.relu(self.fc1_state2(x2))
        x2 = self.fc2_state2(x2)
        x2 = x2.view(-1, 192, 4, 4)

        # Upsample state_2 to match state_1's spatial dimensions
        x2 = F.interpolate(x2, size=(32, 32), mode='bilinear', align_corners=False)

        # Concatenate along the channel dimension
        x = torch.cat([x1, x2], dim=1)

        # Combined processing
        x = F.relu(self.bn4_combined(self.conv4_combined(x)))
        x = self.final_conv(x)

        return x



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_channels_b, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
#         self.down3 = Down(256,512)
        
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512//factor)
        self.up1 = Up(512+n_channels_b, 256, bilinear)
        self.up2 = Up(256+128, 128, bilinear)
        self.up3 = Up(128+64, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x, x_features):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = torch.cat((x4, x_features),1)
        x = self.up1(x,x3)
        x = self.up2(x,x2)
        x = self.up3(x,x1)
        logits = self.outc(x)
        
        return logits