import torch
import torch.nn as nn
import torch.nn.functional as F

class ProgressiveFusionUNet(nn.Module):
    def __init__(self, n_channels=1, n_fusion_levels=8):
        super().__init__()
        self.n_channels = n_channels
        self.n_fusion_levels = n_fusion_levels
        base_features = 64
        
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)

        self.fusion_attention = nn.Sequential(
            nn.Conv2d(base_features * 8, base_features * 8, 1),
            nn.Sigmoid()
        )
        
        self.up1 = Up(base_features * 8, base_features * 4)
        self.up2 = Up(base_features * 4, base_features * 2)
        self.up3 = Up(base_features * 2, base_features)

        '''This needs experimentation'''
        self.outc = nn.Conv2d(base_features, n_channels, kernel_size=1)
        
        self.fusion_weights = nn.Parameter(torch.ones(n_fusion_levels) / n_fusion_levels)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        input_image = x
        
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Deep feature refinement with attention
        x4 = x4 * self.fusion_attention(x4)
        
        '''
        norm_weights = self.softmax(self.fusion_weights)
        x4 = x4 * norm_weights[3]
        x3 = x3 * norm_weights[2]
        x2 = x2 * norm_weights[1]
        x1 = x1 * norm_weights[0]'''
        
        # Decoder path
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        #x = x + input_image
        x = 0.8 * x + 0.2 * input_image
        
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.double_conv(x) + self.residual(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

def create_progressive_fusion_unet(n_fusion_levels=8):
    return ProgressiveFusionUNet(n_channels=1, n_fusion_levels=n_fusion_levels)