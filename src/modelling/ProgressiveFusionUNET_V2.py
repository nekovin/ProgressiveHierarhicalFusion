import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """double conv with optional residual connection"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # handle any size mismatches
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ProgressiveFusionUNet(nn.Module):
    """progressive fusion u-net for oct denoising"""
    def __init__(self, n_channels: int = 1, n_fusion_levels: int = 8) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_fusion_levels = n_fusion_levels
        
        # network parameters
        self.base_features = 32
        
        # encoder pathway
        self.inc = DoubleConv(n_channels, self.base_features)
        self.down1 = Down(self.base_features, self.base_features * 2)
        self.down2 = Down(self.base_features * 2, self.base_features * 4)
        self.down3 = Down(self.base_features * 4, self.base_features * 8)
        
        # fusion attention mechanism
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(self.base_features * 8, self.base_features * 8, kernel_size=1),
            nn.Sigmoid()
        )
        
        # decoder pathway
        self.up1 = Up(self.base_features * 8, self.base_features * 4)
        self.up2 = Up(self.base_features * 4, self.base_features * 2)
        self.up3 = Up(self.base_features * 2, self.base_features)
        
        # output layer
        self.outc = nn.Conv2d(self.base_features, n_channels, kernel_size=1)
        
        # learnable parameters
        self.fusion_weights = nn.Parameter(torch.ones(n_fusion_levels) / n_fusion_levels)
        self.residual_weight = nn.Parameter(torch.tensor(0.2))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # store input for residual connection
        input_image = x
        
        # encoder pathway with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # apply fusion attention
        x4 = x4 * self.fusion_attention(x4)
        
        # decoder pathway with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # final output with residual connection
        x = self.outc(x)
        x = (1 - self.residual_weight) * x + self.residual_weight * input_image
        
        return x


def create_progressive_fusion_unet(n_fusion_levels: int = 8) -> ProgressiveFusionUNet:
    """factory function to create progressive fusion unet"""
    return ProgressiveFusionUNet(n_channels=1, n_fusion_levels=n_fusion_levels)