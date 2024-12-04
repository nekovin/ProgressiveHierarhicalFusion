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
        
        self.fusion_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_features * 8, base_features * 8, 1),
                nn.Sigmoid()
            ) for _ in range(n_fusion_levels)
        ])
        
        self.up1 = Up(base_features * 8, base_features * 4)
        self.up2 = Up(base_features * 4, base_features * 2)
        self.up3 = Up(base_features * 2, base_features)
        self.outc = nn.Conv2d(base_features, n_channels, kernel_size=1)
        
        self.fusion_weights = nn.Parameter(torch.ones(n_fusion_levels) / n_fusion_levels)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, n_fusion_levels, C, H, W]
        """
        batch_size = x.shape[0]
        
        features_per_level = []
        for i in range(self.n_fusion_levels):
            curr_x = x[:, i, :, :, :]  # [B, C, H, W]
            
            x1 = self.inc(curr_x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            
            x4 = x4 * self.fusion_attention[i](x4)
            
            features_per_level.append((x1, x2, x3, x4))
        
        norm_weights = self.softmax(self.fusion_weights)
        
        combined_features = []
        for i in range(4):  # For each encoder level
            weighted_sum = sum(
                w * features_per_level[j][i] 
                for j, w in enumerate(norm_weights)
            )
            combined_features.append(weighted_sum)
        
        # Decoder path with combined features
        x = self.up1(combined_features[3], combined_features[2])
        x = self.up2(x, combined_features[1])
        x = self.up3(x, combined_features[0])
        x = self.outc(x)
        
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

    def forward(self, x):
        return self.double_conv(x)

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
        
        # Handle cases where dimensions don't match perfectly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#def create_progressive_fusion_unet(n_fusion_levels=8):
    #return ProgressiveFusionUNet(n_channels=1, n_fusion_levels=n_fusion_levels)