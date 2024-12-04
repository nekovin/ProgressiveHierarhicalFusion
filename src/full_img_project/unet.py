import torch
import torch.nn as nn

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
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OCTUNet(nn.Module):
    def __init__(self, n_channels=1):
        super(OCTUNet, self).__init__()
        self.n_channels = n_channels
        
        base_features = 64

        self.inc = DoubleConv(n_channels, base_features)

        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)

        self.up1 = Up(base_features * 8, base_features * 4)
        self.up2 = Up(base_features * 4, base_features * 2)
        self.up3 = Up(base_features * 2, base_features)

        self.outc = nn.Conv2d(base_features, n_channels, kernel_size=1)

    def get_padding_indices(self, x):
        """Generate indices for the blind-spot padding as per N2V paper"""
        batch_size, channels, height, width = x.shape
        indices = torch.randint(0, height * width, (batch_size,))
        y_idx = indices // width
        x_idx = indices % width
        return y_idx, x_idx

    def mask_random_pixels(self, x):
        """Implement N2V blind-spot masking"""
        masked = x.clone()
        batch_size = x.shape[0]
        
        y_idx, x_idx = self.get_padding_indices(x)
        
        for i in range(batch_size):
            masked[i, :, y_idx[i], x_idx[i]] = 0
            
        return masked, (y_idx, x_idx)

    def forward(self, x, training=True):
        if training:
            x, indices = self.mask_random_pixels(x)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        x = self.outc(x)
        
        return x

def create_oct_unet(pretrained=False):
    model = OCTUNet(n_channels=1)
    
    if pretrained:
        pass
        
    return model