import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels, attn_dim=None, num_heads=8):
        super(SelfAttention, self).__init__()
        if attn_dim is None:
            attn_dim = in_channels  # Default to same as input channels

        self.num_heads = num_heads
        self.query = nn.Conv2d(in_channels, attn_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attn_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels, attn_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(attn_dim, in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(32, in_channels)  # Use GroupNorm for stability

    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.query(x).view(B, -1, H * W)  # (B, attn_dim, H*W)
        K = self.key(x).view(B, -1, H * W)    # (B, attn_dim, H*W)
        V = self.value(x).view(B, -1, H * W)  # (B, attn_dim, H*W)

        attn_scores = torch.bmm(Q.permute(0, 2, 1), K) / (K.shape[1] ** 0.5)  # (B, H*W, H*W)
        attn_map = self.softmax(attn_scores)  # (B, H*W, H*W)

        attn_out = torch.bmm(V, attn_map.permute(0, 2, 1))  # (B, attn_dim, H*W)
        attn_out = attn_out.view(B, -1, H, W)  # (B, attn_dim, H, W)

        attn_out = self.out_conv(attn_out)  # Map back to original channels
        attn_out = self.norm(attn_out + x)  # Residual connection with normalization
        return attn_out

class DoubleConv(nn.Module):
    """(Conv2d -> IN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True), #Using Instance and Group Norm since in fed we dont have running statistics
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool followed by DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """Upscaling followed by DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Padding if needed (for odd dimensions)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)  # Concatenate skip connection
        return self.conv(x)

class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNetWithAttention, self).__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.attn = SelfAttention(1024)  # Self-attention after bottleneck

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.attn(x5)  # Apply self-attention after bottleneck

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

# Example usage
if __name__ == "__main__":
    model = UNetWithAttention(in_channels=3, num_classes=1)
    x = torch.randn(1, 3, 256, 256)  # Sample input
    y = model(x)
    print(y.shape)  # Should be (1, num_classes, 256, 256)
