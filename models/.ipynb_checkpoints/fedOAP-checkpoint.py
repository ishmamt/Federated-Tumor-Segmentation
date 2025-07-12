import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CROSS_ATTENTION = True
USE_SPATIAL_ADAPTER = True

class CrossAttention(nn.Module):
    def __init__(self, in_channels, attn_dim=None, num_heads=8):
        super(CrossAttention, self).__init__()
        if attn_dim is None:
            attn_dim = in_channels

        self.num_heads = num_heads
        self.query = nn.Conv2d(in_channels, attn_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attn_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels, attn_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(attn_dim, in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(32, in_channels)

    def forward(self, query_input, key_value_input):
        B, C, H, W = query_input.shape
        Q = self.query(query_input).view(B, -1, H * W)
        K = self.key(key_value_input).view(B, -1, H * W)
        V = self.value(key_value_input).view(B, -1, H * W)

        attn_scores = torch.bmm(Q.permute(0, 2, 1), K) / (K.shape[1] ** 0.5)
        attn_map = self.softmax(attn_scores)

        attn_out = torch.bmm(V, attn_map.permute(0, 2, 1))
        attn_out = attn_out.view(B, -1, H, W)

        attn_out = self.out_conv(attn_out)
        return self.norm(attn_out + query_input)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SpatialAdapter(nn.Module):
    def __init__(self, in_channels, resType='res3'):
        super(SpatialAdapter, self).__init__()
        self.resType = resType
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels, affine=True)
        )
        if self.resType != 'none':
            self.residual = (
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                    nn.InstanceNorm2d(in_channels, affine=True),
                ) if self.resType == 'res3' else nn.Identity()
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_out = self.conv(x)
        if self.resType != 'none':
            res = self.residual(x)
            return self.act(conv_out+res) 
        return self.act(conv_out)

class UNetWithCrossAttention(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNetWithCrossAttention, self).__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bottleneckQ = Down(512, 1024)
        self.bottleneckKV = Down(512, 1024)

        self.cross_attn = CrossAttention(1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.adapter = SpatialAdapter(64)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneckQ(x4)
        

        if USE_CROSS_ATTENTION:
          x6 = self.bottleneckKV(x4)
          x5 = self.cross_attn(x5, x6)  # cross-attention between bottleneck and deepest skip
        #Even if we don't use cross attention we can use bottleneckQ as a simple bottleneck layer  
        x = self.up1(x5,x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if USE_SPATIAL_ADAPTER:
          x = self.adapter(x)
          
        return self.outc(x)

# --- Example test run ---
if __name__ == "__main__":
    model = UNetWithCrossAttention(in_channels=1, num_classes=1)
    x = torch.randn(1, 1, 256, 256)
    out = model(x)
    print(out.shape)  # Should be [1, 1, 256, 256]
