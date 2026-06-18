"""
baselines.py - Top-Journal Level Medical Segmentation Baseline Model Library (Full Dimension Fixed Version)
Included: UNet, Attention-UNet, UNet++, TransUNet, nnU-Net, Swin-UNet, MobileUNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# ===================== Common Basic Modules =====================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        attn_weight = self.psi(psi)
        return x * attn_weight
# ===================== 1. Original UNet (2015) =====================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.bottle = ConvBlock(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.d4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottle(self.pool(e4))
        
        d4 = self.d4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)
# ===================== 2. Original Attention-UNet (2018) =====================
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.bottle = ConvBlock(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.ag4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.d4 = ConvBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ag3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.d3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ag2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.d2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ag1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.d1 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottle(self.pool(e4))
        
        d4_up = self.up4(b)
        d4 = self.d4(torch.cat([d4_up, self.ag4(g=d4_up, x=e4)], dim=1))
        d3_up = self.up3(d4)
        d3 = self.d3(torch.cat([d3_up, self.ag3(g=d3_up, x=e3)], dim=1))
        d2_up = self.up2(d3)
        d2 = self.d2(torch.cat([d2_up, self.ag2(g=d2_up, x=e2)], dim=1))
        d1_up = self.up1(d2)
        d1 = self.d1(torch.cat([d1_up, self.ag1(g=d1_up, x=e1)], dim=1))
        return self.out(d1)
# ===================== 3. UNet++ (2018) =====================
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv0_0 = ConvBlock(in_channels, 32)
        self.conv1_0 = ConvBlock(32, 64)
        self.conv2_0 = ConvBlock(64, 128)
        self.conv3_0 = ConvBlock(128, 256)
        self.conv4_0 = ConvBlock(256, 512)
        
        self.up1_0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv0_1 = ConvBlock(64, 32)
        self.up2_0 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv1_1 = ConvBlock(128, 64)
        self.up3_0 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2_1 = ConvBlock(256, 128)
        
        self.up1_1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv0_2 = ConvBlock(96, 32)
        self.up2_1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv1_2 = ConvBlock(192, 64)
        
        self.up1_2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv0_3 = ConvBlock(128, 32)
        
        self.up4_0 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv3_1 = ConvBlock(512, 256)
        self.up3_1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2_2 = ConvBlock(384, 128)
        self.up2_2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv1_3 = ConvBlock(256, 64)
        self.up1_3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv0_4 = ConvBlock(160, 32)
        
        self.final = nn.Conv2d(32, out_channels, 1)
        
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))
        return self.final(x0_4)
# ===================== 4. TransUNet (2021) =====================
class TransUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.cnn_e1 = ConvBlock(in_channels, 64)
        self.cnn_e2 = ConvBlock(64, 128)
        self.cnn_e3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        self.vit_embed = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.GroupNorm(8, 512)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2, 
            activation='gelu', batch_first=True, norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3 = ConvBlock(256 + 256, 256) 
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2 = ConvBlock(128 + 128, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1 = ConvBlock(64 + 64, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        e1 = self.cnn_e1(x)
        e2 = self.cnn_e2(self.pool(e1))
        e3 = self.cnn_e3(self.pool(e2))
        
        vit_in = self.vit_embed(self.pool(e3)) 
        B, C, H, W = vit_in.shape
        vit_in_flat = vit_in.flatten(2).transpose(1, 2) 
        
        vit_out_flat = self.transformer(vit_in_flat) + vit_in_flat
        vit_out = vit_out_flat.transpose(1, 2).view(B, C, H, W)
        
        d3 = self.d3(torch.cat([self.up3(vit_out), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)
# ===================== 5. nnU-Net (2021) =====================
class nnUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.e4 = ConvBlock(128, 256)
        self.e5 = ConvBlock(256, 512)
        self.bottle = ConvBlock(512, 1024)
        
        self.up5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.d5 = ConvBlock(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, out_channels, 1)
        
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.e5(self.pool(e4))
        b = self.bottle(self.pool(e5))
        
        d5 = self.d5(torch.cat([self.up5(b), e5], dim=1))
        d4 = self.d4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)
# ===================== 6. Swin-UNet (2021) - [Dimension Skip Connection Bug Fully Fixed] =====================
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows 
def window_reverse(windows, window_size, H, W):
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
    return x 
class WindowAttentionBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        
        self.norm1 = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.GroupNorm(8, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        x_windows = window_partition(x_norm, self.window_size)
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
        x_attn = window_reverse(attn_windows, self.window_size, H, W)
        x = shortcut + x_attn
        
        x_norm2 = self.norm2(x)
        x_mlp = self.mlp(x_norm2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + x_mlp
        return x
class SwinUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        
        self.swin_bottle = WindowAttentionBlock(dim=256, window_size=8, num_heads=8)
        
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.d3 = ConvBlock(256 + 256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2 = ConvBlock(128 + 128, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1 = ConvBlock(64 + 64, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        
        b = self.swin_bottle(self.pool(e3))
        
        d3 = self.d3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)
# ===================== 7. MobileUNet (2020) - Lightweight =====================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)
class MobileUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = DepthwiseSeparableConv(in_channels, 32)
        self.e2 = DepthwiseSeparableConv(32, 64)
        self.e3 = DepthwiseSeparableConv(64, 128)
        self.e4 = DepthwiseSeparableConv(128, 256)
        self.bottle = DepthwiseSeparableConv(256, 512)
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d4 = DepthwiseSeparableConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = DepthwiseSeparableConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = DepthwiseSeparableConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = DepthwiseSeparableConv(64, 32)
        
        self.out = nn.Conv2d(32, out_channels, 1)
        
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottle(self.pool(e4))
        
        d4 = self.d4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)