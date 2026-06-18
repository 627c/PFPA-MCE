"""
net_seg_only.py 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class SEBlock(nn.Module):
    """Global channel attention, used to suppress random bright speckle noise unique to ultrasound images"""
    def __init__(self, ch, red=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // red, ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch)
        )
    def forward(self, x):
        return self.conv(x)
class MRAGate(nn.Module):
    """
    Core innovation module: Multi-scale Receptive Attention Gate
    Adaptively predicts and assigns control weights for three multi-scale multi-receptive field branches according to the size of acoustic dropout regions, achieving flexible dynamic fusion of dilated features.
    
    Ablation parameters:
    - dilations: list of dilation rates, controls receptive field size
    - use_dynamic_weight: whether to use dynamic weight predictor (False = fixed sum)
    """
    def __init__(self, F_g, F_l, F_int, dilations=[1, 3, 5], use_dynamic_weight=True):
        super().__init__()
        # Deep gating guidance signal mapping
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.dilations = dilations
        self.use_dynamic_weight = use_dynamic_weight
        self.num_branches = len(dilations)
        
        # Dynamically create multi-scale dilated convolution branches (supports any number of dilation rates)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Conv2d(F_l, F_int, kernel_size=3, padding=d, dilation=d, bias=False)
            )
        
        # Dynamically create weight predictor (only when branches > 1 and dynamic weight is enabled)
        if self.use_dynamic_weight and self.num_branches > 1:
            self.weight_predictor = nn.Sequential(
                nn.Conv2d(F_int * self.num_branches, self.num_branches, kernel_size=1, bias=True),
                nn.Softmax(dim=1)
            )
        
        self.W_x_bn = nn.BatchNorm2d(F_int)
        self.relu = nn.ReLU(inplace=True)
        
        # Spatial attention map mapping
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        
        # Extract features from all branches in parallel
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # Dynamically fuse branch features
        if self.use_dynamic_weight and self.num_branches > 1:
            # Concatenate all branch features
            x_cat = torch.cat(branch_outputs, dim=1)
            # Predict branch weights for each spatial position
            scale_weights = self.weight_predictor(x_cat)  # shape: [B, num_branches, H, W]
            # Weighted fusion
            x_fused = 0
            for i in range(self.num_branches):
                x_fused += scale_weights[:, i:i+1, :, :] * branch_outputs[i]
        else:
            # Fixed sum fusion
            x_fused = sum(branch_outputs)
        
        x1 = self.W_x_bn(x_fused)
        
        # Combine deep guidance and shallow multi-scale features to generate spatial gating filter mechanism
        psi = self.relu(g1 + x1)
        attn_weight = self.psi(psi)
        return x * attn_weight
class ADRNet(nn.Module):
    """
    Full ablation experiment parameter description:
    - use_attention: whether to use attention mechanism (False = original UNet)
    - use_dynamic_weight: whether to use dynamic scale weight (False = fixed sum)
    - use_asymmetric_dilation: whether to use asymmetric physiological receptive field degradation mechanism
    - base_dilations: base dilation rate configuration (used for all layers when use_asymmetric_dilation=False)
    - remove_rate3: whether to remove rate=3 branch (for ablation experiments)
    - remove_rate5: whether to remove rate=5 branch (for ablation experiments)
    """
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1,
                 use_attention=True,
                 use_dynamic_weight=True,
                 use_asymmetric_dilation=True,
                 base_dilations=[1, 3, 5],
                 remove_rate3=False,
                 remove_rate5=False):
        super().__init__()
        
        self.use_attention = use_attention
        
        # 1. Encoder stage
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.bottle = ConvBlock(512, 1024)
        
        # 2. Transposed upsampling in Decoder stage
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        # Dynamically configure dilation rates for each layer according to ablation parameters
        if use_asymmetric_dilation:
            # Asymmetric physiological receptive field configuration (original ADR-Net)
            ag4_dilations = [1, 3, 5]
            ag3_dilations = [1, 2, 3]
            ag2_dilations = [1, 1, 2]
            ag1_dilations = [1, 1, 1]
        else:
            # Unified receptive field configuration (for ablation experiments)
            ag4_dilations = base_dilations.copy()
            ag3_dilations = base_dilations.copy()
            ag2_dilations = base_dilations.copy()
            ag1_dilations = base_dilations.copy()
        
        # Handle branch removal ablation
        if remove_rate3:
            ag4_dilations = [d for d in ag4_dilations if d != 3]
            ag3_dilations = [d for d in ag3_dilations if d != 3]
            ag2_dilations = [d for d in ag2_dilations if d != 3]
            ag1_dilations = [d for d in ag1_dilations if d != 3]
        
        if remove_rate5:
            ag4_dilations = [d for d in ag4_dilations if d != 5]
            ag3_dilations = [d for d in ag3_dilations if d != 5]
            ag2_dilations = [d for d in ag2_dilations if d != 5]
            ag1_dilations = [d for d in ag1_dilations if d != 5]
        
        # 3. Deploy MRA-Gate attention gates (only when attention is enabled)
        if self.use_attention:
            self.ag4 = MRAGate(F_g=512, F_l=512, F_int=256, 
                              dilations=ag4_dilations, use_dynamic_weight=use_dynamic_weight)
            self.ag3 = MRAGate(F_g=256, F_l=256, F_int=128, 
                              dilations=ag3_dilations, use_dynamic_weight=use_dynamic_weight)
            self.ag2 = MRAGate(F_g=128, F_l=128, F_int=64,  
                              dilations=ag2_dilations, use_dynamic_weight=use_dynamic_weight)
            self.ag1 = MRAGate(F_g=64,  F_l=64,  F_int=32,  
                              dilations=ag1_dilations, use_dynamic_weight=use_dynamic_weight)
        
        # 4. Decoding feature fusion blocks
        self.d4 = ConvBlock(512 + 512, 512)
        self.d3 = ConvBlock(256 + 256, 256)
        self.d2 = ConvBlock(128 + 128, 128)
        self.d1 = ConvBlock(64 + 64, 64)
        
        # Segmentation output mapping
        self.seg_head = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottle(self.pool(e4))
        
        d4_up = self.up4(b)
        if self.use_attention:
            e4_attn = self.ag4(g=d4_up, x=e4)
            d4 = self.d4(torch.cat([d4_up, e4_attn], dim=1))
        else:
            # Original UNet: directly concatenate skip connections
            d4 = self.d4(torch.cat([d4_up, e4], dim=1))
        
        d3_up = self.up3(d4)
        if self.use_attention:
            e3_attn = self.ag3(g=d3_up, x=e3)
            d3 = self.d3(torch.cat([d3_up, e3_attn], dim=1))
        else:
            d3 = self.d3(torch.cat([d3_up, e3], dim=1))
        
        d2_up = self.up2(d3)
        if self.use_attention:
            e2_attn = self.ag2(g=d2_up, x=e2)
            d2 = self.d2(torch.cat([d2_up, e2_attn], dim=1))
        else:
            d2 = self.d2(torch.cat([d2_up, e2], dim=1))
        
        d1_up = self.up1(d2)
        if self.use_attention:
            e1_attn = self.ag1(g=d1_up, x=e1)
            d1 = self.d1(torch.cat([d1_up, e1_attn], dim=1))
        else:
            d1 = self.d1(torch.cat([d1_up, e1], dim=1))
        
        return self.seg_head(d1)
#  Hard-locked external interface alias to ensure backward compatibility for legacy training scripts!
AttentionUNet = ADRNet