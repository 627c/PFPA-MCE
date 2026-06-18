"""
net_seg_only.py - 顶刊级原创方法学版：ADR-Net (Acoustic-Dropout-Resilient Network)
核心创新：
1. 搭载自适应尺度权重预测头(Dynamic Scale Weight Predictor)的 MRA-Gate
2. 引入非对称生理感受野退化机制，彻底根除高分辨率特征图上的网格伪影(Gridding Artifacts)
3. 专用于几千张静态带掩膜图像的单通道训练 Track 1
4. ✅ 新增：完整消融实验参数支持（一键切换所有变体）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """全局通道注意力，用于压制超声特有的随机高亮斑点噪声"""
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
    👑 核心创新模块：多尺度感受野自适应注意力门 (Multi-scale Receptive Attention Gate)
    根据声学脱落区域的大小，自适应预测并分配三个多尺度多感受野分支的控制权重，实现空洞特征的柔性动态融合。
    
    ✅ 消融参数：
    - dilations: 空洞率列表，控制感受野大小
    - use_dynamic_weight: 是否使用动态权重预测头（False=固定相加）
    """
    def __init__(self, F_g, F_l, F_int, dilations=[1, 3, 5], use_dynamic_weight=True):
        super().__init__()
        # 深层门控引导信号映射
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.dilations = dilations
        self.use_dynamic_weight = use_dynamic_weight
        self.num_branches = len(dilations)
        
        # ✅ 动态创建多尺度空洞卷积分支（支持任意数量的空洞率）
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Conv2d(F_l, F_int, kernel_size=3, padding=d, dilation=d, bias=False)
            )
        
        # ✅ 动态创建权重预测头（仅当分支数>1且启用动态权重时）
        if self.use_dynamic_weight and self.num_branches > 1:
            self.weight_predictor = nn.Sequential(
                nn.Conv2d(F_int * self.num_branches, self.num_branches, kernel_size=1, bias=True),
                nn.Softmax(dim=1)
            )
        
        self.W_x_bn = nn.BatchNorm2d(F_int)
        self.relu = nn.ReLU(inplace=True)
        
        # 空间注意力图映射
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        
        # ✅ 并行提取所有分支的特征
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # ✅ 动态融合分支特征
        if self.use_dynamic_weight and self.num_branches > 1:
            # 拼接所有分支特征
            x_cat = torch.cat(branch_outputs, dim=1)
            # 预测每个空间位置的分支权重
            scale_weights = self.weight_predictor(x_cat)  # 形状: [B, num_branches, H, W]
            # 加权融合
            x_fused = 0
            for i in range(self.num_branches):
                x_fused += scale_weights[:, i:i+1, :, :] * branch_outputs[i]
        else:
            # 固定相加融合
            x_fused = sum(branch_outputs)
        
        x1 = self.W_x_bn(x_fused)
        
        # 结合深层导向与浅层多尺度特征生成空间门控过滤机制
        psi = self.relu(g1 + x1)
        attn_weight = self.psi(psi)
        return x * attn_weight

class ADRNet(nn.Module):
    """
    ✅ 完整消融实验参数说明：
    - use_attention: 是否使用注意力机制（False=原版UNet）
    - use_dynamic_weight: 是否使用动态尺度权重（False=固定相加）
    - use_asymmetric_dilation: 是否使用非对称生理感受野退化机制
    - base_dilations: 基础空洞率配置（当use_asymmetric_dilation=False时所有层使用此配置）
    - remove_rate3: 是否移除rate=3分支（消融实验用）
    - remove_rate5: 是否移除rate=5分支（消融实验用）
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
        
        # 1. Encoder 阶段
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.bottle = ConvBlock(512, 1024)
        
        # 2. Decoder 阶段的转置上采样
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        # ✅ 根据消融参数动态配置各层的空洞率
        if use_asymmetric_dilation:
            # 非对称生理感受野配置（原版ADR-Net）
            ag4_dilations = [1, 3, 5]
            ag3_dilations = [1, 2, 3]
            ag2_dilations = [1, 1, 2]
            ag1_dilations = [1, 1, 1]
        else:
            # 统一感受野配置（消融实验用）
            ag4_dilations = base_dilations.copy()
            ag3_dilations = base_dilations.copy()
            ag2_dilations = base_dilations.copy()
            ag1_dilations = base_dilations.copy()
        
        # ✅ 处理分支移除消融
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
        
        # 3. 部署 MRA-Gate 注意力门（仅当启用注意力时）
        if self.use_attention:
            self.ag4 = MRAGate(F_g=512, F_l=512, F_int=256, 
                              dilations=ag4_dilations, use_dynamic_weight=use_dynamic_weight)
            self.ag3 = MRAGate(F_g=256, F_l=256, F_int=128, 
                              dilations=ag3_dilations, use_dynamic_weight=use_dynamic_weight)
            self.ag2 = MRAGate(F_g=128, F_l=128, F_int=64,  
                              dilations=ag2_dilations, use_dynamic_weight=use_dynamic_weight)
            self.ag1 = MRAGate(F_g=64,  F_l=64,  F_int=32,  
                              dilations=ag1_dilations, use_dynamic_weight=use_dynamic_weight)
        
        # 4. 解码特征融合块
        self.d4 = ConvBlock(512 + 512, 512)
        self.d3 = ConvBlock(256 + 256, 256)
        self.d2 = ConvBlock(128 + 128, 128)
        self.d1 = ConvBlock(64 + 64, 64)
        
        # 分割输出映射
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
            # 原版UNet：直接拼接跳跃连接
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

# ✅ 强力锁死对外接口别名，确保历史训练脚本无损重连运行！
AttentionUNet = ADRNet