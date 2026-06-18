"""
net.py - 顶刊级大一统多任务版：ADR-Net (Acoustic-Dropout-Resilient Network)
核心功能：
1. 完美集成带自适应尺度选择权重的满血 MRA-Gate
2. 3通道时空三元组输入 [Destroy, Mid, Plateau] 的全物理灌注回归支持
3. 物理约束回归头 (Softplus + Hardtanh) 组合拳锁死数值边界，杜绝负数和无穷大参数
4. 完美无缝兼容 train.py (多任务联调) 与 main.py (PLMB特征融合截断)
5. ✅ 新增：与纯分割版完全一致的完整消融实验参数支持
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
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
    👑 动态特征级多尺度自适应门控模块
    专用于对抗动态时空三元组中由于呼吸漂移引起的边界异动与严重的局部声学黑洞。
    
    ✅ 消融参数（与纯分割版100%一致）：
    - dilations: 空洞率列表，控制感受野大小
    - use_dynamic_weight: 是否使用动态权重预测头（False=固定相加）
    """
    def __init__(self, F_g, F_l, F_int, dilations=[1, 3, 5], use_dynamic_weight=True):
        super().__init__()
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
            x_cat = torch.cat(branch_outputs, dim=1)
            scale_weights = self.weight_predictor(x_cat)  # 形状: [B, num_branches, H, W]
            x_fused = 0
            for i in range(self.num_branches):
                x_fused += scale_weights[:, i:i+1, :, :] * branch_outputs[i]
        else:
            x_fused = sum(branch_outputs)
        
        x1 = self.W_x_bn(x_fused)
        psi = self.relu(g1 + x1)
        attn_weight = self.psi(psi)
        return x * attn_weight

class ADRNetMultiTask(nn.Module):
    """
    ✅ 完整消融实验参数说明（与纯分割版100%一致）：
    - use_attention: 是否使用注意力机制（False=原版UNet）
    - use_dynamic_weight: 是否使用动态尺度权重（False=固定相加）
    - use_asymmetric_dilation: 是否使用非对称生理感受野退化机制
    - base_dilations: 基础空洞率配置（当use_asymmetric_dilation=False时所有层使用此配置）
    - remove_rate3: 是否移除rate=3分支（消融实验用）
    - remove_rate5: 是否移除rate=5分支（消融实验用）
    """
    def __init__(self, 
                 in_channels=3, 
                 out_channels=1,
                 use_attention=True,
                 use_dynamic_weight=True,
                 use_asymmetric_dilation=True,
                 base_dilations=[1, 3, 5],
                 remove_rate3=False,
                 remove_rate5=False):
        """
        in_channels=3: 精确流转 3通道时空三元组 [Destroy, Mid, Plateau]
        out_channels=1: 用于拉紧浅层空间流转形态的辅助分割掩膜输出
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        # 1. Encoder 组
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)
        
        # 2. Decoder 转置上采样映射组
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        # ✅ 根据消融参数动态配置各层的空洞率（与纯分割版完全一致）
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
        
        # 3. 👑 动态尺度决策级 MRA-Gate 跳跃连接矩阵
        if self.use_attention:
            self.ag4 = MRAGate(F_g=512, F_l=512, F_int=256, 
                              dilations=ag4_dilations, use_dynamic_weight=use_dynamic_weight)
            self.ag3 = MRAGate(F_g=256, F_l=256, F_int=128, 
                              dilations=ag3_dilations, use_dynamic_weight=use_dynamic_weight)
            self.ag2 = MRAGate(F_g=128, F_l=128, F_int=64,  
                              dilations=ag2_dilations, use_dynamic_weight=use_dynamic_weight)
            self.ag1 = MRAGate(F_g=64,  F_l=64,  F_int=32,  
                              dilations=ag1_dilations, use_dynamic_weight=use_dynamic_weight)
        
        # 4. 特征深度聚合卷积块
        self.dec4 = ConvBlock(512 + 512, 512)
        self.dec3 = ConvBlock(256 + 256, 256)
        self.dec2 = ConvBlock(128 + 128, 128)
        self.dec1 = ConvBlock(64 + 64, 64)
        
        # 5. 双路任务解耦头 (Dual-Task Decoupled Heads)
        # 辅助任务：解剖形态学分割映射
        self.seg_head = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # 核心任务：定量参数物理级硬回归空间中枢 (A 和 beta 深度回归)
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1), # 通道0为A，通道1为beta
            nn.Softplus(),                     # 数学约束：回归出的灌注物理参数绝对不能出现负数
            nn.Hardtanh(min_val=1e-5, max_val=1000.0) # 物理边界：防止无穷大导致的数值爆炸
        )
    
    def forward_features(self, x):
        """
        专供推理引擎 main.py 调用！
        阻断并抓取 dec1 层吐出的 64通道高阶语义特征图，用于注入下游 PLMB 多视图交叉融合矩阵。
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        
        d4_up = self.up4(bn)
        if self.use_attention:
            e4_attn = self.ag4(g=d4_up, x=e4)
            d4 = self.dec4(torch.cat([d4_up, e4_attn], dim=1))
        else:
            d4 = self.dec4(torch.cat([d4_up, e4], dim=1))
        
        d3_up = self.up3(d4)
        if self.use_attention:
            e3_attn = self.ag3(g=d3_up, x=e3)
            d3 = self.dec3(torch.cat([d3_up, e3_attn], dim=1))
        else:
            d3 = self.dec3(torch.cat([d3_up, e3], dim=1))
        
        d2_up = self.up2(d3)
        if self.use_attention:
            e2_attn = self.ag2(g=d2_up, x=e2)
            d2 = self.dec2(torch.cat([d2_up, e2_attn], dim=1))
        else:
            d2 = self.dec2(torch.cat([d2_up, e2], dim=1))
        
        d1_up = self.up1(d2)
        if self.use_attention:
            e1_attn = self.ag1(g=d1_up, x=e1)
            d1 = self.dec1(torch.cat([d1_up, e1_attn], dim=1))
        else:
            d1 = self.dec1(torch.cat([d1_up, e1], dim=1))
        
        seg_logits = self.seg_head(d1)
        return seg_logits, d1
    
    def forward_regression(self, fused_features):
        """
        专供推理引擎 main.py 调用！
        接收被 PLMB 洗练、增强、补充多切面记忆后的连续特征图，一击回归出临床级灌注热力图。
        """
        return self.reg_head(fused_features)
    
    def forward(self, x):
        """
        专供端到端训练引擎 train.py 调用！
        实现多任务统一推演，同步分流。
        """
        seg_logits, d1 = self.forward_features(x)
        pred_perf = self.forward_regression(d1)
        return seg_logits, pred_perf, d1

# ✅ 强制兼容大模型对外全局统一接口
AttentionUNet = ADRNetMultiTask