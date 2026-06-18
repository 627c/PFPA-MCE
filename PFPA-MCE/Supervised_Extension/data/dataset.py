"""
dataset.py - 极速双轨纯净版（支撑 Track1 静态图与 Track2 动态 NPY 三元组）
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from core.anatomy import generate_aha17_mask
from utils.helpers import extract_patient_and_view

def keep_image_size_open(path, size=(512, 512), is_label=False):
    """保持长宽比的黑边填充读取法（专为PNG图片设计）"""
    img = Image.open(path).convert('L')
    target_w, target_h = size
    orig_w, orig_h = img.size
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    resample_method = Image.Resampling.NEAREST if is_label else Image.Resampling.BICUBIC
    img = img.resize((new_w, new_h), resample_method)
    new_img = Image.new('L', (target_w, target_h), 0)
    new_img.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return new_img

class MCEDatasetNpy(Dataset):
    """专为 Track 2 (动态灌注多任务) 设计的高速 NPY 加载器"""
    def __init__(self, image_dir, mask_dir, perfusion_dir=None, size=(512, 512), use_aha=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.perfusion_dir = perfusion_dir
        self.size = size
        self.use_aha = use_aha
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith('.npy')]
        self.filenames.sort()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name_ext = self.filenames[idx]
        name = os.path.splitext(name_ext)[0]
        
        # 1. 极速读取 3 通道时空三元组 (.npy)
        triplet = np.load(os.path.join(self.image_dir, name_ext))
        img = torch.from_numpy(triplet).float()
        
        # 2. 读取对应掩膜 (.png)
        mask_path = os.path.join(self.mask_dir, name + '.png')
        mask_img = keep_image_size_open(mask_path, self.size, is_label=True)
        mask = np.array(mask_img)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        # 3. 极速读取预生成的灌注 GT (.npy)
        perfusion = None
        if self.perfusion_dir:
            perf_path = os.path.join(self.perfusion_dir, name_ext)
            if os.path.exists(perf_path):
                perfusion = torch.from_numpy(np.load(perf_path)).float()
                
        # 4. 实时生成 AHA 掩膜
        aha = None
        if self.use_aha:
            _, view = extract_patient_and_view(name)
            aha_np = generate_aha17_mask(mask.squeeze(0).numpy(), view)
            aha = torch.from_numpy(aha_np).long()
            
        ret = {'image': img, 'mask': mask, 'filename': name}
        if perfusion is not None: ret['perfusion'] = perfusion
        if aha is not None: ret['aha_mask'] = aha
        return ret

def build_dataloader(image_dir, mask_dir, perfusion_dir=None, batch_size=8, train=True, use_aha=False):
    """
    统一的 Dataloader 构建接口，供 train.py 直接调用
    """
    dataset = MCEDatasetNpy(image_dir, mask_dir, perfusion_dir, use_aha=use_aha)
    # 训练集打乱，验证集不打乱
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)