"""
dataset.py - High-speed dual-track pure version (supports Track1 static images and Track2 dynamic NPY triplets)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from core.anatomy import generate_aha17_mask
from utils.helpers import extract_patient_and_view

def keep_image_size_open(path, size=(512, 512), is_label=False):
    """Aspect-ratio-preserving reading with black border padding (designed specifically for PNG images)"""
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
    """High-speed NPY loader designed for Track 2 (dynamic perfusion multi-tasking)"""
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
        
        # 1. High-speed reading of 3-channel spatiotemporal triplet (.npy)
        triplet = np.load(os.path.join(self.image_dir, name_ext))
        img = torch.from_numpy(triplet).float()
        
        # 2. Read corresponding mask (.png)
        mask_path = os.path.join(self.mask_dir, name + '.png')
        mask_img = keep_image_size_open(mask_path, self.size, is_label=True)
        mask = np.array(mask_img)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        # 3. High-speed reading of pre-generated perfusion GT (.npy)
        perfusion = None
        if self.perfusion_dir:
            perf_path = os.path.join(self.perfusion_dir, name_ext)
            if os.path.exists(perf_path):
                perfusion = torch.from_numpy(np.load(perf_path)).float()
                
        # 4. Generate AHA mask on-the-fly
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
    Unified Dataloader construction interface, directly callable by train.py
    """
    dataset = MCEDatasetNpy(image_dir, mask_dir, perfusion_dir, use_aha=use_aha)
    # Shuffle training set, do not shuffle validation set
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)