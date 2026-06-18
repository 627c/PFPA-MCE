"""
train.py - Knowledge Injection and Multi-Task Joint Training Version
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from models.net import AttentionUNet
from data.dataset import build_dataloader
from config import BASE_CONFIG, PATHS

class MultiTaskLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.7, mse_weight=0.5):
        super().__init__()
        self.bce_weight, self.dice_weight, self.mse_weight = bce_weight, dice_weight, mse_weight
    
    def forward(self, pred_mask, gt_mask, pred_perf, gt_perf, mask):
        gt = gt_mask.float()
        bce = nn.BCEWithLogitsLoss()(pred_mask, gt)
        pred_sig = torch.clamp(torch.sigmoid(pred_mask), 1e-6, 1.0 - 1e-6)
        
        intersection = (pred_sig * gt).sum(dim=(2,3))
        union = pred_sig.sum(dim=(2,3)) + gt.sum(dim=(2,3))
        dice = 1.0 - (2.0 * intersection / (union + 1e-6)).mean()
        
        myo = (mask > 0.5).squeeze(1).bool()
        if myo.sum() > 0:
            mse_A = F.mse_loss(pred_perf[:,0][myo], gt_perf[:,0][myo])
            mse_beta = F.mse_loss(pred_perf[:,1][myo], gt_perf[:,1][myo])
            mse = mse_A + mse_beta * 50.0 
        else:
            mse = torch.tensor(0.0, device=pred_mask.device, requires_grad=True)
            
        return self.bce_weight * bce + self.dice_weight * dice + self.mse_weight * mse, mse.item()

def train():
    device = BASE_CONFIG['device']
    model = AttentionUNet().to(device)

    # Brain Transfer (1ch -> 3ch)
    seg_ckpt_path = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/checkpoints/best_seg_model.pth'
    if os.path.exists(seg_ckpt_path):
        print(f"\n Injecting segmentation prior knowledge...")
        pretrained_dict = torch.load(seg_ckpt_path, map_location=device)
        pretrained_dict = pretrained_dict.get('model_state_dict', pretrained_dict)
        model_dict = model.state_dict()
        adapted_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if k == 'enc1.conv.0.weight' and v.shape[1] == 1 and model_dict[k].shape[1] == 3:
                    adapted_dict[k] = v.repeat(1, 3, 1, 1) / 3.0
                elif v.shape == model_dict[k].shape:
                    adapted_dict[k] = v
        model_dict.update(adapted_dict)
        model.load_state_dict(model_dict)
        print(f"Successfully injected perfect weights for {len(adapted_dict)} network layers!\n")

    criterion = MultiTaskLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Note: For ablation studies, point the train path in PATHS to training_no_tf
    train_loader = build_dataloader(PATHS['train_images'], PATHS['train_masks'], PATHS['train_perfusion'], batch_size=8, train=True)
    val_loader = build_dataloader(PATHS['val_images'], PATHS['val_masks'], PATHS['val_perfusion'], batch_size=4, train=False)

    best_mse = float('inf')
    os.makedirs(os.path.dirname(PATHS['model_save']), exist_ok=True)

    for epoch in range(50):
        model.train()
        t_loss = t_mse = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/50 Train")
        for batch in train_pbar:
            img, mask, gt_perf = batch['image'].to(device), batch['mask'].to(device), batch['perfusion'].to(device)
            optimizer.zero_grad()
            pred_mask, pred_perf, _ = model(img)
            loss, mse_val = criterion(pred_mask, mask, pred_perf, gt_perf, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            t_loss += loss.item() * img.size(0)
            t_mse += mse_val * img.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.2f}", 'mse': f"{mse_val:.3f}"})

        scheduler.step()
        t_loss /= len(train_loader.dataset)
        t_mse /= len(train_loader.dataset)

        model.eval()
        v_loss = v_mse = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                img, mask, gt_perf = batch['image'].to(device), batch['mask'].to(device), batch['perfusion'].to(device)
                pred_mask, pred_perf, _ = model(img)
                loss, mse_val = criterion(pred_mask, mask, pred_perf, gt_perf, mask)
                v_loss += loss.item() * img.size(0)
                v_mse += mse_val * img.size(0)
        
        v_loss /= len(val_loader.dataset)
        v_mse /= len(val_loader.dataset)
        
        print(f'\nEpoch {epoch+1} | Train: loss={t_loss:.4f}, mse={t_mse:.6f} | Val: loss={v_loss:.4f}, mse={v_mse:.6f}')
        
        if v_mse < best_mse:
            best_mse = v_mse
            torch.save(model.state_dict(), PATHS['model_save'])
            print(f"Best MSE={best_mse:.6f} saved")

if __name__ == "__main__":
    train()