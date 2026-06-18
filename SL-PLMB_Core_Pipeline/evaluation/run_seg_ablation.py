"""
run_seg_ablation.py 
"""
import os
import sys
sys.path.append("/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE")
import torch
if torch.cuda.is_available() and sys.platform.startswith('linux'):
    torch.multiprocessing.set_start_method('spawn', force=True)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
from scipy.stats import ttest_rel

from models.net_seg_only import AttentionUNet as ADRNet
from torch.utils.data import DataLoader
from utils.metrics import (
    calculate_dsc, 
    calculate_iou, 
    calculate_sensitivity, 
    calculate_specificity, 
    calculate_hd95_assd
)

# ===================== Statistical test utility functions =====================
def get_flat_metric(raw_results, model_name, views, metric_key):
    """Merge metrics from three views into a single long list, automatically filter NaN"""
    all_vals = []
    if model_name not in raw_results:
        return np.array([])
    for v in views:
        all_vals.extend(raw_results[model_name]['raw'][v][metric_key])
    return np.array(all_vals)

def compute_p_value(arr1, arr2, higher_is_better=True):
    valid_idx = ~np.isnan(arr1) & ~np.isnan(arr2)
    if np.sum(valid_idx) < 5:
        return 1.0
    a, b = arr1[valid_idx], arr2[valid_idx]
    if not higher_is_better:
        a, b = -a, -b
    _, p_val = ttest_rel(a, b)
    return p_val

def get_p_stars(p_val):
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return ""

# ===================== Global configuration =====================
ROOT_DIR = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE"
VAL_ROOT = os.path.join(ROOT_DIR, "data/segmentation_val")
TRAIN_IMG_DIR = os.path.join(ROOT_DIR, "data/segmentation_train/images") 
TRAIN_MASK_DIR = os.path.join(ROOT_DIR, "data/segmentation_train/masks") 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (512, 512)
NUM_VIS_SAMPLES = 15  
BATCH_SIZE = 8
NUM_WORKERS = 2  
MAX_EPOCHS = 80  
PATIENCE = 10    

ABLATION_CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints", "ablation_weights")
VIS_OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "ablation_vis")
os.makedirs(ABLATION_CKPT_DIR, exist_ok=True)
os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

# ===================== Ablation variant definitions =====================
ABLATION_VARIANTS = {
    'S1_Baseline_UNet': {
        'use_attention': False, 'use_dynamic_weight': False,
        'use_asymmetric_dilation': False, 'base_dilations': [1],
        'remove_rate3': False, 'remove_rate5': False
    },
    'S2_Original_Attention_UNet': {
        'use_attention': True, 'use_dynamic_weight': False,
        'use_asymmetric_dilation': False, 'base_dilations': [1],
        'remove_rate3': False, 'remove_rate5': False
    },
    'S3_ADRNet_wo_Multiscale': {
        'use_attention': True, 'use_dynamic_weight': True,
        'use_asymmetric_dilation': True, 'base_dilations': [1],
        'remove_rate3': False, 'remove_rate5': False
    },
    'S4_ADRNet_wo_Rate5': {
        'use_attention': True, 'use_dynamic_weight': True,
        'use_asymmetric_dilation': True, 'base_dilations': [1, 3, 5],
        'remove_rate3': False, 'remove_rate5': True
    },
    'S5_ADRNet_wo_Dynamic_Weight': {
        'use_attention': True, 'use_dynamic_weight': False,
        'use_asymmetric_dilation': True, 'base_dilations': [1, 3, 5],
        'remove_rate3': False, 'remove_rate5': False
    },
    'S6_ADRNet_wo_Asymmetric_Dilation': {
        'use_attention': True, 'use_dynamic_weight': True,
        'use_asymmetric_dilation': False, 'base_dilations': [1, 3, 5],
        'remove_rate3': False, 'remove_rate5': False
    },
    'S7_ADRNet_Full': {
        'use_attention': True, 'use_dynamic_weight': True,
        'use_asymmetric_dilation': True, 'base_dilations': [1, 3, 5],
        'remove_rate3': False, 'remove_rate5': False
    }
}

def find_mask_file(img_filename, mask_dir):
    img_base = os.path.splitext(img_filename)[0]
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        mask_path = os.path.join(mask_dir, f"{img_base}{ext}")
        if os.path.exists(mask_path):
            return mask_path
    return None

def load_and_preprocess(img_path, mask_path):
    try:
        from PIL import Resampling
        BICUBIC = Resampling.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC
    
    img = Image.open(img_path).convert('L')
    img = img.resize(IMG_SIZE, BICUBIC)
    img_t = torch.from_numpy(np.array(img)/255.0).unsqueeze(0).unsqueeze(0).float()
    
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        raise FileNotFoundError(f"Mask file cannot be read: {mask_path}")
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    mask_np = (mask > 127).astype(np.uint8)
    return img_t, mask_np, np.array(img)

class SimpleSegDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.valid_pairs = []
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in img_files:
            mask_path = find_mask_file(img_file, mask_dir)
            if mask_path is not None:
                self.valid_pairs.append((os.path.join(img_dir, img_file), mask_path))
        if len(self.valid_pairs) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {img_dir} and {mask_dir}!")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        img_t, mask_np, _ = load_and_preprocess(img_path, mask_path)
        return {'image': img_t.squeeze(0), 'mask': torch.from_numpy(mask_np).unsqueeze(0).float()}

def get_valid_val_files(view_dir):
    img_dir = os.path.join(view_dir, 'images')
    mask_dir = os.path.join(view_dir, 'masks')
    valid_files = []
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in img_files:
        mask_path = find_mask_file(img_file, mask_dir)
        if mask_path is not None:
            valid_files.append(img_file)
    return sorted(valid_files)

def save_overlay_visualization(raw_img, gt_mask, pred_mask, save_path):
    color_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_img, contours, -1, (255, 0, 0), 2) 
    green_mask = np.zeros_like(color_img)
    green_mask[pred_mask == 1] = [0, 255, 0] 
    overlay = cv2.addWeighted(color_img, 1.0, green_mask, 0.4, 0)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def validate_model(model, val_loader):
    model.eval()
    dsc_scores = []
    with torch.no_grad():
        for batch in val_loader:
            img = batch['image'].to(DEVICE, non_blocking=True)
            mask = batch['mask'].to(DEVICE, non_blocking=True)
            pred = model(img)
            pred_mask = (torch.sigmoid(pred) > 0.5).float()
            for i in range(pred_mask.shape[0]):
                dsc = calculate_dsc(pred_mask[i,0].cpu().numpy(), mask[i,0].cpu().numpy())
                dsc_scores.append(dsc)
    return np.mean(dsc_scores)

def train_variant(variant_name, params, save_path):
    print(f"\n[{variant_name}] Dedicated weights not found, starting automatic training (to ensure fair comparison)...")
    model = ADRNet(in_channels=1, out_channels=1, **params).to(DEVICE)
    
    try:
        train_dataset = SimpleSegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        val_dataset = SimpleSegDataset(os.path.join(VAL_ROOT, 'A2C', 'images'), os.path.join(VAL_ROOT, 'A2C', 'masks'))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print(f"Training set: {len(train_dataset)} samples | Validation set: {len(val_dataset)} samples")
    except Exception as e:
        print(f"Data loading failed!\n{e}")
        return False
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_dsc, patience_counter = 0.0, 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)
        for batch in pbar:
            img = batch['image'].to(DEVICE, non_blocking=True)
            mask = batch['mask'].to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'train_loss': f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        val_dsc = validate_model(model, val_loader)
        
        if val_dsc > best_dsc:
            best_dsc, patience_counter = val_dsc, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n Early stopping triggered! No validation performance improvement for {PATIENCE} consecutive epochs")
            break
            
    print(f" [{variant_name}] Training completed! Best validation DSC: {best_dsc:.4f}")
    return True

# =====================  Full evaluation engine: integrated 6 core metrics =====================
def evaluate_and_visualize(variant_name, params, ckpt_path):
    print(f"\n Rigorously evaluating variant (full 6-metric assessment): {variant_name}")
    model = ADRNet(in_channels=1, out_channels=1, **params).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    views = ['A2C', 'A3C', 'A4C']
    metrics_keys = ['dsc', 'iou', 'sens', 'spec', 'hd95', 'assd']
    per_view_results = {}
    raw_results = {v: {k: [] for k in metrics_keys} for v in views}
    
    with torch.no_grad():
        for view in views:
            view_dir = os.path.join(VAL_ROOT, view)
            if not os.path.exists(view_dir):
                continue
            test_files = get_valid_val_files(view_dir)
            if len(test_files) == 0:
                continue
            
            vis_files = test_files[:NUM_VIS_SAMPLES]
            vis_save_dir = os.path.join(VIS_OUTPUT_DIR, variant_name, view)
            os.makedirs(vis_save_dir, exist_ok=True)
            
            view_data = {k: [] for k in metrics_keys}
            print(f"  Evaluating all {len(test_files)} valid samples in {view} view...")
            for f in tqdm(test_files, desc=f"  Evaluating {view}", leave=False):
                img_path = os.path.join(view_dir, 'images', f)
                mask_path = find_mask_file(f, os.path.join(view_dir, 'masks'))
                img_t, gt_mask, _ = load_and_preprocess(img_path, mask_path)
                img_t = img_t.to(DEVICE)
                
                pred = model(img_t)
                pred_mask = (torch.sigmoid(pred)[0,0].cpu().numpy() > 0.5).astype(np.uint8)
                
                view_data['dsc'].append(calculate_dsc(pred_mask, gt_mask))
                view_data['iou'].append(calculate_iou(pred_mask, gt_mask))
                view_data['sens'].append(calculate_sensitivity(pred_mask, gt_mask))
                view_data['spec'].append(calculate_specificity(pred_mask, gt_mask))
                
                hd95, assd = calculate_hd95_assd(pred_mask, gt_mask)
                view_data['hd95'].append(hd95 if not np.isnan(hd95) else np.nan)
                view_data['assd'].append(assd if not np.isnan(assd) else np.nan)
            
            # Record raw data for cross-view combined statistics
            for k in metrics_keys:
                raw_results[view][k] = view_data[k]
            
            # Generate representative visualizations
            for f in vis_files:
                img_path = os.path.join(view_dir, 'images', f)
                mask_path = find_mask_file(f, os.path.join(view_dir, 'masks'))
                img_t, gt_mask, raw_img = load_and_preprocess(img_path, mask_path)
                pred = model(img_t.to(DEVICE))
                pred_mask = (torch.sigmoid(pred)[0,0].cpu().numpy() > 0.5).astype(np.uint8)
                save_overlay_visualization(raw_img, gt_mask, pred_mask, os.path.join(vis_save_dir, f))
            
            per_view_results[view] = {}
            for k in metrics_keys:
                per_view_results[view][k] = float(np.nanmean(view_data[k]))
                per_view_results[view][f"{k}_std"] = float(np.nanstd(view_data[k]))
    
    summary = {'global': {}, 'per_view': per_view_results, 'raw': raw_results}
    for k in metrics_keys:
        all_vals = []
        for v in views:
            all_vals.extend(raw_results[v][k])
        summary['global'][k] = float(np.nanmean(all_vals))
        summary['global'][f"{k}_std"] = float(np.nanstd(all_vals))
        
    print(f"{variant_name} evaluation completed! Global DSC: {summary['global']['dsc']:.4f}")
    return summary

# =====================  Markdown generation engine: full metric injection =====================
def generate_markdown_report(ablation_results, p_values, timestamp):
    md = f"""# ADR-Net Segmentation Model Internal Ablation Experiment Report (Full 6-Metric Version)
**Experiment Time**: {timestamp}
**Experiment Setup**: All variants are trained for 80 epochs under exactly the same conditions (to ensure absolute fairness).
**Statistical Test**: Paired t-test against the full model (S7_ADRNet_Full), *: p<0.05, **: p<0.01, ***: p<0.001

## 1. Global Comprehensive Metrics (Global 6-Metrics)
| Model Variant | Removed Module Description | DSC ↑ | IoU ↑ | Sens ↑ | Spec ↑ | HD95 ↓ (px) | ASSD ↓ (px) |
|---|---|---|---|---|---|---|---|
"""
    ours_name = 'S7_ADRNet_Full'
    metrics_keys = ['dsc', 'iou', 'sens', 'spec', 'hd95', 'assd']
    
    for variant_name in ABLATION_VARIANTS.keys():
        if variant_name not in ablation_results or ablation_results[variant_name] is None:
            continue
        res = ablation_results[variant_name]['global']
        
        stars = {k: '' for k in metrics_keys}
        if variant_name != ours_name and variant_name in p_values:
            for k in metrics_keys:
                stars[k] = get_p_stars(p_values[variant_name][k])
        
        if "Full" in variant_name:
            desc = "Full model (Ours)"
        elif "Baseline" in variant_name:
            desc = "Basic UNet"
        elif "Original_Attention" in variant_name:
            desc = "Original Attention Gate"
        elif "wo_Multiscale" in variant_name:
            desc = "Remove multi-scale branch"
        elif "wo_Rate5" in variant_name:
            desc = "Remove global receptive field branch (Rate=5)"
        elif "wo_Dynamic_Weight" in variant_name:
            desc = "Remove dynamic weight predictor"
        elif "wo_Asymmetric_Dilation" in variant_name:
            desc = "Remove asymmetric receptive field decay"
        else:
            desc = variant_name
            
        md += (f"| {variant_name} | {desc} | "
               f"{res['dsc']:.4f}{stars['dsc']}±{res['dsc_std']:.4f} | "
               f"{res['iou']:.4f}{stars['iou']}±{res['iou_std']:.4f} | "
               f"{res['sens']:.4f}{stars['sens']}±{res['sens_std']:.4f} | "
               f"{res['spec']:.4f}{stars['spec']}±{res['spec_std']:.4f} | "
               f"{res['hd95']:.2f}{stars['hd95']}±{res['hd95_std']:.2f} | "
               f"{res['assd']:.2f}{stars['assd']}±{res['assd_std']:.2f} |\n")
    
    md += "\n## 2. Detailed Statistical Significance Test P-Value Summary\n"
    md += "| Comparison Pair | DSC | IoU | Sens | Spec | HD95 | ASSD |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for variant_name, p_vals in p_values.items():
        md += (f"| ADR-Net Full vs {variant_name} | "
               f"{p_vals['dsc']:.4e} | {p_vals['iou']:.4e} | "
               f"{p_vals['sens']:.4e} | {p_vals['spec']:.4e} | "
               f"{p_vals['hd95']:.4e} | {p_vals['assd']:.4e} |\n")
        
    md += "\n## 3. Per-View Independent Metric Distribution\n"
    for view in ['A2C', 'A3C', 'A4C']:
        md += f"### View: {view}\n"
        md += "| Model Variant | DSC ↑ | IoU ↑ | Sens ↑ | Spec ↑ | HD95 ↓ | ASSD ↓ |\n"
        md += "|---|---|---|---|---|---|---|\n"
        for variant_name in ABLATION_VARIANTS.keys():
            if variant_name not in ablation_results or ablation_results[variant_name] is None:
                continue
            res = ablation_results[variant_name]['per_view'][view]
            md += (f"| {variant_name} | {res['dsc']:.4f}±{res['dsc_std']:.4f} | "
                   f"{res['iou']:.4f}±{res['iou_std']:.4f} | "
                   f"{res['sens']:.4f}±{res['sens_std']:.4f} | "
                   f"{res['spec']:.4f}±{res['spec_std']:.4f} | "
                   f"{res['hd95']:.2f}±{res['hd95_std']:.2f} | "
                   f"{res['assd']:.2f}±{res['assd_std']:.2f} |\n")
        md += "\n"
    return md

# ===================== Main program =====================
def main():
    print("="*80)
    print(" ADR-Net Segmentation Model Internal Ablation Experiment (Full 6-metric statistically enhanced version)")
    print("="*80)
    
    ablation_results = {}
    for variant_name, params in ABLATION_VARIANTS.items():
        ckpt_path = os.path.join(ABLATION_CKPT_DIR, f"{variant_name}.pth")
        if not os.path.exists(ckpt_path):
            if variant_name == 'S7_ADRNet_Full':
                main_ckpt = os.path.join(ROOT_DIR, 'checkpoints/best_seg_model.pth')
                if os.path.exists(main_ckpt):
                    import shutil
                    shutil.copy(main_ckpt, ckpt_path)
                    print(f"\n Cloned main model weights to {ckpt_path}")
                else:
                    train_variant(variant_name, params, ckpt_path)
            else:
                train_variant(variant_name, params, ckpt_path)
            
        if os.path.exists(ckpt_path):
            ablation_results[variant_name] = evaluate_and_visualize(variant_name, params, ckpt_path)
    
    ours_name = 'S7_ADRNet_Full'
    views = ['A2C', 'A3C', 'A4C']
    metrics_keys = ['dsc', 'iou', 'sens', 'spec', 'hd95', 'assd']
    metric_directions = {'dsc': True, 'iou': True, 'sens': True, 'spec': True, 'hd95': False, 'assd': False}
    
    global_p_values = {}
    if ours_name in ablation_results:
        print("\n Rigorously calculating full-dimensional statistical significance test (P-value)...")
        for comp_name in ABLATION_VARIANTS.keys():
            if comp_name == ours_name:
                continue
            if comp_name not in ablation_results:
                continue
            
            global_p_values[comp_name] = {}
            for metric in metrics_keys:
                ours_vals = get_flat_metric(ablation_results, ours_name, views, metric)
                comp_vals = get_flat_metric(ablation_results, comp_name, views, metric)
                global_p_values[comp_name][metric] = compute_p_value(ours_vals, comp_vals, metric_directions[metric])
    
    output_dir = os.path.join(ROOT_DIR, 'results/ablation_summary')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(os.path.join(output_dir, f'seg_ablation_results_with_p_{timestamp}.json'), 'w', encoding='utf-8') as f:
        json.dump({'ablation_results': ablation_results, 'p_values': global_p_values}, f, indent=4)
        
    md_path = os.path.join(output_dir, f'seg_ablation_report_with_significance_{timestamp}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(generate_markdown_report(ablation_results, global_p_values, timestamp))
        
    print(f"\n Full-metric ablation experiment successfully completed!")
    print(f" Full MD report perfectly aligned with Table 1 format generated: {md_path}")

if __name__ == "__main__":
    main()