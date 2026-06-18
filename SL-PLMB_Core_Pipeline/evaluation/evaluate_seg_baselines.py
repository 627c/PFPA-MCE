"""
evaluate_seg_baselines.py 
1. Calculate 6 major metrics (DSC, IoU, Sens, Spec, HD95, ASSD)
2. Automatically generate and save detailed Markdown evaluation reports
"""
import sys
sys.path.append("/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE")
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.stats import ttest_rel
import warnings
# Keep your original warning filters
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")

from models.net_seg_only import ADRNet
from models.baselines import (
    UNet, 
    AttentionUNet, 
    UNetPlusPlus, 
    nnUNet, 
    SwinUNet, 
    MobileUNet
)
from utils.metrics import calculate_dsc, calculate_iou, calculate_sensitivity, calculate_specificity, calculate_hd95_assd

# ==================== Configuration Section ====================
VAL_ROOT = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/segmentation_val"
RESULTS_DIR = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results/seg_baselines"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (512, 512)
NUM_VISUAL_SAMPLES = 15  # Generate 15 comparison images per view (45 total), adjustable as needed
VISUAL_SAVE_DIR = os.path.join(RESULTS_DIR, "visual_comparisons")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUAL_SAVE_DIR, exist_ok=True)

MODEL_CKPTS = {
    'ADR-Net (Ours)': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/checkpoints/best_seg_model.pth',
    'Attention-UNet': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/checkpoints/attention_unet_best.pth',
    'UNet': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/checkpoints/unet_best.pth',
    'UNet++': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/checkpoints/unetpp_best.pth',
    'nnU-Net': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/checkpoints/nnunet_best.pth',
    'Swin-UNet': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/checkpoints/swin_unet_best.pth',
    'MobileUNet': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/checkpoints/mobile_unet_best.pth'
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
COLOR_GT = (102, 194, 165)     # Mint green
COLOR_PRED = (252, 141, 98)    # Warning orange
MASK_ALPHA = 0.4               # Mask transparency (balances original image visibility and segmentation clarity)
CONTOUR_WIDTH = 2              # Contour line width

# ==================== Statistical Test Utility Functions ====================
def get_flat_metric(raw_results, model_name, views, metric_key):
    """Merge metrics from three views into a single long list, with automatic NaN filtering"""
    all_vals = []
    if model_name not in raw_results: return np.array([])
    for v in views:
        all_vals.extend(raw_results[model_name][v][metric_key])
    return np.array(all_vals)

def compute_p_value(arr1, arr2, higher_is_better=True):
    """
    Calculate p-value of paired t-test, automatically handles "higher is better" and "lower is better" metrics
    Args:
        higher_is_better: True = higher metric is better (DSC/IoU etc.), False = lower metric is better (HD95/ASSD etc.)
    """
    valid_idx = ~np.isnan(arr1) & ~np.isnan(arr2)
    n_valid = np.sum(valid_idx)
    if n_valid < 5:  # Insufficient sample size for statistical test
        return 1.0
    
    a, b = arr1[valid_idx], arr2[valid_idx]
    
    # For lower-is-better metrics, invert values before t-test to ensure correct significance direction
    if not higher_is_better:
        a, b = -a, -b
    
    # Two-tailed paired t-test
    _, p_val = ttest_rel(a, b)
    return p_val

def get_p_stars(p_val):
    """Top-journal standard significance markers"""
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return ""  # No marker for non-significant results

def load_and_preprocess(img_path, mask_path):
    try:
        from PIL import Resampling
        BICUBIC = Resampling.BICUBIC
    except ImportError:
        from PIL import Image
        BICUBIC = Image.BICUBIC
    
    # Load and preprocess image
    img = Image.open(img_path).convert('L')
    img = img.resize(IMG_SIZE, BICUBIC)
    img_t = torch.from_numpy(np.array(img)/255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    img_base = os.path.splitext(os.path.basename(img_path))[0]
    mask_real_path = os.path.join(os.path.dirname(mask_path), f"{img_base}.png")
    mask = cv2.imread(mask_real_path, 0)
    
    # Add fault tolerance: prevent complete missing of mask files
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {mask_real_path}")
        
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    mask_np = (mask > 127).astype(np.uint8)
    return img_t, mask_np

def get_model_instance(name):
    if name == 'ADR-Net (Ours)': return ADRNet()
    if name == 'Attention-UNet': return AttentionUNet()
    if name == 'UNet': return UNet()
    if name == 'UNet++': return UNetPlusPlus()
    if name == 'nnU-Net': return nnUNet()
    if name == 'Swin-UNet': return SwinUNet()
    if name == 'MobileUNet': return MobileUNet()
    return None

def generate_visual_comparisons(visual_samples, views):
    """
    Generate top-journal level visual comparison charts for segmentation results
    Layout: [Original] [Ground Truth] [ADR-Net] [nnU-Net] [UNet++] [UNet] [Swin-UNet] [MobileUNet] [Attention-UNet]
    """
    model_display_order = [
        'ADR-Net (Ours)', 
        'nnU-Net', 
        'UNet++', 
        'UNet', 
        'Swin-UNet', 
        'MobileUNet', 
        'Attention-UNet'
    ]
    
    for view in views:
        samples = visual_samples[view]
        if not samples:
            continue
            
        print(f"\n Generating visual comparison charts for view {view}...")
        for sample_idx, sample in enumerate(tqdm(samples, desc=f"  Generating {view}")):
            img = sample['img']
            gt_mask = sample['gt']
            all_preds = sample['preds']
            filename = sample['filename']
            
            # Filter models with valid prediction results
            valid_models = [m for m in model_display_order if m in all_preds]
            if not valid_models:
                continue
            
            # Calculate number of subplots: original + ground truth + all models
            n_subplots = 2 + len(valid_models)
            fig, axes = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 5), dpi=100)
            
            # Convert to uint8 format for OpenCV drawing
            img_uint8 = (img * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
            
            # 1. Draw original image
            axes[0].imshow(img_uint8, cmap='gray')
            axes[0].set_title('Original\nImage', fontsize=12, fontweight='bold', pad=10)
            axes[0].axis('off')
            
            # 2. Draw ground truth
            gt_overlay = img_rgb.copy()
            gt_color_mask = np.zeros_like(gt_overlay)
            gt_color_mask[gt_mask == 1] = COLOR_GT
            gt_overlay = cv2.addWeighted(gt_overlay, 1, gt_color_mask, MASK_ALPHA, 0)
            # Draw ground truth contour
            gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_overlay, gt_contours, -1, COLOR_GT, CONTOUR_WIDTH)
            axes[1].imshow(gt_overlay)
            axes[1].set_title('Ground\nTruth', fontsize=12, fontweight='bold', pad=10)
            axes[1].axis('off')
            
            # 3. Draw prediction results of each model
            for i, model_name in enumerate(valid_models):
                pred_mask = all_preds[model_name]
                pred_overlay = img_rgb.copy()
                pred_color_mask = np.zeros_like(pred_overlay)
                pred_color_mask[pred_mask == 1] = COLOR_PRED
                pred_overlay = cv2.addWeighted(pred_overlay, 1, pred_color_mask, MASK_ALPHA, 0)
                # Draw prediction contour
                pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pred_overlay, pred_contours, -1, COLOR_PRED, CONTOUR_WIDTH)
                
                # Calculate and display DSC value
                dsc = calculate_dsc(pred_mask, gt_mask)
                axes[i+2].imshow(pred_overlay)
                title = f"{model_name}\nDSC: {dsc:.4f}"
                fontweight = 'bold' if 'Ours' in model_name else 'normal'
                axes[i+2].set_title(title, fontsize=12, fontweight=fontweight, pad=10)
                axes[i+2].axis('off')
            
            # Add global title
            plt.suptitle(
                f"View: {view} | Sample: {os.path.splitext(filename)[0]}",
                fontsize=16, 
                fontweight='bold',
                y=1.05
            )
            
            # Adjust layout and save
            plt.tight_layout()
            save_path = os.path.join(VISUAL_SAVE_DIR, f"{view}_sample_{sample_idx:03d}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

# ==================== Core Evaluation Engine ====================
def evaluate_models():
    views = ['A2C', 'A3C', 'A4C']
    metrics_keys = ['dsc', 'iou', 'sens', 'spec', 'hd95', 'assd']
    # Define direction of each metric: True = higher is better, False = lower is better
    metric_directions = {
        'dsc': True, 'iou': True, 'sens': True, 'spec': True,
        'hd95': False, 'assd': False
    }
    
    # Store specific values of all samples, for mean, variance and statistical test calculation
    raw_results = {m: {v: {k: [] for k in metrics_keys} for v in views} for m in MODEL_CKPTS.keys()}
    
    # Store visualization sample data
    visual_samples = {v: [] for v in views}
    samples_collected = {v: False for v in views}
    
    for model_name, ckpt_path in MODEL_CKPTS.items():
        print(f"\n Evaluating: {model_name}")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint file {ckpt_path} not found, skipping.")
            continue
            
        model = get_model_instance(model_name).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        model.eval()
        
        with torch.no_grad():
            for view in views:
                view_dir = os.path.join(VAL_ROOT, view)
                if not os.path.exists(view_dir): continue
                
                img_dir, mask_dir = os.path.join(view_dir, 'images'), os.path.join(view_dir, 'masks')
                # Critical: sorting ensures all models read files in exactly the same order, which is the premise of paired t-test
                file_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                # When processing the first model, collect original images and ground truth for visualization samples
                if not samples_collected[view] and len(file_list) >= NUM_VISUAL_SAMPLES:
                    print(f"📸 Collecting visualization samples for view {view}...")
                    for f in tqdm(file_list[:NUM_VISUAL_SAMPLES], desc=f"  Collecting {view}"):
                        img_t, gt_mask = load_and_preprocess(os.path.join(img_dir, f), os.path.join(mask_dir, f))
                        img_np = img_t[0,0].cpu().numpy()
                        visual_samples[view].append({
                            'filename': f,
                            'img': img_np,
                            'gt': gt_mask,
                            'preds': {}
                        })
                    samples_collected[view] = True
                
                # Evaluate all samples
                for idx, f in enumerate(tqdm(file_list, desc=f"  Evaluating {view}")):
                    img_t, gt_mask = load_and_preprocess(os.path.join(img_dir, f), os.path.join(mask_dir, f))
                    pred = model(img_t)
                    pred_mask = (torch.sigmoid(pred)[0,0].cpu().numpy() > 0.5).astype(np.uint8)
                    
                    # Calculate metrics
                    raw_results[model_name][view]['dsc'].append(calculate_dsc(pred_mask, gt_mask))
                    raw_results[model_name][view]['iou'].append(calculate_iou(pred_mask, gt_mask))
                    raw_results[model_name][view]['sens'].append(calculate_sensitivity(pred_mask, gt_mask))
                    raw_results[model_name][view]['spec'].append(calculate_specificity(pred_mask, gt_mask))
                    
                    hd95, assd = calculate_hd95_assd(pred_mask, gt_mask)
                    # Even NaN values must be stored to ensure array lengths are perfectly aligned across all models
                    raw_results[model_name][view]['hd95'].append(hd95)
                    raw_results[model_name][view]['assd'].append(assd)
                    
                    # Save prediction results for visualization samples
                    if idx < NUM_VISUAL_SAMPLES and samples_collected[view]:
                        visual_samples[view][idx]['preds'][model_name] = pred_mask

    # Precompute all global p-values
    ours_name = 'ADR-Net (Ours)'
    global_p_values = {}
    if ours_name in raw_results and len(raw_results[ours_name]['A2C']['dsc']) > 0:
        for comp_name in MODEL_CKPTS.keys():
            if comp_name == ours_name: continue
            if len(raw_results[comp_name]['A2C']['dsc']) == 0: continue
            
            global_p_values[comp_name] = {}
            for metric in metrics_keys:
                ours_vals = get_flat_metric(raw_results, ours_name, views, metric)
                comp_vals = get_flat_metric(raw_results, comp_name, views, metric)
                p = compute_p_value(ours_vals, comp_vals, metric_directions[metric])
                global_p_values[comp_name][metric] = p

    # Generate all reports and charts
    generate_report(raw_results, views, metrics_keys, metric_directions, global_p_values, ours_name)
    plot_view_metrics(raw_results, views)
    plot_boxplot(raw_results, views)
    plot_radar_chart(raw_results, views)
    generate_visual_comparisons(visual_samples, views)  # New: generate visual comparison charts
    
    print(f"\n All evaluation tasks completed!")
    print(f"Evaluation reports and statistical charts saved to: {RESULTS_DIR}")
    print(f"Visual comparison charts of segmentation results saved to: {VISUAL_SAVE_DIR}")
    print(f"Report includes complete statistical significance test results, ready for direct use in paper Table 1")

# ==================== Generate Markdown Report (with significance markers) ====================
def generate_report(raw_results, views, metrics_keys, metric_directions, global_p_values, ours_name):
    report_path = os.path.join(RESULTS_DIR, "Segmentation_Evaluation_Report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# PFPA-MCE Segmentation Baseline Evaluation Report\n\n")
        
        ## 1. Global Summary Metrics
        f.write("## 1. Global Summary Metrics\n")
        f.write("*Note: Significance markers are results of paired t-tests against ADR-Net (Ours).*  \n")
        f.write("*\\*: p<0.05, \\*\\*: p<0.01, \\*\\*\\*: p<0.001, no marker indicates non-significant (p≥0.05)*\n\n")
        
        f.write("| Model | DSC | IoU | Sens | Spec | HD95 (px) | ASSD (px) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for m in raw_results.keys():
            if len(raw_results[m]['A2C']['dsc']) == 0: continue
            
            means = {}
            stars = {}
            for k in metrics_keys:
                all_vals = get_flat_metric(raw_results, m, views, k)
                means[k] = np.nanmean(all_vals)
                # Add significance asterisks for baseline models
                if m != ours_name and m in global_p_values:
                    stars[k] = get_p_stars(global_p_values[m][k])
                else:
                    stars[k] = ""
            
            # Formatted output with asterisks
            f.write(f"| {m} | "
                   f"{means['dsc']:.4f}{stars['dsc']} | "
                   f"{means['iou']:.4f}{stars['iou']} | "
                   f"{means['sens']:.4f}{stars['sens']} | "
                   f"{means['spec']:.4f}{stars['spec']} | "
                   f"{means['hd95']:.2f}{stars['hd95']} | "
                   f"{means['assd']:.2f}{stars['assd']} |\n")
        f.write("\n")
        
        ## 2. Detailed Statistical Significance Tests
        f.write("## 2. Detailed Statistical Significance Tests\n")
        f.write("| Comparison | DSC | IoU | Sens | Spec | HD95 | ASSD |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for comp_name, p_vals in global_p_values.items():
            f.write(f"| ADR-Net vs {comp_name} | "
                   f"{p_vals['dsc']:.4e} | "
                   f"{p_vals['iou']:.4e} | "
                   f"{p_vals['sens']:.4e} | "
                   f"{p_vals['spec']:.4e} | "
                   f"{p_vals['hd95']:.4e} | "
                   f"{p_vals['assd']:.4e} |\n")
        f.write("\n")
        
        ## 3. Per-View Metrics Breakdown
        f.write("## 3. Per-View Metrics Breakdown\n")
        for v in views:
            f.write(f"### View: {v}\n")
            f.write("| Model | DSC | IoU | Sens | Spec | HD95 (px) | ASSD (px) |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for m in raw_results.keys():
                if len(raw_results[m][v]['dsc']) == 0: continue
                means = {k: np.nanmean(raw_results[m][v][k]) for k in metrics_keys}
                f.write(f"| {m} | {means['dsc']:.4f} | {means['iou']:.4f} | {means['sens']:.4f} | {means['spec']:.4f} | {means['hd95']:.2f} | {means['assd']:.2f} |\n")
            f.write("\n")

# ==================== Plotting Functions ====================
def plot_view_metrics(raw_results, views):
    """Original per-view comparison bar chart"""
    models = [m for m in raw_results.keys() if len(raw_results[m]['A2C']['dsc']) > 0]
    if not models: return
    
    x = np.arange(len(views))
    width = 0.8 / len(models)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    for i, m in enumerate(models):
        dsc_means = [np.nanmean(raw_results[m][v]['dsc']) for v in views]
        hd95_means = [np.nanmean(raw_results[m][v]['hd95']) for v in views]
        offset = (i - len(models)/2 + 0.5) * width
        ax1.bar(x + offset, dsc_means, width, label=m, alpha=0.8)
        ax2.bar(x + offset, hd95_means, width, label=m, alpha=0.8)
        
    ax1.set_ylabel('DSC (Higher is better)', fontsize=14)
    ax1.set_title('Segmentation DSC across Views', fontsize=16, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(views, fontsize=12)
    ax1.legend(fontsize=10); ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.set_ylabel('HD95 [px] (Lower is better)', fontsize=14)
    ax2.set_title('Boundary Error (HD95) across Views', fontsize=16, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(views, fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "barplot_views.png"), dpi=300)
    plt.close()

def plot_boxplot(raw_results, views):
    """New: DSC box plot for stability analysis"""
    models = [m for m in raw_results.keys() if len(raw_results[m]['A2C']['dsc']) > 0]
    if not models: return
    
    data = []
    for m in models:
        all_dsc = get_flat_metric(raw_results, m, views, 'dsc')
        data.append(all_dsc[~np.isnan(all_dsc)])
        
    plt.figure(figsize=(14, 7))
    plt.boxplot(data, labels=models, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    plt.title('Distribution of Global DSC (Stability Analysis)', fontsize=16, fontweight='bold')
    plt.ylabel('Dice Similarity Coefficient', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "boxplot_dsc.png"), dpi=300)
    plt.close()

def plot_radar_chart(raw_results, views):
    """New: multi-dimensional performance radar chart"""
    models = [m for m in raw_results.keys() if len(raw_results[m]['A2C']['dsc']) > 0]
    if not models: return
    
    # Select 5 most representative models for radar chart plotting
    selected_models = ['ADR-Net (Ours)', 'nnU-Net', 'Swin-UNet', 'Attention-UNet', 'UNet']
    models_to_plot = [m for m in selected_models if m in models]
    
    categories = ['DSC', 'IoU', 'Sensitivity', '1-HD95 (norm)', '1-ASSD (norm)']
    N = len(categories)
    
    # Radar chart angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Distance metric normalization (lower is better, converted to outward expansion = better)
    for m in models_to_plot:
        all_dsc = get_flat_metric(raw_results, m, views, 'dsc')
        all_iou = get_flat_metric(raw_results, m, views, 'iou')
        all_sens = get_flat_metric(raw_results, m, views, 'sens')
        all_hd95 = get_flat_metric(raw_results, m, views, 'hd95')
        all_assd = get_flat_metric(raw_results, m, views, 'assd')
        
        # Normalize to 0-1 range
        norm_hd95 = max(0, 1 - (np.nanmean(all_hd95) / 100.0))
        norm_assd = max(0, 1 - (np.nanmean(all_assd) / 10.0))
        
        values = [np.nanmean(all_dsc), np.nanmean(all_iou), np.nanmean(all_sens), norm_hd95, norm_assd]
        values += values[:1]
        
        linewidth = 3 if 'Ours' in m else 1.5
        linestyle = '-' if 'Ours' in m else '--'
        ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle, label=m)
        if 'Ours' in m:
            ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_title("Multi-dimensional Performance Radar", size=16, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "radar_chart.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    evaluate_models()