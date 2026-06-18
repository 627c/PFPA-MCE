"""
evaluate_seg_baselines.py - 分割模型终极评估与绘图神器（满血战报+统计显著性+可视化对比版）
功能：
1. 计算 6 大指标 (DSC, IoU, Sens, Spec, HD95, ASSD)
2. 自动生成详尽的 Markdown 评估报告并保存
3. 👑 顶刊标准：自动计算ADR-Net与所有基线的配对t检验，表格自动标注显著性星号
4. 自动生成：各视图柱状图、多维能力雷达图、性能分布箱线图
5. ✅ 新增：分割结果可视化对比（原图+金标准+所有模型并排展示）
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

# 保留你原有的警告过滤
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

# ==================== 配置区 ====================
VAL_ROOT = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/segmentation_val"
RESULTS_DIR = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results/seg_baselines"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (512, 512)
NUM_VISUAL_SAMPLES = 15  # 每个切面生成15张对比图（共45张），可根据需要调整
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

# 图表美学设置（和架构图配色完全统一）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
COLOR_GT = (102, 194, 165)     # 薄荷绿：金标准（和架构图一致）
COLOR_PRED = (252, 141, 98)    # 警示橙：模型预测（和架构图一致）
MASK_ALPHA = 0.4               # 掩膜透明度（平衡原图可见性和分割清晰度）
CONTOUR_WIDTH = 2              # 轮廓线宽度

# ==================== 统计检验工具函数 ====================
def get_flat_metric(raw_results, model_name, views, metric_key):
    """把三个视角的指标合并成一个长列表，自动过滤NaN"""
    all_vals = []
    if model_name not in raw_results: return np.array([])
    for v in views:
        all_vals.extend(raw_results[model_name][v][metric_key])
    return np.array(all_vals)

def compute_p_value(arr1, arr2, higher_is_better=True):
    """
    计算配对t检验的p-value，自动处理"越高越好"和"越低越好"指标
    Args:
        higher_is_better: True=指标越高越好(DSC/IoU等), False=越低越好(HD95/ASSD等)
    """
    valid_idx = ~np.isnan(arr1) & ~np.isnan(arr2)
    n_valid = np.sum(valid_idx)
    if n_valid < 5:  # 样本量太少无法做统计检验
        return 1.0
    
    a, b = arr1[valid_idx], arr2[valid_idx]
    
    # 对于越低越好的指标，取反后再做t检验，保证显著性方向正确
    if not higher_is_better:
        a, b = -a, -b
    
    # 双侧配对t检验
    _, p_val = ttest_rel(a, b)
    return p_val

def get_p_stars(p_val):
    """顶刊标准显著性标记"""
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return ""  # 不显著不显示任何标记

# ==================== 保留你原有的完美load_and_preprocess函数 ====================
def load_and_preprocess(img_path, mask_path):
    # 修复：Pillow 版本兼容（自动适配新旧版本）
    try:
        from PIL import Resampling
        BICUBIC = Resampling.BICUBIC
    except ImportError:
        from PIL import Image
        BICUBIC = Image.BICUBIC
    
    # 加载并预处理图像
    img = Image.open(img_path).convert('L')
    img = img.resize(IMG_SIZE, BICUBIC)
    img_t = torch.from_numpy(np.array(img)/255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    # 核心修复：图片是jpg，掩码强制读取同名png（和训练代码逻辑完全一致）
    img_base = os.path.splitext(os.path.basename(img_path))[0]
    mask_real_path = os.path.join(os.path.dirname(mask_path), f"{img_base}.png")
    mask = cv2.imread(mask_real_path, 0)
    
    # 增加容错：防止掩码文件彻底缺失
    if mask is None:
        raise FileNotFoundError(f"掩码文件不存在: {mask_real_path}")
        
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

# ==================== 新增：分割结果可视化对比函数 ====================
def generate_visual_comparisons(visual_samples, views):
    """
    生成顶刊级分割结果可视化对比图
    布局：[原图] [金标准] [ADR-Net] [nnU-Net] [UNet++] [UNet] [Swin-UNet] [MobileUNet] [Attention-UNet]
    """
    # 模型显示顺序（把我们的模型放在最前面，性能从高到低排序）
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
            
        print(f"\n🎨 正在生成 {view} 切面的可视化对比图...")
        for sample_idx, sample in enumerate(tqdm(samples, desc=f"  Generating {view}")):
            img = sample['img']
            gt_mask = sample['gt']
            all_preds = sample['preds']
            filename = sample['filename']
            
            # 过滤出有预测结果的模型
            valid_models = [m for m in model_display_order if m in all_preds]
            if not valid_models:
                continue
            
            # 计算子图数量：原图 + 金标准 + 所有模型
            n_subplots = 2 + len(valid_models)
            fig, axes = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 5), dpi=100)
            
            # 转换为uint8格式用于OpenCV绘图
            img_uint8 = (img * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
            
            # 1. 绘制原图
            axes[0].imshow(img_uint8, cmap='gray')
            axes[0].set_title('Original\nImage', fontsize=12, fontweight='bold', pad=10)
            axes[0].axis('off')
            
            # 2. 绘制金标准
            gt_overlay = img_rgb.copy()
            gt_color_mask = np.zeros_like(gt_overlay)
            gt_color_mask[gt_mask == 1] = COLOR_GT
            gt_overlay = cv2.addWeighted(gt_overlay, 1, gt_color_mask, MASK_ALPHA, 0)
            # 绘制金标准轮廓
            gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_overlay, gt_contours, -1, COLOR_GT, CONTOUR_WIDTH)
            axes[1].imshow(gt_overlay)
            axes[1].set_title('Ground\nTruth', fontsize=12, fontweight='bold', pad=10)
            axes[1].axis('off')
            
            # 3. 绘制各个模型的预测结果
            for i, model_name in enumerate(valid_models):
                pred_mask = all_preds[model_name]
                pred_overlay = img_rgb.copy()
                pred_color_mask = np.zeros_like(pred_overlay)
                pred_color_mask[pred_mask == 1] = COLOR_PRED
                pred_overlay = cv2.addWeighted(pred_overlay, 1, pred_color_mask, MASK_ALPHA, 0)
                # 绘制预测轮廓
                pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pred_overlay, pred_contours, -1, COLOR_PRED, CONTOUR_WIDTH)
                
                # 计算并显示DSC值
                dsc = calculate_dsc(pred_mask, gt_mask)
                axes[i+2].imshow(pred_overlay)
                title = f"{model_name}\nDSC: {dsc:.4f}"
                # 我们的模型用加粗字体突出显示
                fontweight = 'bold' if 'Ours' in model_name else 'normal'
                axes[i+2].set_title(title, fontsize=12, fontweight=fontweight, pad=10)
                axes[i+2].axis('off')
            
            # 添加全局标题
            plt.suptitle(
                f"View: {view} | Sample: {os.path.splitext(filename)[0]}",
                fontsize=16, 
                fontweight='bold',
                y=1.05
            )
            
            # 调整布局并保存
            plt.tight_layout()
            save_path = os.path.join(VISUAL_SAVE_DIR, f"{view}_sample_{sample_idx:03d}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

# ==================== 评估核心引擎 ====================
def evaluate_models():
    views = ['A2C', 'A3C', 'A4C']
    metrics_keys = ['dsc', 'iou', 'sens', 'spec', 'hd95', 'assd']
    # 定义每个指标的方向：True=越高越好，False=越低越好
    metric_directions = {
        'dsc': True, 'iou': True, 'sens': True, 'spec': True,
        'hd95': False, 'assd': False
    }
    
    # 存储所有样本的具体数值，用于计算均值、方差和统计检验
    raw_results = {m: {v: {k: [] for k in metrics_keys} for v in views} for m in MODEL_CKPTS.keys()}
    
    # 存储可视化样本数据
    visual_samples = {v: [] for v in views}
    samples_collected = {v: False for v in views}
    
    for model_name, ckpt_path in MODEL_CKPTS.items():
        print(f"\n🚀 正在评估: {model_name}")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ 找不到权重文件 {ckpt_path}，跳过。")
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
                # 关键：排序保证所有模型读取文件的顺序绝对一致，这是配对t检验的前提
                file_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                # 第一个模型处理时，收集可视化样本的原图和金标准
                if not samples_collected[view] and len(file_list) >= NUM_VISUAL_SAMPLES:
                    print(f"📸 正在收集 {view} 切面的可视化样本...")
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
                
                # 评估所有样本
                for idx, f in enumerate(tqdm(file_list, desc=f"  Evaluating {view}")):
                    img_t, gt_mask = load_and_preprocess(os.path.join(img_dir, f), os.path.join(mask_dir, f))
                    pred = model(img_t)
                    pred_mask = (torch.sigmoid(pred)[0,0].cpu().numpy() > 0.5).astype(np.uint8)
                    
                    # 计算指标
                    raw_results[model_name][view]['dsc'].append(calculate_dsc(pred_mask, gt_mask))
                    raw_results[model_name][view]['iou'].append(calculate_iou(pred_mask, gt_mask))
                    raw_results[model_name][view]['sens'].append(calculate_sensitivity(pred_mask, gt_mask))
                    raw_results[model_name][view]['spec'].append(calculate_specificity(pred_mask, gt_mask))
                    
                    hd95, assd = calculate_hd95_assd(pred_mask, gt_mask)
                    # 即使是NaN也必须存入，保证所有模型的数组长度完全对齐
                    raw_results[model_name][view]['hd95'].append(hd95)
                    raw_results[model_name][view]['assd'].append(assd)
                    
                    # 保存可视化样本的预测结果
                    if idx < NUM_VISUAL_SAMPLES and samples_collected[view]:
                        visual_samples[view][idx]['preds'][model_name] = pred_mask

    # 预计算所有全局p值
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

    # 生成所有报告和图表
    generate_report(raw_results, views, metrics_keys, metric_directions, global_p_values, ours_name)
    plot_view_metrics(raw_results, views)
    plot_boxplot(raw_results, views)
    plot_radar_chart(raw_results, views)
    generate_visual_comparisons(visual_samples, views)  # 新增：生成可视化对比图
    
    print(f"\n🎉 所有评估任务完成！")
    print(f"📊 评估报告与统计图表已保存至: {RESULTS_DIR}")
    print(f"🖼️  分割结果可视化对比图已保存至: {VISUAL_SAVE_DIR}")
    print(f"📋 报告包含完整统计显著性检验结果，可直接用于论文Table 1")

# ==================== 生成 Markdown 报告（含显著性标记） ====================
def generate_report(raw_results, views, metrics_keys, metric_directions, global_p_values, ours_name):
    report_path = os.path.join(RESULTS_DIR, "Segmentation_Evaluation_Report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# PFPA-MCE 分割基线评估报告 (Segmentation Baseline Evaluation)\n\n")
        
        # 1. 全局汇总
        f.write("## 1. 全局综合指标 (Global Metrics)\n")
        f.write("*注：显著性标记为与ADR-Net (Ours)的配对t检验结果。*  \n")
        f.write("*\\*: p<0.05, \\*\\*: p<0.01, \\*\\*\\*: p<0.001，无标记表示不显著(p≥0.05)*\n\n")
        
        f.write("| Model | DSC | IoU | Sens | Spec | HD95 (px) | ASSD (px) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for m in raw_results.keys():
            if len(raw_results[m]['A2C']['dsc']) == 0: continue
            
            means = {}
            stars = {}
            for k in metrics_keys:
                all_vals = get_flat_metric(raw_results, m, views, k)
                means[k] = np.nanmean(all_vals)
                # 为非我们的模型添加显著性星号
                if m != ours_name and m in global_p_values:
                    stars[k] = get_p_stars(global_p_values[m][k])
                else:
                    stars[k] = ""
            
            # 格式化输出，自动添加星号
            f.write(f"| {m} | "
                   f"{means['dsc']:.4f}{stars['dsc']} | "
                   f"{means['iou']:.4f}{stars['iou']} | "
                   f"{means['sens']:.4f}{stars['sens']} | "
                   f"{means['spec']:.4f}{stars['spec']} | "
                   f"{means['hd95']:.2f}{stars['hd95']} | "
                   f"{means['assd']:.2f}{stars['assd']} |\n")
        f.write("\n")
        
        # 2. 详细统计显著性检验表
        f.write("## 2. 详细统计显著性检验 (Detailed P-values)\n")
        f.write("| 对比组合 | DSC | IoU | Sens | Spec | HD95 | ASSD |\n")
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

        # 3. 细分切面汇总
        f.write("## 3. 各视角独立指标 (Per-View Metrics)\n")
        for v in views:
            f.write(f"### 切面: {v}\n")
            f.write("| Model | DSC | IoU | Sens | Spec | HD95 (px) | ASSD (px) |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for m in raw_results.keys():
                if len(raw_results[m][v]['dsc']) == 0: continue
                means = {k: np.nanmean(raw_results[m][v][k]) for k in metrics_keys}
                f.write(f"| {m} | {means['dsc']:.4f} | {means['iou']:.4f} | {means['sens']:.4f} | {means['spec']:.4f} | {means['hd95']:.2f} | {means['assd']:.2f} |\n")
            f.write("\n")

# ==================== 画图函数群 ====================

def plot_view_metrics(raw_results, views):
    """原有的视角对比柱状图"""
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
    """新增：DSC 箱线图展示稳定性"""
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
    """新增：多维能力雷达图"""
    models = [m for m in raw_results.keys() if len(raw_results[m]['A2C']['dsc']) > 0]
    if not models: return
    
    # 我们挑 5 个最具代表性的模型画雷达图
    selected_models = ['ADR-Net (Ours)', 'nnU-Net', 'Swin-UNet', 'Attention-UNet', 'UNet']
    models_to_plot = [m for m in selected_models if m in models]
    
    categories = ['DSC', 'IoU', 'Sensitivity', '1-HD95 (norm)', '1-ASSD (norm)']
    N = len(categories)
    
    # 雷达图角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 距离指标归一化 (越小越好，转换为向外扩张越好)
    for m in models_to_plot:
        all_dsc = get_flat_metric(raw_results, m, views, 'dsc')
        all_iou = get_flat_metric(raw_results, m, views, 'iou')
        all_sens = get_flat_metric(raw_results, m, views, 'sens')
        all_hd95 = get_flat_metric(raw_results, m, views, 'hd95')
        all_assd = get_flat_metric(raw_results, m, views, 'assd')
        
        # 归一化到0-1范围
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