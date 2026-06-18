"""
generate_all_visualizations.py 修正版
修正点：
1. 修正牛眼图基底/中段30°角度偏移，1号节段对准顶部
2. 修正心尖段分割线错位问题，分割线对齐节段边界
3. 严格对齐AHA 17节段标准顺序
4. 调整色条参数，解决色条遮挡主图问题
5. 保留RBF插值逻辑不动
6. 帧逻辑、平滑、切面图其他逻辑保持与原版一致
"""
import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from scipy.interpolate import Rbf
from scipy.signal import savgol_filter
sys.path.append("/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE")
from config import PATHS
from core.tac import time_frequency_filter, detect_heart_rate
from core.perfusion import generate_perfusion_maps
from core.anatomy import generate_aha17_mask

# ===================== 全局配置（与主程序完全对齐）=====================
# 可视化背景标注帧
FRAME_CONFIG = {
    "P001_A4C": 160, "P001_A3C": 204, "P002_A3C": 100, "P002_A4C": 284,
    "P003_A2C": 135, "P003_A3C": 112, "P003_A4C": 180, "P004_A2C": 424,
    "P004_A3C": 150, "P004_A4C": 360, "P005_A2C": 191, "P005_A3C": 121, "P005_A4C": 118,
}
# 运算用空白Flash帧（手动指定 0下标）
FLASH_FRAME_CONFIG = {
    "P001_A4C": 61,
    "P001_A3C": 68,
    "P002_A3C": 5,
    "P002_A4C": 11,
    "P003_A2C": 13,
    "P003_A3C": 29,
    "P003_A4C": 37,
    "P004_A2C": 1,
    "P004_A3C": 31,
    "P004_A4C": 30,
    "P005_A2C": 29,
    "P005_A3C": 68,
    "P005_A4C": 37,
}

POST_FLASH_SECONDS = 5.0
ERODE_KERNEL_SIZE = 3
IMG_SIZE = (512, 512)
MIN_VALID_FRAME_DURATION = 1.0
MIN_MASK_AREA = 20
EPS = 1e-6

PHYSICAL_SCALES_META = {
    'A': ('Blood Volume (A)', 'a.u.'),
    'beta': ('Velocity (β)', 's⁻¹'),
    'mbf': ('Blood Flow (MBF)', 'a.u./s')
}

# AHA 17节段环形划分：(内半径, 外半径, 节段数, 起始索引)
# 基底(0-5→1-6)、中段(6-11→7-12)、心尖段(12-15→13-16)、心尖帽(16→17)
AHA17_RINGS = [
    (0.6, 1.0, 6, 0),   # 基底段
    (0.3, 0.6, 6, 6),   # 中段
    (0.1, 0.3, 4, 12),  # 心尖段
    (0.0, 0.1, 1, 16)   # 心尖帽
]

# 心尖段4个节段的中心角度（顺时针：0°顶部=前壁，90°右侧=间隔壁，180°底部=下壁，270°左侧=侧壁）
# 对应节段：13(前壁心尖)、14(间隔心尖)、15(下壁心尖)、16(侧壁心尖)，严格符合AHA标准
APICAL_CENTER_ANGLES = [0.0, np.pi/2, np.pi, 3*np.pi/2]
# 心尖段分割线角度（节段边界，每个节段占90°，分割线在中心±45°位置）
APICAL_EDGE_ANGLES = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]

# 绘图样式
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE))


# ===================== 工具函数 =====================
def restore_mask_size_perfect(mask_512, orig_h, orig_w):
    """掩膜还原（与原版逻辑完全一致）"""
    target_h, target_w = IMG_SIZE
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    left_pad = (target_w - new_w) // 2
    top_pad = (target_h - new_h) // 2
    mask_cropped = mask_512[top_pad:top_pad+new_h, left_pad:left_pad+new_w]
    mask_restored_float = cv2.resize(mask_cropped.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return (mask_restored_float > 127).astype(np.uint8)


def savgol_smooth(tac, fps):
    """与主程序统一的Savitzky-Golay平滑"""
    window_length = int(fps)
    if window_length % 2 == 0:
        window_length += 1
    if len(tac) <= window_length:
        return tac
    return savgol_filter(tac, window_length=window_length, polyorder=3)


def compute_global_scales(fused_data, eligible_patients):
    """全局色阶统一计算"""
    max_vals = {'A': 0.0, 'beta': 0.0, 'mbf': 0.0}
    for pid in eligible_patients:
        for view in ['A2C', 'A3C', 'A4C']:
            if view in fused_data[pid]:
                for seg_val in fused_data[pid][view]['segments'].values():
                    for p in ['A', 'beta', 'mbf']:
                        if seg_val[p] > max_vals[p]:
                            max_vals[p] = max(max_vals[p], seg_val[p])
    return {
        'A': (0.0, np.ceil(max_vals['A'] * 1.1 / 10) * 10),
        'beta': (0.0, np.ceil(max_vals['beta'] * 1.1)),
        'mbf': (0.0, np.ceil(max_vals['mbf'] * 1.1 / 100) * 100)
    }


# ===================== 切面热力图绘制 =====================
def plot_beautiful_heatmap(bg_gray, param_map, mask_bin, seg_mask, save_path, param_type, title, global_scales):
    """切面灌注热力图（像素级+分段标注）"""
    vmin, vmax = global_scales[param_type]
    _, unit = PHYSICAL_SCALES_META[param_type]
    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 像素级热力图叠加
    color_overlay = np.zeros((*bg_gray.shape, 4))
    mask_idx = (mask_bin > 0)
    color_overlay[mask_idx] = cmap(norm(param_map[mask_idx]))
    color_overlay[mask_idx, 3] = 0.65

    # 分段轮廓与编号
    seg_centers = []
    for seg_id in range(1, 18):
        seg_idx = (seg_mask == seg_id)
        if seg_idx.sum() > 0:
            ys, xs = np.where(seg_idx)
            cy, cx = np.mean(ys), np.mean(xs)
            seg_centers.append((cx, cy, seg_id))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # 左图：纯像素级灌注
    ax1.imshow(bg_gray, cmap='gray', vmin=0, vmax=255)
    ax1.imshow(color_overlay)
    ax1.axis('off')
    ax1.set_title(f'{title}\n(Pixel-level Perfusion)', fontsize=14, fontweight='bold')

    # 右图：像素级灌注+分段编号
    ax2.imshow(bg_gray, cmap='gray', vmin=0, vmax=255)
    ax2.imshow(color_overlay)
    for cx, cy, sid in seg_centers:
        ax2.text(cx, cy, str(sid), color='white', fontsize=10, fontweight='bold',
                 ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7, pad=1))
    ax2.axis('off')
    ax2.set_title(f'{title}\n(AHA Segmented Annotation)', fontsize=14, fontweight='bold')

    # 统一色条（修正pad，避免遮挡主图）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.02, pad=0.08)
    cbar.set_label(f'{unit}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===================== 牛眼图绘制（核心修正，RBF插值完全保留）=====================
def plot_beautiful_bullseye(seg_data_17, save_path, param_type, title, global_scales):
    """
    AHA 17节段标准牛眼图（角度/顺序修正版）
    :param seg_data_17: 长度为17的数组，索引0对应节段1，索引16对应节段17
    """
    vmin, vmax = global_scales[param_type]
    _, unit = PHYSICAL_SCALES_META[param_type]
    cmap = plt.cm.jet

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_direction(-1)       # 顺时针方向
    ax.set_theta_offset(np.pi / 2)   # 0°对准顶部（12点钟方向）
    ax.set_ylim(0, 1.0)

    # 极坐标转直角坐标（用于RBF插值锚点）
    def polar2cart(r, theta):
        return r * np.cos(np.pi/2 - theta), r * np.sin(np.pi/2 - theta)

    valid_segs = [i for i, val in enumerate(seg_data_17) if val > 1e-4]
    zero_segs = [i for i, val in enumerate(seg_data_17) if val <= 1e-4]

    # ---------- 1. 收集RBF插值锚点（RBF逻辑完全保留，仅修正角度）----------
    anchors_x, anchors_y, anchors_val = [], [], []
    for r_inner, r_outer, n_segs, start_idx in AHA17_RINGS:
        if start_idx == 12:
            # 心尖段4个节段（角度修正：中心位置正确）
            for i in range(n_segs):
                seg_pos = start_idx + i
                if seg_pos in valid_segs:
                    r_mid = (r_inner + r_outer) / 2.0
                    val = min(max(seg_data_17[seg_pos], vmin), vmax)
                    x, y = polar2cart(r_mid, APICAL_CENTER_ANGLES[i])
                    anchors_x.append(x)
                    anchors_y.append(y)
                    anchors_val.append(val)
        else:
            # 基底、中段、心尖帽（修正：整体左移30°，让1号节段中心对准顶部）
            theta_edges = np.linspace(-np.pi/6, 2*np.pi - np.pi/6, n_segs + 1)
            for i in range(n_segs):
                seg_pos = start_idx + i
                if seg_pos in valid_segs:
                    theta_mid = (theta_edges[i] + theta_edges[i+1]) / 2.0
                    r_mid = (r_inner + r_outer) / 2.0
                    val = min(max(seg_data_17[seg_pos], vmin), vmax)
                    x, y = polar2cart(r_mid, theta_mid)
                    anchors_x.append(x)
                    anchors_y.append(y)
                    anchors_val.append(val)
                    # 基底段额外添加外边界锚点，保证边缘平滑（保留原逻辑）
                    if start_idx == 0:
                        x_e, y_e = polar2cart(1.0, theta_mid)
                        anchors_x.append(x_e)
                        anchors_y.append(y_e)
                        anchors_val.append(val)

    # 添加中心锚点（心尖帽，保留原逻辑）
    if 16 in valid_segs:
        anchors_x.append(0.0)
        anchors_y.append(0.0)
        anchors_val.append(min(max(seg_data_17[16], vmin), vmax))

    # ---------- 2. RBF平滑插值生成像素级牛眼图（完全保留原逻辑不动）----------
    grid_r, grid_t = np.mgrid[0:1.0:200j, 0:2*np.pi:360j]
    grid_x = grid_r * np.cos(np.pi/2 - grid_t)
    grid_y = grid_r * np.sin(np.pi/2 - grid_t)

    if len(anchors_val) >= 4:
        rbf = Rbf(anchors_x, anchors_y, anchors_val, function='thin_plate', smooth=0.05)
        grid_z = rbf(grid_x, grid_y)
        grid_z = np.clip(grid_z, vmin, vmax)
        grid_z[grid_r > 1.0] = np.nan
        mesh = ax.pcolormesh(grid_t, grid_r, grid_z, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud', zorder=1)
    else:
        mesh = ax.pcolormesh(grid_t, grid_r, np.zeros_like(grid_r), cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)

    # ---------- 3. 无效节段填充（修正角度对齐）----------
    for r_inner, r_outer, n_segs, start_idx in AHA17_RINGS:
        if start_idx == 12:
            # 心尖段无效节段
            for i in range(n_segs):
                seg_pos = start_idx + i
                if seg_pos in zero_segs:
                    theta_start = APICAL_EDGE_ANGLES[i-1] if i > 0 else APICAL_EDGE_ANGLES[-1] - 2*np.pi
                    theta_end = APICAL_EDGE_ANGLES[i]
                    ax.fill_between(np.linspace(theta_start, theta_end, 50),
                                    r_inner, r_outer, facecolor='#F0F0F0', edgecolor='#A0A0A0',
                                    hatch='////', linewidth=1.2, zorder=2)
        else:
            # 其他环无效节段（修正角度偏移）
            theta_edges = np.linspace(-np.pi/6, 2*np.pi - np.pi/6, n_segs + 1)
            for i in range(n_segs):
                seg_pos = start_idx + i
                if seg_pos in zero_segs:
                    ax.fill_between(np.linspace(theta_edges[i], theta_edges[i+1], 50),
                                    r_inner, r_outer, facecolor='#F0F0F0', edgecolor='#A0A0A0',
                                    hatch='////', linewidth=1.2, zorder=2)

    # ---------- 4. 绘制分割线（核心修正：心尖段分割线对齐节段边界）----------
    for r_inner, r_outer, n_segs, start_idx in AHA17_RINGS:
        # 环形分隔线（保留原逻辑）
        ax.plot(np.linspace(0, 2*np.pi, 200), np.ones(200)*r_outer,
                color='white', linewidth=1.2, alpha=0.9, zorder=3)
        # 心尖帽无径向分割线（保留原逻辑）
        if start_idx == 16:
            continue
        # 心尖段径向分割线（修正：使用节段边界角度，而非中心角度）
        if start_idx == 12:
            for theta in APICAL_EDGE_ANGLES:
                ax.plot([theta, theta], [r_inner, r_outer],
                        color='white', linewidth=1.2, alpha=0.9, zorder=3)
        # 基底/中段径向分割线（修正：对齐偏移后的边界）
        else:
            theta_edges = np.linspace(-np.pi/6, 2*np.pi - np.pi/6, n_segs + 1)
            for i in range(n_segs):
                ax.plot([theta_edges[i], theta_edges[i]], [r_inner, r_outer],
                        color='white', linewidth=1.2, alpha=0.9, zorder=3)

    # ---------- 5. 节段编号标注（修正位置对齐节段中心）----------
    for r_in, r_out, n_segs, start_idx in AHA17_RINGS:
        r_mid = (r_in + r_out) / 2.0
        if start_idx == 12:
            # 心尖段编号（中心位置正确）
            for i in range(n_segs):
                seg_idx = start_idx + i
                val = seg_data_17[seg_idx]
                ratio = (val - vmin) / (vmax - vmin + EPS)
                col = '#666666' if val <= 1e-4 else ('white' if (ratio < 0.35 or ratio > 0.8) else 'black')
                ax.text(APICAL_CENTER_ANGLES[i], r_mid, str(seg_idx+1),
                        ha='center', va='center', fontsize=14, fontweight='bold', color=col, zorder=4)
        elif start_idx == 16:
            # 心尖帽单独中心标注，跳过循环（保留原逻辑，避免重复）
            continue
        else:
            # 基底/中段编号（修正：对齐偏移后的中心）
            theta_edges = np.linspace(-np.pi/6, 2*np.pi - np.pi/6, n_segs + 1)
            for i in range(n_segs):
                seg_idx = start_idx + i
                val = seg_data_17[seg_idx]
                ratio = (val - vmin) / (vmax - vmin + EPS)
                col = '#666666' if val <= 1e-4 else ('white' if (ratio < 0.35 or ratio > 0.8) else 'black')
                theta_mid = (theta_edges[i] + theta_edges[i+1]) / 2.0
                ax.text(theta_mid, r_mid, str(seg_idx+1),
                        ha='center', va='center', fontsize=14, fontweight='bold', color=col, zorder=4)

    # 心尖帽17号中心标注（仅标注一次，修复重复问题）
    val = seg_data_17[16]
    ratio = (val - vmin) / (vmax - vmin + EPS)
    col = '#666666' if val <= 1e-4 else ('white' if (ratio < 0.35 or ratio > 0.8) else 'black')
    ax.text(0.0, 0.0, '17', ha='center', va='center', fontsize=14, fontweight='bold', color=col, zorder=4)

    # ---------- 6. 样式与输出（修正色条位置，避免遮挡）----------
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', y=1.08)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label(f'{unit}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===================== 主流程 =====================
def main():
    print("=" * 80)
    print("🎨 灌注可视化引擎（牛眼图角度/顺序修正版，保留RBF插值）")
    print("=" * 80)

    FIG_ROOT = os.path.join(PATHS['results_root'], 'figures_final')
    VIDEO_DIR = PATHS['video_input']
    MASK_DIR = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/raw_data/pseudo_masks"
    EVAL_DIR = os.path.join(PATHS['results_root'], 'evaluation')

    os.makedirs(FIG_ROOT, exist_ok=True)

    # 加载融合结果
    fused_path = os.path.join(EVAL_DIR, 'method_C_fused.json')
    if not os.path.exists(fused_path):
        print("❌ 请先运行主流水线生成method_C_fused.json")
        return
    with open(fused_path, 'r', encoding='utf-8') as f:
        fused_data = json.load(f)

    # 筛选同时具备三个切面的患者
    eligible = [pid for pid, views in fused_data.items()
                if all(v in views for v in ['A2C', 'A3C', 'A4C'])]
    if not eligible:
        print("⚠️  未找到满足条件的患者数据")
        return

    GLOBAL_SCALES = compute_global_scales(fused_data, eligible)

    # 预加载所有视频路径
    all_video_files = {}
    for f in os.listdir(VIDEO_DIR):
        if f.endswith(('.avi', '.mp4')):
            base = os.path.splitext(f)[0]
            all_video_files[base] = os.path.join(VIDEO_DIR, f)

    # 逐患者生成可视化
    for pid in tqdm(eligible, desc="生成可视化结果"):
        pat_dir = os.path.join(FIG_ROOT, f'Patient_{pid}')
        os.makedirs(pat_dir, exist_ok=True)

        # 牛眼图累加数据
        bull_data = {
            'A': np.zeros(17),
            'beta': np.zeros(17),
            'mbf': np.zeros(17),
            'count': np.zeros(17)
        }

        for view in ['A2C', 'A3C', 'A4C']:
            base_name = f"{pid}_{view}"
            if base_name not in all_video_files:
                continue
            vid_path = all_video_files[base_name]
            mask_path = os.path.join(MASK_DIR, f'{base_name}.png')

            if not os.path.exists(vid_path) or not os.path.exists(mask_path):
                continue

            # 读取视频
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cap.release()
            video_full = np.array(frames)
            total_frames, orig_h, orig_w = video_full.shape
            if total_frames < 10:
                continue

            # Flash帧截取
            flash_idx = FLASH_FRAME_CONFIG.get(f"{pid}_{view}", 0)
            flash_idx = max(0, min(flash_idx, total_frames - 1))
            post_flash_frames = int(POST_FLASH_SECONDS * fps)
            start_idx = flash_idx + 1
            end_idx = min(start_idx + post_flash_frames, total_frames)
            if end_idx - start_idx < int(MIN_VALID_FRAME_DURATION * fps):
                continue
            video_tensor = video_full[start_idx:end_idx]

            # 掩膜处理
            mask_512 = cv2.imread(mask_path, 0)
            mask_restored = restore_mask_size_perfect(mask_512, orig_h, orig_w)
            mask_bin = cv2.erode(mask_restored, kernel)
            if np.sum(mask_bin) < MIN_MASK_AREA:
                continue

            # AHA分段
            seg_mask = generate_aha17_mask(mask_bin, view)

            # 像素级TAC平滑
            ys, xs = np.where(mask_bin > 0)
            if len(ys) == 0:
                continue
            purified = np.zeros_like(video_tensor, dtype=np.float32)
            for y, x in zip(ys, xs):
                purified[:, y, x] = savgol_smooth(video_tensor[:, y, x], fps)

            # 生成灌注参数图
            A_map, beta_map, mbf_map = generate_perfusion_maps(purified, mask_bin)

            # 标注帧背景
            label_frame_idx = FRAME_CONFIG.get(f"{pid}_{view}", 0)
            label_frame_idx = max(0, min(label_frame_idx, total_frames - 1))
            bg_frame = video_full[label_frame_idx].copy()

            # 绘制切面热力图
            plot_beautiful_heatmap(bg_frame, mbf_map, mask_bin, seg_mask,
                                   os.path.join(pat_dir, f'{view}_MBF.png'),
                                   'mbf', f'{pid} {view} - MBF', GLOBAL_SCALES)
            plot_beautiful_heatmap(bg_frame, A_map, mask_bin, seg_mask,
                                   os.path.join(pat_dir, f'Sup_{view}_A.png'),
                                   'A', f'{pid} {view} - Blood Volume', GLOBAL_SCALES)
            plot_beautiful_heatmap(bg_frame, beta_map, mask_bin, seg_mask,
                                   os.path.join(pat_dir, f'Sup_{view}_beta.png'),
                                   'beta', f'{pid} {view} - Velocity', GLOBAL_SCALES)

            # 累加节段数据用于牛眼图
            for seg_id in range(1, 18):
                seg_key = f'seg_{seg_id}'
                if seg_key in fused_data[pid][view]['segments']:
                    val = fused_data[pid][view]['segments'][seg_key]
                    if val['mbf'] > 1e-4:
                        idx = seg_id - 1
                        bull_data['A'][idx] += val['A']
                        bull_data['beta'][idx] += val['beta']
                        bull_data['mbf'][idx] += val['mbf']
                        bull_data['count'][idx] += 1

        # 绘制牛眼图
        for param in ['A', 'beta', 'mbf']:
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_vals = np.where(bull_data['count'] > 0,
                                     bull_data[param] / bull_data['count'], 0.0)
            plot_beautiful_bullseye(avg_vals,
                                     os.path.join(pat_dir, f'Bullseye_{param}.png'),
                                     param,
                                     f'{pid} - 17-Segment {PHYSICAL_SCALES_META[param][0]}',
                                     GLOBAL_SCALES)

    print(f"\n🎉 所有可视化结果已生成至：{FIG_ROOT}")


if __name__ == "__main__":
    main()