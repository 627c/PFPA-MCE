"""
config.py - 全局配置（修复所有路径斜杠）
"""
import os
import torch

BASE_CONFIG = {
    'project_name': 'PFPA_MCE_Analysis',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 42,
    'img_size': 512,
    'fps': 30,
    'ensure_divisible_by': 16,
}

# ===================== 【所有路径全部修复】 =====================
PATHS = {
    'video_input': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/raw_data/videos',
    'original_view_dirs': {
        'A2C': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/raw_data/A2C',
        'A3C': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/raw_data/A3C',
        'A4C': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/raw_data/A4C'
    },
    'train_images': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/train/images',
    'train_masks': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/train/masks',
    'train_perfusion': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/train/perfusion',
    'val_images': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/val/images',
    'val_masks': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/val/masks',
    'val_perfusion': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/val/perfusion',
    'test_images': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/test/images',
    'test_masks': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/test/masks',
    'test_perfusion': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/test/perfusion',
    'aha17_dir': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/aha17',

    'model_save': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/checkpoints/best_model.pth',
    'results_root': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results',
    'plmb_dir': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results/global_plmb',
    'tac_output': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results/tac_signals',
    'fitting_output': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results/tac_curves',
    'evaluation_output': '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results/evaluation',
}

TRAIN_PARAMS = {
    'epochs': 80,
    'batch_size': 8,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'reduction': 16,
    'dice_weight': 0.7,
    'bce_weight': 0.5,
    'mse_weight': 0.5,
    'pos_weight': 10.0,
}

SIGNAL_PARAMS = {
    'stft_nperseg': 64,
    'freq_low': 0.5,
    'freq_high': 5.0,
    'sd_threshold': 15.0,
    'fusion_alpha': 0.2,
    'min_heart_rate': 60,
    'max_heart_rate': 100,
}

for path in PATHS.values():
    if isinstance(path, str) and not path.endswith('.pth'):
        os.makedirs(path, exist_ok=True)
    elif isinstance(path, dict):
        for subpath in path.values():
            os.makedirs(subpath, exist_ok=True)

print("=" * 60)
print("PFPA-MCE 系统配置加载完成")
print("=" * 60)