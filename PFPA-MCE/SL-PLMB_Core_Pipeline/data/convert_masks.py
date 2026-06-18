"""
convert_masks.py - 分切面递归版
自动转换所有子文件夹中的彩色掩膜为单通道黑白图
"""
import os
import cv2
import numpy as np
from glob import glob

# ✅ 只需要修改这两个根目录路径
TRAIN_MASK_ROOT = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/segmentation_train'
VAL_MASK_ROOT = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/data/segmentation_val'

def convert_single_directory(mask_dir):
    """转换单个目录下的所有掩膜"""
    # 递归查找所有.png文件
    mask_files = glob(os.path.join(mask_dir, '**', '*.png'), recursive=True)
    print(f"在 {mask_dir} 中找到 {len(mask_files)} 个掩膜文件")
    
    success_count = 0
    for path in mask_files:
        try:
            # 1. 用彩色模式读取（必须！）
            img = cv2.imread(path)
            if img is None:
                print(f"警告：无法读取 {path}")
                continue
            
            # 2. 提取红色通道 (OpenCV是BGR格式，红色在第2通道)
            red_channel = img[:, :, 2]
            
            # 3. 二值化：红色区域转为纯白(255)，其余转为纯黑(0)
            _, binary_mask = cv2.threshold(red_channel, 100, 255, cv2.THRESH_BINARY)
            
            # 4. 覆盖保存为单通道灰度图
            cv2.imwrite(path, binary_mask)
            success_count += 1
            
        except Exception as e:
            print(f"错误：转换 {path} 失败: {str(e)}")
    
    print(f"成功转换 {success_count} 个掩膜\n")
    return success_count

if __name__ == "__main__":
    print("=" * 60)
    print("开始转换训练集掩膜...")
    print("=" * 60)
    train_success = convert_single_directory(TRAIN_MASK_ROOT)
    
    print("=" * 60)
    print("开始转换验证集掩膜...")
    print("=" * 60)
    val_success = convert_single_directory(VAL_MASK_ROOT)
    
    print("=" * 60)
    print(f"✅ 全部转换完成！总共转换 {train_success + val_success} 个掩膜")
    print("=" * 60)