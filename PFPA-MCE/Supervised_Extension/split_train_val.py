"""
split_train_val.py - 按患者自动划分训练集和验证集
确保同一患者的所有视频都在同一集合，避免数据泄露
"""
import os
import shutil
import numpy as np
import random  # ✅ 新增：导入random模块
from utils.helpers import extract_patient_and_view
from config import BASE_CONFIG  # ✅ 新增：导入配置文件

def main():
    # ✅ 必须在最开头锁死所有随机种子！保证每次划分绝对一致！
    np.random.seed(BASE_CONFIG['seed'])
    random.seed(BASE_CONFIG['seed'])
    
    source_dir = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/training/'
    train_dir = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/train/'
    val_dir = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/val/'
    
    # 创建目标目录
    for split in [train_dir, val_dir]:
        for subdir in ['images', 'masks', 'perfusion']:
            os.makedirs(os.path.join(split, subdir), exist_ok=True)
    
    # 获取所有样本
    all_samples = [f for f in os.listdir(os.path.join(source_dir, 'images')) if f.endswith('.npy')]
    print(f"找到 {len(all_samples)} 个样本")
    
    # 按患者分组
    patient_samples = {}
    for sample in all_samples:
        base_name = os.path.splitext(sample)[0]
        patient_id, _ = extract_patient_and_view(base_name)
        if patient_id not in patient_samples:
            patient_samples[patient_id] = []
        patient_samples[patient_id].append(base_name)
    
    unique_patients = sorted(list(patient_samples.keys()))
    print(f"找到 {len(unique_patients)} 个独立患者")
    
    # 随机划分（7:3）
    np.random.shuffle(unique_patients)
    split_idx = int(len(unique_patients) * 0.7)
    train_patients = set(unique_patients[:split_idx])
    val_patients = set(unique_patients[split_idx:])
    
    print(f"训练集患者: {len(train_patients)} 个")
    print(f"验证集患者: {len(val_patients)} 个")
    
    # 复制文件
    train_count = 0
    val_count = 0
    
    for patient_id, samples in patient_samples.items():
        target_dir = train_dir if patient_id in train_patients else val_dir
        count = train_count if patient_id in train_patients else val_count
        
        for base_name in samples:
            # 复制图像
            src_img = os.path.join(source_dir, 'images', f'{base_name}.npy')
            dst_img = os.path.join(target_dir, 'images', f'{base_name}.npy')
            shutil.copy(src_img, dst_img)
            
            # 复制掩膜
            src_mask = os.path.join(source_dir, 'masks', f'{base_name}.png')
            dst_mask = os.path.join(target_dir, 'masks', f'{base_name}.png')
            shutil.copy(src_mask, dst_mask)
            
            # 复制灌注标签
            src_perf = os.path.join(source_dir, 'perfusion', f'{base_name}.npy')
            dst_perf = os.path.join(target_dir, 'perfusion', f'{base_name}.npy')
            shutil.copy(src_perf, dst_perf)
            
            count += 1
        
        if patient_id in train_patients:
            train_count = count
        else:
            val_count = count
    
    print(f"\n划分完成！")
    print(f"训练集样本: {train_count} 个")
    print(f"验证集样本: {val_count} 个")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")

if __name__ == "__main__":
    main()