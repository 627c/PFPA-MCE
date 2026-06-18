"""
helpers.py - 终极安全与临床严谨版 (新增三元组时间轴裂变扩增)
"""
import os, shutil
import numpy as np

def clear_folder(folder_path):
    if not os.path.exists(folder_path): return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path): os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception: pass

def extract_patient_and_view(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    name_lower = basename.lower()
    view = 'Unknown'
    if '2c' in name_lower or 'a2c' in name_lower: view = 'A2C'
    elif '3c' in name_lower or 'a3c' in name_lower: view = 'A3C'
    elif '4c' in name_lower or 'a4c' in name_lower: view = 'A4C'
        
    if '_' in basename:
        patient_id = basename.split('_')[0]
    else:
        import re
        match = re.match(r'(case\d+)', name_lower)
        patient_id = match.group(1) if match else basename
    return patient_id, view

def extract_temporal_triplet(video_matrix):
    """标准的单次三元组提取 (用于测试集/推理)"""
    T = video_matrix.shape[0]
    if T < 10: return np.stack([video_matrix[0]] * 3, axis=0), (0, 0, 0)
    
    global_intensity = np.mean(video_matrix, axis=(1, 2))
    destroy_idx = np.argmax(global_intensity)
    if destroy_idx > 0 and destroy_idx < T - 1:
        if global_intensity[destroy_idx] < np.mean(global_intensity[:destroy_idx]) * 1.5:
            destroy_idx = np.argmin(global_intensity)
            
    post_flash_intensity = global_intensity[destroy_idx + 1:]
    if len(post_flash_intensity) < 5: return np.stack([video_matrix[0]] * 3, axis=0), (0, 0, 0)
        
    smoothed = np.convolve(post_flash_intensity, np.ones(5)/5, mode='same')
    plateau_idx = min(destroy_idx + 1 + np.argmax(smoothed), T - 1)
    if plateau_idx <= destroy_idx: plateau_idx = T - 1
    
    target_mid = global_intensity[destroy_idx] + (global_intensity[plateau_idx] - global_intensity[destroy_idx]) * 0.5
    mid_idx = destroy_idx
    min_diff = float('inf')
    for i in range(destroy_idx + 1, plateau_idx):
        diff = abs(global_intensity[i] - target_mid)
        if diff < min_diff:
            min_diff, mid_idx = diff, i
    if mid_idx <= destroy_idx or mid_idx >= plateau_idx:
        mid_idx = destroy_idx + max(1, (plateau_idx - destroy_idx) // 2)
        
    triplet = np.stack([video_matrix[destroy_idx], video_matrix[mid_idx], video_matrix[plateau_idx]], axis=0)
    return triplet, (destroy_idx, mid_idx, plateau_idx)

def extract_augmented_triplets(video_matrix, num_samples=10):
    """👑 时间轴裂变扩增 (Data Fission): 提取10个轻微时间偏移的三元组"""
    T = video_matrix.shape[0]
    if T < 10: return [extract_temporal_triplet(video_matrix)]
        
    global_intensity = np.mean(video_matrix, axis=(1, 2))
    destroy_idx = np.argmax(global_intensity)
    if destroy_idx > 0 and destroy_idx < T - 1:
        if global_intensity[destroy_idx] < np.mean(global_intensity[:destroy_idx]) * 1.5:
            destroy_idx = np.argmin(global_intensity)
            
    post_flash_intensity = global_intensity[destroy_idx + 1:]
    if len(post_flash_intensity) < 5: return [extract_temporal_triplet(video_matrix)]
        
    smoothed = np.convolve(post_flash_intensity, np.ones(5)/5, mode='same')
    plateau_idx_base = min(destroy_idx + 1 + np.argmax(smoothed), T - 1)
    if plateau_idx_base <= destroy_idx: plateau_idx_base = T - 1
    
    target_mid = global_intensity[destroy_idx] + (global_intensity[plateau_idx_base] - global_intensity[destroy_idx]) * 0.5
    mid_idx_base, min_diff = destroy_idx, float('inf')
    for i in range(destroy_idx + 1, plateau_idx_base):
        diff = abs(global_intensity[i] - target_mid)
        if diff < min_diff: min_diff, mid_idx_base = diff, i
    if mid_idx_base <= destroy_idx or mid_idx_base >= plateau_idx_base:
        mid_idx_base = destroy_idx + max(1, (plateau_idx_base - destroy_idx) // 2)

    triplets = []
    for i in range(num_samples):
        m_idx = max(destroy_idx + 1, min(mid_idx_base + int(np.random.uniform(-2, 3)), plateau_idx_base - 2))
        p_min, p_max = max(m_idx + 5, plateau_idx_base - 30), min(T - 1, plateau_idx_base + 15)
        p_idx = int(np.random.uniform(p_min, p_max + 1)) if p_max > p_min else plateau_idx_base
        triplet = np.stack([video_matrix[destroy_idx], video_matrix[m_idx], video_matrix[p_idx]], axis=0)
        triplets.append((triplet, (destroy_idx, m_idx, p_idx)))
    return triplets