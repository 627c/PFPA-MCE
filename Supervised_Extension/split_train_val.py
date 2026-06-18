"""
split_train_val.py - Automatically split training and validation sets by patient
Ensures all videos from the same patient are in the same set to avoid data leakage
"""
import os
import shutil
import numpy as np
import random  
from utils.helpers import extract_patient_and_view
from config import BASE_CONFIG  

def main():
    np.random.seed(BASE_CONFIG['seed'])
    random.seed(BASE_CONFIG['seed'])
    
    source_dir = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/training/'
    train_dir = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/train/'
    val_dir = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/val/'
    
    # Create target directories
    for split in [train_dir, val_dir]:
        for subdir in ['images', 'masks', 'perfusion']:
            os.makedirs(os.path.join(split, subdir), exist_ok=True)
    
    # Get all samples
    all_samples = [f for f in os.listdir(os.path.join(source_dir, 'images')) if f.endswith('.npy')]
    print(f"Found {len(all_samples)} samples")
    
    # Group by patient
    patient_samples = {}
    for sample in all_samples:
        base_name = os.path.splitext(sample)[0]
        patient_id, _ = extract_patient_and_view(base_name)
        if patient_id not in patient_samples:
            patient_samples[patient_id] = []
        patient_samples[patient_id].append(base_name)
    
    unique_patients = sorted(list(patient_samples.keys()))
    print(f"Found {len(unique_patients)} independent patients")
    
    # Random split (7:3)
    np.random.shuffle(unique_patients)
    split_idx = int(len(unique_patients) * 0.7)
    train_patients = set(unique_patients[:split_idx])
    val_patients = set(unique_patients[split_idx:])
    
    print(f"Training set patients: {len(train_patients)}")
    print(f"Validation set patients: {len(val_patients)}")
    
    # Copy files
    train_count = 0
    val_count = 0
    
    for patient_id, samples in patient_samples.items():
        target_dir = train_dir if patient_id in train_patients else val_dir
        count = train_count if patient_id in train_patients else val_count
        
        for base_name in samples:
            # Copy image
            src_img = os.path.join(source_dir, 'images', f'{base_name}.npy')
            dst_img = os.path.join(target_dir, 'images', f'{base_name}.npy')
            shutil.copy(src_img, dst_img)
            
            # Copy mask
            src_mask = os.path.join(source_dir, 'masks', f'{base_name}.png')
            dst_mask = os.path.join(target_dir, 'masks', f'{base_name}.png')
            shutil.copy(src_mask, dst_mask)
            
            # Copy perfusion label
            src_perf = os.path.join(source_dir, 'perfusion', f'{base_name}.npy')
            dst_perf = os.path.join(target_dir, 'perfusion', f'{base_name}.npy')
            shutil.copy(src_perf, dst_perf)
            
            count += 1
        
        if patient_id in train_patients:
            train_count = count
        else:
            val_count = count
    
    print(f"\nSplit completed!")
    print(f"Training set samples: {train_count}")
    print(f"Validation set samples: {val_count}")
    print(f"Training set directory: {train_dir}")
    print(f"Validation set directory: {val_dir}")

if __name__ == "__main__":
    main()