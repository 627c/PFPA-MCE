"""
convert_masks.py - View-wise Recursive Version
Automatically converts color masks in all subfolders to single-channel grayscale masks
"""
import os
import cv2
import numpy as np
from glob import glob

TRAIN_MASK_ROOT = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/segmentation_train'
VAL_MASK_ROOT = '/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/data/segmentation_val'
def convert_single_directory(mask_dir):
    """Convert all masks in a single directory"""
    # Recursively find all .png files
    mask_files = glob(os.path.join(mask_dir, '**', '*.png'), recursive=True)
    print(f"Found {len(mask_files)} mask files in {mask_dir}")
    
    success_count = 0
    for path in mask_files:
        try:
            # 1. Read in color mode (required!)
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Failed to read {path}")
                continue
            
            # 2. Extract red channel (OpenCV uses BGR format, red is in channel 2)
            red_channel = img[:, :, 2]
            
            # 3. Binarization: convert red regions to pure white (255), others to pure black (0)
            _, binary_mask = cv2.threshold(red_channel, 100, 255, cv2.THRESH_BINARY)
            
            # 4. Overwrite and save as single-channel grayscale image
            cv2.imwrite(path, binary_mask)
            success_count += 1
            
        except Exception as e:
            print(f"Error: Failed to convert {path}: {str(e)}")
    
    print(f"Successfully converted {success_count} masks\n")
    return success_count
if __name__ == "__main__":
    print("=" * 60)
    print("Starting training set mask conversion...")
    print("=" * 60)
    train_success = convert_single_directory(TRAIN_MASK_ROOT)
    
    print("=" * 60)
    print("Starting validation set mask conversion...")
    print("=" * 60)
    val_success = convert_single_directory(VAL_MASK_ROOT)
    
    print("=" * 60)
    print(f"All conversions completed! Total converted masks: {train_success + val_success}")
    print("=" * 60)