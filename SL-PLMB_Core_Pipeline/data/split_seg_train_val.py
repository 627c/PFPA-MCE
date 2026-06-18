import os
import shutil
import random
from tqdm import tqdm
# ---------------------- Configuration (only edit here) ----------------------
# Root directory of your segmentation training set
TRAIN_ROOT = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/segmentation_train"
# Root directory of target validation set (will be created automatically)
VAL_ROOT = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/segmentation_val"
# Validation set ratio (recommended 0.2-0.3, using 0.3 here)
VAL_RATIO = 0.3
# -----------------------------------------------------------------------------
def split_single_view(view_dir):
    """Split training/validation set for a single view (A2C/A3C/A4C)"""
    view_name = os.path.basename(view_dir)
    print(f"\nProcessing view: {view_name}")
    
    # Source directories
    src_img_dir = os.path.join(view_dir, "images")
    src_mask_dir = os.path.join(view_dir, "masks")
    # Target directories
    dst_img_dir = os.path.join(VAL_ROOT, view_name, "images")
    dst_mask_dir = os.path.join(VAL_ROOT, view_name, "masks")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    
    # Get all image files (automatically match masks with the same name)
    img_files = [f for f in os.listdir(src_img_dir) 
                 if f.endswith((".jpg", ".jpeg", ".png"))]
    print(f"Total samples in this view: {len(img_files)}")
    
    # Random split
    random.shuffle(img_files)
    val_count = int(len(img_files) * VAL_RATIO)
    val_files = img_files[:val_count]
    print(f"Number of samples assigned to validation set: {val_count}")
    
    # Move files (note: move, not copy, to avoid duplicates)
    moved_count = 0
    for img_file in tqdm(val_files, desc=f"Moving {view_name} samples"):
        # Image file
        src_img = os.path.join(src_img_dir, img_file)
        dst_img = os.path.join(dst_img_dir, img_file)
        shutil.move(src_img, dst_img)
        
        # Corresponding mask file (same name, different extension only)
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.png"
        src_mask = os.path.join(src_mask_dir, mask_file)
        dst_mask = os.path.join(dst_mask_dir, mask_file)
        
        if os.path.exists(src_mask):
            shutil.move(src_mask, dst_mask)
            moved_count += 1
        else:
            print(f"Warning: corresponding mask file {mask_file} not found, skipped")
    
    print(f"Successfully moved {moved_count} pairs of image + mask to validation set")
def main():
    print("=" * 60)
    print("One-click split tool for segmentation training/validation set")
    print("=" * 60)
    print(f"Source directory: {TRAIN_ROOT}")
    print(f"Target directory: {VAL_ROOT}")
    print(f"Validation set ratio: {VAL_RATIO * 100:.0f}%")
    print("=" * 60)
    
    # Iterate over three views
    for view in ["A2C", "A3C", "A4C"]:
        view_dir = os.path.join(TRAIN_ROOT, view)
        if os.path.exists(view_dir):
            split_single_view(view_dir)
        else:
            print(f"Warning: view directory {view_dir} not found, skipped")
    
    print("\n" + "=" * 60)
    print("✅ Split completed! Your directory structure is now:")
    print(f"- Training set: {TRAIN_ROOT}")
    print(f"- Validation set: {VAL_ROOT}")
    print("=" * 60)
if __name__ == "__main__":
    # Set random seed to ensure consistent split results every time (for reproducibility)
    random.seed(42)
    main()