import os
import shutil
import random
from tqdm import tqdm

# ---------------------- 配置（只改这里） ----------------------
# 你的分割训练集根目录
TRAIN_ROOT = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/segmentation_train"
# 目标验证集根目录（会自动创建）
VAL_ROOT = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/data/segmentation_val"
# 验证集比例（推荐0.2-0.3，这里用0.3）
VAL_RATIO = 0.3
# -------------------------------------------------------------

def split_single_view(view_dir):
    """对单个切面（A2C/A3C/A4C）进行训练/验证划分"""
    view_name = os.path.basename(view_dir)
    print(f"\n处理切面: {view_name}")
    
    # 源目录
    src_img_dir = os.path.join(view_dir, "images")
    src_mask_dir = os.path.join(view_dir, "masks")
    # 目标目录
    dst_img_dir = os.path.join(VAL_ROOT, view_name, "images")
    dst_mask_dir = os.path.join(VAL_ROOT, view_name, "masks")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    
    # 获取所有图像文件（自动匹配同名mask）
    img_files = [f for f in os.listdir(src_img_dir) 
                 if f.endswith((".jpg", ".jpeg", ".png"))]
    print(f"该切面总样本数: {len(img_files)}")
    
    # 随机划分
    random.shuffle(img_files)
    val_count = int(len(img_files) * VAL_RATIO)
    val_files = img_files[:val_count]
    print(f"划分到验证集的样本数: {val_count}")
    
    # 移动文件（注意是移动，不是复制，避免重复）
    moved_count = 0
    for img_file in tqdm(val_files, desc=f"移动 {view_name} 样本"):
        # 图像文件
        src_img = os.path.join(src_img_dir, img_file)
        dst_img = os.path.join(dst_img_dir, img_file)
        shutil.move(src_img, dst_img)
        
        # 对应mask文件（同名，仅后缀不同）
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.png"
        src_mask = os.path.join(src_mask_dir, mask_file)
        dst_mask = os.path.join(dst_mask_dir, mask_file)
        
        if os.path.exists(src_mask):
            shutil.move(src_mask, dst_mask)
            moved_count += 1
        else:
            print(f"警告: 找不到对应mask文件 {mask_file}，跳过")
    
    print(f"成功移动 {moved_count} 对图像+掩膜到验证集")

def main():
    print("=" * 60)
    print("分割训练/验证集一键划分工具")
    print("=" * 60)
    print(f"源目录: {TRAIN_ROOT}")
    print(f"目标目录: {VAL_ROOT}")
    print(f"验证集比例: {VAL_RATIO * 100:.0f}%")
    print("=" * 60)
    
    # 遍历三个切面
    for view in ["A2C", "A3C", "A4C"]:
        view_dir = os.path.join(TRAIN_ROOT, view)
        if os.path.exists(view_dir):
            split_single_view(view_dir)
        else:
            print(f"警告: 找不到切面目录 {view_dir}，跳过")
    
    print("\n" + "=" * 60)
    print("✅ 划分完成！现在你的目录结构是:")
    print(f"- 训练集: {TRAIN_ROOT}")
    print(f"- 验证集: {VAL_ROOT}")
    print("=" * 60)

if __name__ == "__main__":
    # 设置随机种子，保证每次划分结果一致（方便复现）
    random.seed(42)
    main()