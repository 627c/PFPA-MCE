"""
interactive_labeler_nogui.py - 无GUI服务器专用交互式标注器
✅ 完全不需要图形界面，纯终端操作
✅ 自动生成关键帧预览图
✅ 支持实时调整预测阈值
✅ 保留所有原代码的核心功能
✅ 完美适配远程SSH服务器环境

操作流程：
1. 脚本自动生成关键帧预览图
2. 你用scp下载预览图到本地查看
3. 在终端输入最佳帧的编号
4. 脚本生成该帧的掩膜图
5. 输入y保存，n重新选择，t调整阈值
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append("/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE")
from models.net_seg_only import AttentionUNet
from config import BASE_CONFIG

# ================= 配置区 =================
IMG_SIZE = (512, 512)
ROOT_DIR = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE"
CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints/best_seg_model.pth')
VIDEO_DIR = os.path.join(ROOT_DIR, 'raw_data/videos/')
MASK_OUTPUT_DIR = os.path.join(ROOT_DIR, 'raw_data/pseudo_masks/')
PREVIEW_DIR = os.path.join(ROOT_DIR, 'raw_data/preview_frames/')
# ==========================================

def keep_image_size_open(img):
    """预处理函数（保持不变）"""
    img = Image.fromarray(img).convert('L')
    ratio = min(IMG_SIZE[0] / img.size[0], IMG_SIZE[1] / img.size[1])
    new_w, new_h = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    new_img = Image.new('L', IMG_SIZE, 0)
    new_img.paste(img, ((IMG_SIZE[0] - new_w) // 2, (IMG_SIZE[1] - new_h) // 2))
    return np.array(new_img)

def predict_mask(frame, model, device, threshold=0.3):
    """调用 AI 预测掩膜（保持不变）"""
    img_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        # TTA 增强预测
        pred1 = model(img_t)
        pred2 = model(torch.flip(img_t, dims=[3]))
        pred_mask = (pred1 + torch.flip(pred2, dims=[3])) / 2.0
        pred_mask = torch.sigmoid(pred_mask)[0,0].cpu().numpy()
    
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return binary_mask

def generate_preview_grid(frames, video_name, num_frames=10):
    """生成关键帧预览网格图"""
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    
    # 均匀抽取10个关键帧
    sample_indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    
    # 创建2行5列的网格
    grid = []
    for i in range(2):
        row = []
        for j in range(5):
            idx = sample_indices[i*5 + j]
            frame = (frames[idx] * 255).astype(np.uint8)
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # 在帧上添加编号
            cv2.putText(frame_color, f"{idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            row.append(frame_color)
        grid.append(np.hstack(row))
    
    preview_img = np.vstack(grid)
    preview_path = os.path.join(PREVIEW_DIR, f"{os.path.splitext(video_name)[0]}_preview.png")
    cv2.imwrite(preview_path, preview_img)
    
    return preview_path, sample_indices

def generate_mask_preview(frame, mask, video_name, frame_idx, threshold):
    """生成掩膜预览图"""
    frame_vis = (frame * 255).astype(np.uint8)
    frame_color = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)
    
    mask_vis = frame_color.copy()
    mask_vis[mask > 127] = [0, 255, 0]
    overlay = cv2.addWeighted(frame_color, 0.7, mask_vis, 0.3, 0)
    
    cv2.putText(overlay, f"Frame: {frame_idx} | Threshold: {threshold:.2f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    mask_preview_path = os.path.join(PREVIEW_DIR, f"{os.path.splitext(video_name)[0]}_mask_{frame_idx}.png")
    cv2.imwrite(mask_preview_path, overlay)
    
    return mask_preview_path

def main():
    print("="*60)
    print("🎬 无GUI交互式掩膜生成器 启动中...")
    print("="*60)
    
    # 1. 检查路径
    print(f"\n📂 检查路径配置:")
    print(f"   模型路径: {CHECKPOINT_PATH}")
    print(f"   视频目录: {VIDEO_DIR}")
    print(f"   输出目录: {MASK_OUTPUT_DIR}")
    print(f"   预览目录: {PREVIEW_DIR}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ 错误：找不到模型文件 {CHECKPOINT_PATH}")
        return
    
    if not os.path.exists(VIDEO_DIR):
        print(f"\n❌ 错误：找不到视频目录 {VIDEO_DIR}")
        return
    
    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    
    # 2. 加载模型
    print(f"\n🧠 加载分割模型...")
    try:
        device = BASE_CONFIG['device']
        model = AttentionUNet().to(device)
        
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            best_dsc = checkpoint.get('best_dsc', 0.0)
            print(f"✅ 模型加载成功！最佳验证DSC: {best_dsc:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ 旧版模型加载成功！")
        
        model.eval()
    except Exception as e:
        print(f"\n❌ 模型加载失败: {str(e)}")
        return
    
    # 3. 获取视频列表
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi'))]
    
    if len(videos) == 0:
        print(f"\n❌ 错误：在 {VIDEO_DIR} 中没有找到任何视频文件")
        return
    
    print(f"\n📹 找到 {len(videos)} 个视频文件")
    
    # 4. 处理每个视频
    for i, vid_name in enumerate(videos):
        mask_path = os.path.join(MASK_OUTPUT_DIR, os.path.splitext(vid_name)[0] + '.png')
        
        if os.path.exists(mask_path):
            print(f"\n⏭️  [{i+1}/{len(videos)}] 跳过已处理: {vid_name}")
            continue
        
        print(f"\n🎥  [{i+1}/{len(videos)}] 正在处理: {vid_name}")
        
        # 读取视频
        try:
            cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, vid_name))
            frames = []
            
            while cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                gray = keep_image_size_open(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                frames.append(gray / 255.0)
            
            cap.release()
            
            if len(frames) == 0:
                print(f"   ⚠️  无法读取视频或视频为空，跳过")
                continue
            
            print(f"   ✅ 成功读取视频，共 {len(frames)} 帧")
        except Exception as e:
            print(f"   ❌ 视频读取失败: {str(e)}")
            continue
        
        # 生成关键帧预览图
        preview_path, sample_indices = generate_preview_grid(frames, vid_name)
        print(f"\n   🖼️  关键帧预览图已生成: {preview_path}")
        print(f"   📥  请在本地终端执行以下命令下载预览图:")
        print(f"      scp stu1@your_server_ip:{preview_path} ./")
        print(f"   👀  查看预览图后，输入最佳帧的编号（0-{len(frames)-1}）")
        
        current_threshold = 0.3
        
        while True:
            # 获取用户输入的帧编号
            while True:
                user_input = input("\n   请输入最佳帧编号 (输入q退出): ")
                
                if user_input.lower() == 'q':
                    print("\n👋 程序退出")
                    return
                
                try:
                    frame_idx = int(user_input)
                    if 0 <= frame_idx < len(frames):
                        break
                    else:
                        print(f"   ❌ 帧编号必须在0到{len(frames)-1}之间")
                except ValueError:
                    print("   ❌ 请输入有效的数字")
            
            # 预测掩膜
            print(f"   🧠 AI 正在预测第 {frame_idx} 帧的掩膜...")
            mask = predict_mask(frames[frame_idx], model, device, current_threshold)
            
            # 生成掩膜预览图
            mask_preview_path = generate_mask_preview(frames[frame_idx], mask, vid_name, frame_idx, current_threshold)
            print(f"   ✅ 掩膜预览图已生成: {mask_preview_path}")
            print(f"   📥  请下载并查看掩膜预览图")
            
            # 获取用户反馈
            while True:
                feedback = input("\n   掩膜是否满意？(y=保存, n=重新选帧, t=调整阈值): ").lower()
                
                if feedback == 'y':
                    # 保存掩膜
                    try:
                        cv2.imwrite(mask_path, mask)
                        print(f"   ✅ 掩膜已保存: {mask_path}")
                        break
                    except Exception as e:
                        print(f"   ❌ 保存失败: {str(e)}")
                elif feedback == 'n':
                    print("   🔄 重新选择帧")
                    break
                elif feedback == 't':
                    # 调整阈值
                    while True:
                        threshold_input = input(f"   请输入新的阈值 (当前: {current_threshold:.2f}, 范围0.1-0.5): ")
                        try:
                            new_threshold = float(threshold_input)
                            if 0.1 <= new_threshold <= 0.5:
                                current_threshold = new_threshold
                                print(f"   ✅ 阈值已调整为: {current_threshold:.2f}")
                                # 重新预测掩膜
                                mask = predict_mask(frames[frame_idx], model, device, current_threshold)
                                mask_preview_path = generate_mask_preview(frames[frame_idx], mask, vid_name, frame_idx, current_threshold)
                                print(f"   ✅ 新的掩膜预览图已生成: {mask_preview_path}")
                                break
                            else:
                                print("   ❌ 阈值必须在0.1到0.5之间")
                        except ValueError:
                            print("   ❌ 请输入有效的数字")
                else:
                    print("   ❌ 请输入 y, n 或 t")
            
            if feedback == 'y':
                break
    
    print("\n" + "="*60)
    print("🎉 所有视频处理完成！")
    print(f"📊 结果已保存到: {MASK_OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()