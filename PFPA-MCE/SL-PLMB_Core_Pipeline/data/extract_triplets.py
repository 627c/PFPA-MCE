"""
extract_triplets.py - MCE 视频生理学三元组帧自动提取与导出工具
功能：输入一个 .avi 视频，自动识别 Flash，精准定位并导出 Destroy、Mid、Plateau 三张高清金标准灰度图片。
"""
import os
import cv2
import numpy as np

def auto_extract_mce_triplets(video_path, output_dir):
    """
    基于微泡充盈动力学自动定位并保存三元组灰度图片
    """
    if not os.path.exists(video_path):
        print(f"❌ 错误：找不到输入的视频文件: {video_path}")
        return False

    # 1. 初始化视频流读取器
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps) or fps > 100:
        fps = 30.0  # 兜底标准超声帧率
    
    raw_frames = []
    mean_intensities = []
    
    print(f"⏳ 正在读取视频并计算全图声学强度流线...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 转为灰度图用于计算全图平均强度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        raw_frames.append(frame)  # 保留原始彩色帧用于后续转换
        mean_intensities.append(np.mean(gray))
        
    cap.release()
    
    total_frames = len(raw_frames)
    mean_intensities = np.array(mean_intensities)
    
    if total_frames < 15:
        print(f"❌ 错误：视频总帧数过短 ({total_frames} 帧)，无法进行生理动力学分析。")
        return False

    print(f"✅ 视频读取成功，共 {total_frames} 帧 | 检测到视频帧率 (FPS): {fps:.2f}")

    # =========================================================================
    # 2. 核心动力学算法定位：三大黄金生理标志点
    # =========================================================================
    
    # 📍 点 1: 寻找绝对的 Flash 闪光最高点
    # 临床上由于高能脉冲破泡，仪器屏幕会产生一个瞬间的全白极高亮物理阻断
    flash_spike_idx = np.argmax(mean_intensities)
    
    # 📍 点 2: 寻找 Destroy 帧（真实的破泡终点）
    # Flash 闪光本身亮，但闪光后微泡全部灰飞烟灭的瞬间，才是心肌最暗的物理谷底
    # 我们在 Flash 随后的 2 秒钟（2 * fps 帧）内寻找强度的绝对最低点
    search_dark_limit = min(total_frames, flash_spike_idx + int(fps * 2))
    search_dark_zone = mean_intensities[flash_spike_idx : search_dark_limit]
    
    if len(search_dark_zone) == 0:
        destroy_idx = flash_spike_idx
    else:
        destroy_idx = flash_spike_idx + np.argmin(search_dark_zone)
        
    # 📍 点 3: 寻找 Plateau 帧（造影剂再灌注饱和平台期）
    # 微泡补充完全，曲线进入长期的稳定高原区。我们使用 5 帧滑动平均平滑信号，寻找后期的充盈峰值
    post_destroy_zone = mean_intensities[destroy_idx + 1:]
    if len(post_destroy_zone) < 5:
        plateau_idx = total_frames - 1
    else:
        smoothed_post_zone = np.convolve(post_destroy_zone, np.ones(5)/5, mode='same')
        plateau_idx = destroy_idx + 1 + np.argmax(smoothed_post_zone)
        
    # 安全边界锁：防止平台帧因呼吸漂移意外跑到序列最末尾干扰视线
    plateau_idx = min(plateau_idx, total_frames - 1)
    if plateau_idx <= destroy_idx:
        plateau_idx = total_frames - 1

    # 📍 点 4: 寻找 Mid 帧（50% 灌注半山腰恢复点）
    # 这一帧的亮度恰好处于全黑破坏谷底与饱和高原的正中间，代表了血流重新充盈的速度曲率
    y_destroy = mean_intensities[destroy_idx]
    y_plateau = mean_intensities[plateau_idx]
    target_mid_intensity = y_destroy + (y_plateau - y_destroy) * 0.5
    
    # 在 Destroy 和 Plateau 之间搜索最接近 50% 恢复亮度的帧
    search_mid_zone = mean_intensities[destroy_idx : plateau_idx]
    if len(search_mid_zone) == 0:
        mid_idx = destroy_idx + (plateau_idx - destroy_idx) // 2
    else:
        mid_idx = destroy_idx + np.argmin(np.abs(search_mid_zone - target_mid_intensity))

    # =========================================================================
    # 3. 图像物理导出与定位报告（已修改为灰度图保存）
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 构造标准顶刊格式命名物理存盘
    destroy_img_path = os.path.join(output_dir, f"{video_base_name}_triplet_1_destroy.png")
    mid_img_path = os.path.join(output_dir, f"{video_base_name}_triplet_2_mid.png")
    plateau_img_path = os.path.join(output_dir, f"{video_base_name}_triplet_3_plateau.png")
    
    # 仅修改此处：转换为灰度图后保存
    cv2.imwrite(destroy_img_path, cv2.cvtColor(raw_frames[destroy_idx], cv2.COLOR_BGR2GRAY))
    cv2.imwrite(mid_img_path, cv2.cvtColor(raw_frames[mid_idx], cv2.COLOR_BGR2GRAY))
    cv2.imwrite(plateau_img_path, cv2.cvtColor(raw_frames[plateau_idx], cv2.COLOR_BGR2GRAY))
    
    print("\n" + "="*60)
    print("🏆 MCE 生理学时间轴三元组帧自动提取报告")
    print("="*60)
    print(f"🎬 正在分析病例视频: {video_base_name}.avi")
    print(f"💥 检测到 Flash 闪光突变点: 第 {flash_spike_idx} 帧")
    print(f"📸 [1/3] 导出 Destroy 帧 (完全全黑): 第 {destroy_idx} 帧 (时间: {destroy_idx/fps:.2f}s)")
    print(f"📸 [2/3] 导出 Mid 帧     (50%充盈):  第 {mid_idx} 帧 (时间: {mid_idx/fps:.2f}s)")
    print(f"📸 [3/3] 导出 Plateau 帧 (完全充满): 第 {plateau_idx} 帧 (时间: {plateau_idx/fps:.2f}s)")
    print(f"📂 高清金标准灰度插图已成功输出至目录: {output_dir}")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    # 💡 只需要在这里配置你的本地路径即可直接运行！
    video_file_path = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/raw_data/videos/P005_A4C.avi"
    output_directory = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results/extracted_triplets/"
    
    auto_extract_mce_triplets(video_file_path, output_directory)