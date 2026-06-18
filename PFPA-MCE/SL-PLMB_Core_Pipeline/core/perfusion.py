"""
perfusion.py - 灌注参数拟合
实现像素级并行拟合，生成A、β和MBF灌注图
"""
import sys
sys.path.append("/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE")
import numpy as np
from scipy.optimize import curve_fit
import warnings
from joblib import Parallel, delayed
from config import BASE_CONFIG, SIGNAL_PARAMS

def exponential_model(t, A, beta, x0, y0):
    """
    经典微泡补充模型 (Replenishment Kinetics Model):
    y(t) = A * (1 - e^(-beta * t)) + y0
    加入时间偏移x0确保曲线通过造影剂破坏瞬间
    """
    return A * ((1 - np.exp(-beta * t)) - (1 - np.exp(-beta * x0))) + y0

def fit_single_tac(tac_sequence, fps=30):
    """
    拟合单条TAC曲线
    Returns: A, beta, fitted_curve
    """
    if len(tac_sequence) < 10:
        return 0.0, 0.0, np.zeros_like(tac_sequence)
    
    time_axis = np.arange(len(tac_sequence)) / fps
    x0 = 0.1
    y0 = np.min(tac_sequence)
    initial_guess = [np.max(tac_sequence) - y0, 1.0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            params, _ = curve_fit(
                lambda t, A, beta: exponential_model(t, A, beta, x0, y0),
                time_axis,
                tac_sequence,
                p0=initial_guess,
                bounds=([0, 0], [np.inf, 10.0]),
                maxfev=2000  # ✅ 修复：显式限制最大迭代次数，防止死锁
            )
            A, beta = params
        except Exception:
            A, beta = 0.0, 0.0
    
    fitted_curve = exponential_model(time_axis, A, beta, x0, y0)
    return A, beta, fitted_curve

def _fit_pixel_wrapper(args):
    """并行计算包装函数"""
    y, x, pixel_series, fps = args
    if np.max(pixel_series) < 1e-3:  # 背景像素跳过
        return y, x, 0.0, 0.0
    
    # 时频滤波
    from core.tac import time_frequency_filter
    purified = time_frequency_filter(pixel_series, fps=fps)
    
    # 拟合参数
    A, beta, _ = fit_single_tac(purified, fps=fps)
    return y, x, A, beta

def generate_perfusion_maps(video_frames, mask, fps=30, n_jobs=-1):
    """
    生成像素级灌注参数图
    Args:
        video_frames: (T, H, W) 视频帧数组
        mask: (H, W) 心肌掩膜
        n_jobs: 并行进程数，-1使用所有CPU核心
    Returns:
        A_map: 血容量图 (H, W)
        beta_map: 血流速度图 (H, W)
        mbf_map: 心肌血流量图 (A*beta) (H, W)
    """
    T, H, W = video_frames.shape
    A_map = np.zeros((H, W), dtype=np.float32)
    beta_map = np.zeros((H, W), dtype=np.float32)
    
    # 提取所有需要计算的心肌像素
    ys, xs = np.where(mask > 0)
    
    # 构建并行任务
    tasks = [(y, x, video_frames[:, y, x], fps) for y, x in zip(ys, xs)]
    
    # 多进程并行计算
    results = Parallel(n_jobs=n_jobs, batch_size='auto', timeout=300)(  # ✅ 新增：5分钟超时保护
        delayed(_fit_pixel_wrapper)(task) for task in tasks
    )
    
    # 填回结果
    for y, x, A, beta in results:
        A_map[y, x] = A
        beta_map[y, x] = beta
    
    # 计算心肌血流量
    mbf_map = A_map * beta_map
    
    return A_map, beta_map, mbf_map

def fit_tac_to_params(tac_sequence_path, output_fig_dir, fps=30):
    """
    拟合TAC序列为灌注参数并绘制结果
    Args:
        tac_sequence_path: TAC序列文件路径
        output_fig_dir: 输出目录
        fps: 视频帧率
    Returns:
        A: 血容量参数
        beta: 血流速度参数
    """
    import os
    import matplotlib.pyplot as plt
    
    # 加载TAC序列
    try:
        tac_data = np.load(tac_sequence_path, allow_pickle=True)
        
        if isinstance(tac_data, np.ndarray) and tac_data.dtype == object:
            tac_data = tac_data.item()
        
        if isinstance(tac_data, dict):
            if 'enhanced_tac' in tac_data:
                tac_sequence = tac_data['enhanced_tac']
            elif 'primary_tac_sequence' in tac_data:
                tac_sequence = tac_data['primary_tac_sequence']
            else:
                tac_sequence = tac_data['raw_tac']
        else:
            tac_sequence = tac_data
        
    except Exception as e:
        print(f"加载TAC序列失败: {e}")
        return 0.0, 0.0
    
    # 拟合参数
    A, beta, fitted_curve = fit_single_tac(tac_sequence, fps)
    
    # 绘制拟合结果
    os.makedirs(output_fig_dir, exist_ok=True)
    time_axis = np.arange(len(tac_sequence)) / fps
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, tac_sequence, label='原始TAC', color='blue', linewidth=2)
    plt.plot(time_axis, fitted_curve, label=f'拟合曲线 (A={A:.4f}, β={beta:.4f})', 
             color='red', linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Mean Intensity', fontsize=12)
    plt.title('TAC曲线与指数拟合结果', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_fig_dir, 'tac_fitting_result.png'), dpi=300)
    plt.close()
    
    # 保存参数
    fit_params = {
        'A': A,
        'beta': beta,
        'perfusion_index': A * beta,
        'time_axis': time_axis,
        'tac_sequence': tac_sequence,
        'fitted_curve': fitted_curve
    }
    
    np.save(os.path.join(output_fig_dir, 'fit_params.npy'), fit_params)
    
    # 保存文本格式
    with open(os.path.join(output_fig_dir, 'perfusion_parameters.txt'), 'w') as f:
        f.write(f"血容量 (A): {A:.4f}\n")
        f.write(f"血流速度 (β): {beta:.4f}\n")
        f.write(f"灌注指数 (A×β): {A*beta:.4f}\n")
    
    print(f"拟合完成: A={A:.4f}, β={beta:.4f}")
    return A, beta