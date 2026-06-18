"""
evaluate.py - 动态战区终极评估版（剥离误导性伪标签分割指标，纯粹评估灌注性能）
专用于评估视频生成的三元组测试集的 MSE 误差。
"""
import os
import torch
import numpy as np
from config import BASE_CONFIG, PATHS
from data.dataset import MCEDatasetNpy  
from utils.helpers import extract_patient_and_view
from models.net import AttentionUNet

def evaluate_full_pipeline(test_patients, source_dir='./data/training/'):
    """
    Args:
        test_patients: list/set, main.py传进来的测试集患者ID列表
        source_dir: 总预处理数据存放地
    """
    device = BASE_CONFIG['device']
    model = AttentionUNet().to(device)
    
    if not os.path.exists(PATHS['model_save']):
        print(f"❌ 找不到模型权重文件: {PATHS['model_save']}")
        return
        
    model.load_state_dict(torch.load(PATHS['model_save'], map_location=device))
    model.eval()

    # 1. 初始化总数据集（读取所有预处理好的三元组数据）
    full_dataset = MCEDatasetNpy(
        image_dir=os.path.join(source_dir, 'images'),
        mask_dir=os.path.join(source_dir, 'masks'),
        perfusion_dir=os.path.join(source_dir, 'perfusion'),
        use_aha=False  # 纯评估MSE，不需要AHA切分
    )
    
    # 2. 动态过滤测试样本，杜绝数据泄露
    test_filenames = []
    for f in full_dataset.filenames:
        pid, _ = extract_patient_and_view(f)
        if pid in test_patients:
            test_filenames.append(f)
    
    full_dataset.filenames = test_filenames
    
    if len(full_dataset) == 0:
        print("⚠️ 评估集为空，请检查 test_patients 列表是否正确传递！")
        return

    loader = torch.utils.data.DataLoader(full_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    all_mse_A, all_mse_beta = [], []
    
    print("⏳ 正在计算灌注参数回归误差 (MSE)...")
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].to(device)
            mask = batch['mask'].to(device) # 这里读取的是伪标签掩膜，仅用于定位心肌
            gt_perf = batch.get('perfusion')
            
            # 推理
            _, pred_perf, _ = model(img)
            
            # 灌注误差计算（MSE）- 绝对核心！
            if gt_perf is not None:
                gt_perf = gt_perf.to(device)
                # 将掩膜作为布尔索引，排除背景零值的污染
                myo = (mask > 0.5).squeeze(1)  # [B, H, W]
                
                if myo.sum() > 0:
                    err_A = ((pred_perf[:, 0][myo] - gt_perf[:, 0][myo])) ** 2
                    err_beta = ((pred_perf[:, 1][myo] - gt_perf[:, 1][myo])) ** 2
                    
                    all_mse_A.append(err_A.mean().item())
                    all_mse_beta.append(err_beta.mean().item())

    # 3. 打印纯粹的定量评估结果
    print("\n" + "="*50)
    print("🚀 动态战区 (视频三元组) 灌注定量评估结果")
    print("="*50)
    print(f"测试集大小: {len(full_dataset)} 个三元组样本")
    
    if all_mse_A:
        print(f"► 血容量 (A) MSE:     {np.mean(all_mse_A):.4f}")
        print(f"► 血流速度 (β) MSE:  {np.mean(all_mse_beta):.4f}")
        print("💡 注：此处不计算 DSC/IoU，因掩膜为模型自生成伪标签，无统计学意义。")
    else:
        print("⚠️ 警告：数据集中未找到真实的 Perfusion Ground Truth！")
    print("="*50)

if __name__ == "__main__":
    # 此处为独立调试入口，实际运行应通过 main.py 调用并传入正确的 test_patients
    print("请在 main.py 的流程中调用 evaluate_full_pipeline()")