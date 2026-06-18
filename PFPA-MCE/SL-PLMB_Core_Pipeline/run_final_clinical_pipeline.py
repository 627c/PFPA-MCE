"""
main.py - 临床可解释性增强版（新增节段级结果保存 + 消融实验全局开关）
"""
import os
import torch
import numpy as np
import json
from config import BASE_CONFIG, PATHS
from core.fusion import GlobalPLMBManager
from core.tac import extract_purified_tac
from core.anatomy import get_aha_view_mapping, generate_aha17_mask
from models.net import AttentionUNet
from data.dataset import build_dataloader
from utils.helpers import extract_patient_and_view, extract_temporal_triplet
from evaluation.evaluate import evaluate_full_pipeline

# ✅ 新增：消融实验全局控制开关（会被run_ablation_studies.py动态修改）
USE_PLMB = True

# ✅ 终极修复：sorted()保证数据集划分绝对可复现，彻底杜绝数据泄露
def split_train_test(video_list, seed=42):
    np.random.seed(seed)
    # 必须先排序再打乱！保证每次运行初始顺序完全一致
    unique_patients = sorted(list({v['patient_id'] for v in video_list}))
    np.random.shuffle(unique_patients)
    split = int(len(unique_patients) * 0.7)
    train_patients = set(unique_patients[:split])
    test_patients = set(unique_patients[split:])
    train_videos = [v for v in video_list if v['patient_id'] in train_patients]
    test_videos = [v for v in video_list if v['patient_id'] in test_patients]
    return train_videos, test_videos, list(test_patients)

def build_video_list():
    videos = []
    for f in os.listdir(PATHS['video_input']):
        if f.endswith(('.mp4','.avi')):
            pid, view = extract_patient_and_view(f)
            if pid and view:
                videos.append({'path': os.path.join(PATHS['video_input'], f),
                               'patient_id': pid, 'view': view})
    return videos

def main():
    device = BASE_CONFIG['device']
    os.makedirs(PATHS['results_root'], exist_ok=True)
    video_list = build_video_list()
    
    train_videos, test_videos, test_patients = split_train_test(video_list)
    
    plmb = GlobalPLMBManager()
    plmb.set_training_mode(True)
    
    # 训练阶段（仅当启用PLMB时需要入库训练）
    if USE_PLMB:
        for v in train_videos:
            pid = v['patient_id']
            view = v['view']
            plmb.init_patient(pid)
            tac_dir = os.path.join(PATHS['tac_output'], pid, view)
            extract_purified_tac(v['path'],
                                  os.path.join(PATHS['train_images'], pid, view),
                                  os.path.join(PATHS['train_masks'], pid, view),
                                  PATHS['model_save'], tac_dir, plmb=plmb)
    
    plmb.set_training_mode(False)
    model = AttentionUNet().to(device)
    model.load_state_dict(torch.load(PATHS['model_save'], map_location=device))
    model.eval()
    
    results = {}
    fusion_failure_records = {}
    
    for patient_id in test_patients:
        pat_videos = [v for v in test_videos if v['patient_id'] == patient_id]
        plmb.init_patient(patient_id)
        
        # Pass1：入库（仅当启用PLMB时执行）
        if USE_PLMB:
            for video in pat_videos:
                view = video['view']
                pid = video['patient_id']
                tac_dir = os.path.join(PATHS['tac_output'], pid, view)
                tac_data = np.load(os.path.join(tac_dir, 'full_tac.npy'), allow_pickle=True).item()
                purified_video = tac_data['purified_video']
                global_mask = tac_data['global_mask']
                
                triplet, _ = extract_temporal_triplet(purified_video)
                img_t = torch.from_numpy(triplet).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    _, base_feats = model.forward_features(img_t)
                
                aha_mask = generate_aha17_mask(global_mask, view)
                
                feats_flat = base_feats.squeeze(0).view(64, -1).t()
                for seg_id in range(1, 18):
                    seg_mask_flat = (aha_mask == seg_id).flatten()
                    seg_idx = torch.where(torch.from_numpy(seg_mask_flat).to(device))[0]
                    if len(seg_idx) == 0:
                        continue
                    feats_seg = feats_flat[seg_idx]
                    plmb.update(pid, view, seg_id, feats_seg)
        
        # Pass2：跨视图融合预测（或直接预测）
        for video in pat_videos:
            view = video['view']
            pid = video['patient_id']
            tac_dir = os.path.join(PATHS['tac_output'], pid, view)
            tac_data = np.load(os.path.join(tac_dir, 'full_tac.npy'), allow_pickle=True).item()
            purified_video = tac_data['purified_video']
            global_mask = tac_data['global_mask']
            
            triplet, _ = extract_temporal_triplet(purified_video)
            img_t = torch.from_numpy(triplet).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                _, base_feats = model.forward_features(img_t)
            
            aha_mask = generate_aha17_mask(global_mask, view)
            
            feats_flat = base_feats.squeeze(0).view(64, -1).t()
            
            # ✅ 核心修改：根据USE_PLMB开关决定是否进行跨视图融合
            if USE_PLMB:
                fused_feats = plmb.retrieve_and_fuse(pid, view, feats_flat, aha_mask)
                fused_map = fused_feats.t().view(1, 64, *global_mask.shape)
                pred_perf = model.forward_regression(fused_map)
            else:
                # 无PLMB融合：直接使用原始特征回归
                pred_perf = model.forward_regression(base_feats)
            
            A_map = pred_perf[0, 0].cpu().numpy()
            beta_map = pred_perf[0, 1].cpu().numpy()
            mbf_map = A_map * beta_map
            
            # ✅ 临床增强：同时保存全局平均和AHA17节段级结果
            # 1. 全局平均（保留原有逻辑用于对比）
            myo = global_mask > 0.5
            if myo.sum() > 0:
                global_A = float(np.mean(A_map[myo]))
                global_beta = float(np.mean(beta_map[myo]))
                global_mbf = float(np.mean(mbf_map[myo]))
            else:
                global_A = global_beta = global_mbf = 0.0
            
            # 2. AHA17节段级结果（论文核心数据）
            segment_results = {}
            for seg_id in range(1, 18):
                seg_mask = (aha_mask == seg_id)
                if seg_mask.sum() > 0:  # 仅保存存在的节段
                    seg_A = float(np.mean(A_map[seg_mask]))
                    seg_beta = float(np.mean(beta_map[seg_mask]))
                    seg_mbf = float(np.mean(mbf_map[seg_mask]))
                    segment_results[f'seg_{seg_id}'] = {
                        'A': seg_A,
                        'beta': seg_beta,
                        'mbf': seg_mbf
                    }
            
            # 3. 统一保存结果
            results.setdefault(pid, {})[view] = {
                'global': {
                    'A': global_A,
                    'beta': global_beta,
                    'mbf': global_mbf
                },
                'segments': segment_results
            }
            
            # ✅ 安全初始化嵌套字典，彻底杜绝KeyError
            fusion_failure_records.setdefault(pid, {})[view] = plmb.get_fusion_failures(pid, view) if USE_PLMB else []
        
        plmb.clear_memory(patient_id)
    
    # ✅ 修复：根据USE_PLMB开关保存不同的结果文件，避免消融实验互相覆盖
    if USE_PLMB:
        result_filename = 'final_perfusion_results.json'
        failure_filename = 'fusion_failure_records.json'
    else:
        result_filename = 'no_plmb_results.json'
        failure_filename = 'no_plmb_failure_records.json'
    
    # 保存结果和融合失败记录
    with open(os.path.join(PATHS['results_root'], result_filename), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(PATHS['results_root'], failure_filename), 'w', encoding='utf-8') as f:
        json.dump(fusion_failure_records, f, indent=2)
    
    evaluate_full_pipeline(test_patients=test_patients)

if __name__ == "__main__":
    main()