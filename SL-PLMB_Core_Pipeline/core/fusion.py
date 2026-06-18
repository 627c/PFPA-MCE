"""
fusion.py 
"""
import os
import json
import torch
import torch.nn.functional as F
from config import BASE_CONFIG, PATHS
from core.anatomy import get_aha_view_mapping

class GlobalPLMBManager:
    def __init__(self, save_dir=None, feature_dim=64, capacity=100):
        self.save_dir = save_dir or PATHS['plmb_dir']
        self.feature_dim = feature_dim
        self.capacity = capacity
        self.fusion_alpha = 0.2
        self.memory = {}
        self.seg_counts = {}
        self.fusion_failures = {}
        self.training_mode = True
        os.makedirs(self.save_dir, exist_ok=True)
        self._load_state()

    def _load_state(self):
        try:
            self.memory = torch.load(os.path.join(self.save_dir, 'plmb.pt'), map_location='cpu')
            with open(os.path.join(self.save_dir, 'meta.json'), 'r') as f:
                raw_counts = json.load(f)
            self.seg_counts = {}
            for pid, views in raw_counts.items():
                self.seg_counts[pid] = {}  # Initialize patient-level dictionary first
                for view, segs in views.items():
                    self.seg_counts[pid][view] = {int(k): v for k, v in segs.items()}
        except Exception as e:
            pass

    def _save_state(self):
        torch.save(self.memory, os.path.join(self.save_dir, 'plmb.pt'))
        with open(os.path.join(self.save_dir, 'meta.json'), 'w') as f:
            json.dump(self.seg_counts, f, indent=2)

    def set_training_mode(self, mode):
        self.training_mode = mode

    def init_patient(self, pid):
        if pid not in self.memory:
            self.memory[pid] = {'A2C':{i:[] for i in range(1,18)},
                                 'A3C':{i:[] for i in range(1,18)},
                                 'A4C':{i:[] for i in range(1,18)}}
        if pid not in self.seg_counts:
            self.seg_counts[pid] = {'A2C':{i:0 for i in range(1,18)},
                                     'A3C':{i:0 for i in range(1,18)},
                                     'A4C':{i:0 for i in range(1,18)}}
        if pid not in self.fusion_failures:
            self.fusion_failures[pid] = {'A2C':[], 'A3C':[], 'A4C':[]}

    def update(self, pid, view, seg_id, feat):
        self.init_patient(pid)
        mem = self.memory[pid][view]
        max_samp = 500
        if feat.shape[0] > max_samp:
            idx = torch.randperm(feat.shape[0])[:max_samp]
            feat = feat[idx]
        feat = F.normalize(feat, dim=1)
        mem[seg_id].append(feat.detach().cpu())
        if len(mem[seg_id]) > self.capacity:
            mem[seg_id].pop(0)
        self.seg_counts[pid][view][seg_id] += 1

    def retrieve_and_fuse(self, pid, view, feats, aha_mask):
        self.init_patient(pid)
        device = feats.device
        enhanced = feats.clone()
        other = [v for v in ['A2C','A3C','A4C'] if v != view]
        segs = get_aha_view_mapping()[view]
        self.fusion_failures[pid][view] = []
        
        for seg_id in segs:
            seg_mask_flat = (aha_mask == seg_id).flatten()
            seg_idx = torch.where(torch.from_numpy(seg_mask_flat).to(device))[0]
            if len(seg_idx) == 0:
                continue
            seg_feats = feats[seg_idx]
            # Normalize current features as well to ensure consistent similarity calculation
            seg_feats_norm = F.normalize(seg_feats, dim=1)
            
            hist = []
            for v in other:
                hist.extend(self.memory[pid][v][seg_id])
            if not hist:
                print(f"[PLMB Warning] Patient {pid}, View {view}, Seg {seg_id}: No history features, skip fusion")
                self.fusion_failures[pid][view].append(seg_id)
                continue
            hist = torch.cat(hist, dim=0).to(device)
            # Calculate similarity using normalized features
            sim = torch.matmul(seg_feats_norm, hist.t())
            attn = F.softmax(sim / (self.feature_dim**0.5), dim=1)
            fused = torch.matmul(attn, hist)
            enhanced[seg_idx] += self.fusion_alpha * fused
        return enhanced

    def get_fusion_failures(self, pid, view):
        self.init_patient(pid)
        return self.fusion_failures[pid][view]

    def clear_memory(self, pid):
        if pid in self.memory:
            del self.memory[pid]
        if pid in self.seg_counts:
            del self.seg_counts[pid]
        if pid in self.fusion_failures:
            del self.fusion_failures[pid]