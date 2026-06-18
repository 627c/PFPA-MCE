"""
tac.py - Global Mask and Core Time-Frequency Purification Engine
"""
import os
import numpy as np
import cv2
import torch
from scipy.signal import stft, istft, find_peaks
from config import BASE_CONFIG, PATHS
from models.net import AttentionUNet
from utils.helpers import extract_temporal_triplet

SIGNAL_PARAMS = {
    'fps': 30,
    'heart_rate_min': 50,
    'heart_rate_max': 120,
    'freq_low': 0.5,
    'freq_high': 15.0,
    'stft_nperseg': 64
}

def detect_heart_rate(global_tac, fps=30):
    """
    Detect heart rate based on global TAC and return arrhythmia abnormal mask
    """
    hr_min, hr_max = SIGNAL_PARAMS['heart_rate_min'], SIGNAL_PARAMS['heart_rate_max']
    min_dist = int(fps * 60 / hr_max)
    peaks, _ = find_peaks(global_tac, distance=min_dist)
    
    abnormal_mask = np.zeros(len(global_tac), dtype=bool)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fps
        mean_rr = np.mean(rr_intervals)
        for i, rr in enumerate(rr_intervals):
            if rr < 60/hr_max or rr > 60/hr_min or abs(rr - mean_rr) > mean_rr * 0.2:
                start = peaks[i]
                end = peaks[i+1] if i+1 < len(peaks) else len(global_tac)
                abnormal_mask[start:end] = True
                
    return abnormal_mask, peaks

def time_frequency_filter(signal_1d, fps=30, abnormal_mask=None):
    """
    Core Time-Frequency Domain Dual Filter (Time-Frequency Filter)
    Uses STFT to remove high-frequency speckle noise and low-frequency motion artifacts in the frequency domain, and masks arrhythmia intervals in the time domain.
    """
    if len(signal_1d) < SIGNAL_PARAMS['stft_nperseg']:
        return signal_1d
        
    nperseg = SIGNAL_PARAMS['stft_nperseg']
    noverlap = nperseg // 2
    
    # 1. Frequency-domain purification
    freqs, times, Zxx = stft(signal_1d, fs=fps, nperseg=nperseg, noverlap=noverlap, boundary='even')
    mask = (freqs >= SIGNAL_PARAMS['freq_low']) & (freqs <= SIGNAL_PARAMS['freq_high'])
    
    Zxx_filtered = Zxx.copy()
    Zxx_filtered[~mask] = 0
    
    _, purified = istft(Zxx_filtered, fs=fps, nperseg=nperseg, noverlap=noverlap, boundary='even')
    purified = purified[:len(signal_1d)]
    
    # 2. Time-domain interpolation for abnormal cycles
    if abnormal_mask is not None and np.any(abnormal_mask):
        valid_idx = np.where(~abnormal_mask)[0]
        abnormal_idx = np.where(abnormal_mask)[0]
        if len(valid_idx) > 0:
            purified[abnormal_idx] = np.interp(abnormal_idx, valid_idx, purified[valid_idx])
            
    return purified

def extract_purified_tac(
    video_path, image_dir, mask_dir, model_weight_path,
    output_tac_dir, fps=30, device=None, plmb=None
):
    """
    Extract global TAC signal from video and perform time-frequency purification
    Note: This function depends on the model and is only called in main.py
    """
    device = device or BASE_CONFIG['device']
    os.makedirs(output_tac_dir, exist_ok=True)
    model = AttentionUNet().to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        gray = cv2.resize(gray, (BASE_CONFIG['img_size'], BASE_CONFIG['img_size']))
        frames.append(gray / 255.0)
    cap.release()

    video_tensor = np.array(frames)
    triplet, _ = extract_temporal_triplet(video_tensor)
    img_t = torch.from_numpy(triplet).unsqueeze(0).float().to(device)

    with torch.no_grad():
        seg_logits, _ = model.forward_features(img_t)
        global_mask = (torch.sigmoid(seg_logits)[0,0].cpu().numpy() > 0.5)

    global_tac = np.mean(video_tensor, axis=(1,2))
    abnormal_mask, _ = detect_heart_rate(global_tac, fps)

    H, W = global_mask.shape
    ys, xs = np.where(global_mask)
    purified_video = np.zeros_like(video_tensor)

    for y, x in zip(ys, xs):
        sig = video_tensor[:, y, x]
        purified_video[:, y, x] = time_frequency_filter(sig, fps, abnormal_mask)

    raw_tac = np.mean(video_tensor[:, ys, xs], axis=1)*255
    purified_tac = np.mean(purified_video[:, ys, xs], axis=1)*255

    np.save(os.path.join(output_tac_dir, 'full_tac.npy'), {
        'raw_tac': raw_tac,
        'purified_tac': purified_tac,
        'purified_video': purified_video,
        'global_mask': global_mask
    })

    return purified_tac