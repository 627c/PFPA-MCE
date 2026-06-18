"""
perfusion.py - Perfusion parameter fitting
Implements pixel-level parallel fitting to generate A, β and MBF perfusion maps
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
    Classic microbubble replenishment model (Replenishment Kinetics Model):
    y(t) = A * (1 - e^(-beta * t)) + y0
    Time offset x0 is added to ensure the curve passes through the instant of contrast agent destruction
    """
    return A * ((1 - np.exp(-beta * t)) - (1 - np.exp(-beta * x0))) + y0
def fit_single_tac(tac_sequence, fps=30):
    """
    Fit a single TAC curve
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
                maxfev=2000  
            )
            A, beta = params
        except Exception:
            A, beta = 0.0, 0.0
    
    fitted_curve = exponential_model(time_axis, A, beta, x0, y0)
    return A, beta, fitted_curve
def _fit_pixel_wrapper(args):
    """Parallel computation wrapper function"""
    y, x, pixel_series, fps = args
    if np.max(pixel_series) < 1e-3:  # Skip background pixels
        return y, x, 0.0, 0.0
    
    # Time-frequency filtering
    from core.tac import time_frequency_filter
    purified = time_frequency_filter(pixel_series, fps=fps)
    
    # Fit parameters
    A, beta, _ = fit_single_tac(purified, fps=fps)
    return y, x, A, beta
def generate_perfusion_maps(video_frames, mask, fps=30, n_jobs=-1):
    """
    Generate pixel-level perfusion parameter maps
    Args:
        video_frames: (T, H, W) video frame array
        mask: (H, W) myocardial mask
        n_jobs: number of parallel processes, -1 uses all CPU cores
    Returns:
        A_map: blood volume map (H, W)
        beta_map: blood flow velocity map (H, W)
        mbf_map: myocardial blood flow map (A*beta) (H, W)
    """
    T, H, W = video_frames.shape
    A_map = np.zeros((H, W), dtype=np.float32)
    beta_map = np.zeros((H, W), dtype=np.float32)
    
    # Extract all myocardial pixels to be calculated
    ys, xs = np.where(mask > 0)
    
    # Build parallel tasks
    tasks = [(y, x, video_frames[:, y, x], fps) for y, x in zip(ys, xs)]
    
    # Multi-process parallel computation
    results = Parallel(n_jobs=n_jobs, batch_size='auto', timeout=300)(  
        delayed(_fit_pixel_wrapper)(task) for task in tasks
    )
    
    # Fill back results
    for y, x, A, beta in results:
        A_map[y, x] = A
        beta_map[y, x] = beta
    
    # Calculate myocardial blood flow
    mbf_map = A_map * beta_map
    
    return A_map, beta_map, mbf_map
def fit_tac_to_params(tac_sequence_path, output_fig_dir, fps=30):
    """
    Fit TAC sequence to perfusion parameters and plot results
    Args:
        tac_sequence_path: TAC sequence file path
        output_fig_dir: output directory
        fps: video frame rate
    Returns:
        A: blood volume parameter
        beta: blood flow velocity parameter
    """
    import os
    import matplotlib.pyplot as plt
    
    # Load TAC sequence
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
        print(f"Failed to load TAC sequence: {e}")
        return 0.0, 0.0
    
    # Fit parameters
    A, beta, fitted_curve = fit_single_tac(tac_sequence, fps)
    
    # Plot fitting results
    os.makedirs(output_fig_dir, exist_ok=True)
    time_axis = np.arange(len(tac_sequence)) / fps
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, tac_sequence, label='Raw TAC', color='blue', linewidth=2)
    plt.plot(time_axis, fitted_curve, label=f'Fitted curve (A={A:.4f}, β={beta:.4f})', 
             color='red', linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Mean Intensity', fontsize=12)
    plt.title('TAC Curve and Exponential Fitting Result', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_fig_dir, 'tac_fitting_result.png'), dpi=300)
    plt.close()
    
    # Save parameters
    fit_params = {
        'A': A,
        'beta': beta,
        'perfusion_index': A * beta,
        'time_axis': time_axis,
        'tac_sequence': tac_sequence,
        'fitted_curve': fitted_curve
    }
    
    np.save(os.path.join(output_fig_dir, 'fit_params.npy'), fit_params)
    
    # Save text format
    with open(os.path.join(output_fig_dir, 'perfusion_parameters.txt'), 'w') as f:
        f.write(f"Blood volume (A): {A:.4f}\n")
        f.write(f"Blood flow velocity (β): {beta:.4f}\n")
        f.write(f"Perfusion index (A×β): {A*beta:.4f}\n")
    
    print(f"Fitting completed: A={A:.4f}, β={beta:.4f}")
    return A, beta