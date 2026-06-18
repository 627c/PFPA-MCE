"""
evaluate.py
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
        test_patients: list/set, list of test patient IDs passed from main.py
        source_dir: directory for storing all preprocessed data
    """
    device = BASE_CONFIG['device']
    model = AttentionUNet().to(device)
    
    if not os.path.exists(PATHS['model_save']):
        print(f"Model weight file not found: {PATHS['model_save']}")
        return
        
    model.load_state_dict(torch.load(PATHS['model_save'], map_location=device))
    model.eval()

    # 1. Initialize full dataset (load all preprocessed triplet data)
    full_dataset = MCEDatasetNpy(
        image_dir=os.path.join(source_dir, 'images'),
        mask_dir=os.path.join(source_dir, 'masks'),
        perfusion_dir=os.path.join(source_dir, 'perfusion'),
        use_aha=False  # Pure MSE evaluation, no AHA segmentation needed
    )
    
    # 2. Dynamically filter test samples to prevent data leakage
    test_filenames = []
    for f in full_dataset.filenames:
        pid, _ = extract_patient_and_view(f)
        if pid in test_patients:
            test_filenames.append(f)
    
    full_dataset.filenames = test_filenames
    
    if len(full_dataset) == 0:
        print("Evaluation set is empty, please check if test_patients list is passed correctly!")
        return
    loader = torch.utils.data.DataLoader(full_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    all_mse_A, all_mse_beta = [], []
    
    print("Calculating perfusion parameter regression error (MSE)...")
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].to(device)
            mask = batch['mask'].to(device) # Pseudo-label mask loaded here, only used for locating myocardium
            gt_perf = batch.get('perfusion')
            
            # Inference
            _, pred_perf, _ = model(img)
            
            # Perfusion error calculation (MSE) - absolute core!
            if gt_perf is not None:
                gt_perf = gt_perf.to(device)
                # Use mask as boolean index to exclude contamination from background zero values
                myo = (mask > 0.5).squeeze(1)  # [B, H, W]
                
                if myo.sum() > 0:
                    err_A = ((pred_perf[:, 0][myo] - gt_perf[:, 0][myo])) ** 2
                    err_beta = ((pred_perf[:, 1][myo] - gt_perf[:, 1][myo])) ** 2
                    
                    all_mse_A.append(err_A.mean().item())
                    all_mse_beta.append(err_beta.mean().item())

    # 3. Print pure quantitative evaluation results
    print("\n" + "="*50)
    print("Quantitative perfusion evaluation results for dynamic test zone (video triplets)")
    print("="*50)
    print(f"Test set size: {len(full_dataset)} triplet samples")
    
    if all_mse_A:
        print(f"Blood volume (A) MSE:     {np.mean(all_mse_A):.4f}")
        print(f"Blood flow velocity (β) MSE:  {np.mean(all_mse_beta):.4f}")
        print("Note: DSC/IoU are not calculated here, as the masks are self-generated pseudo-labels by the model with no statistical significance.")
    else:
        print("Warning: No real Perfusion Ground Truth found in the dataset!")
    print("="*50)

if __name__ == "__main__":
    # This is an independent debugging entry. In actual operation, it should be called via main.py with correct test_patients passed in.
    print("Please call evaluate_full_pipeline() in the main.py workflow.")