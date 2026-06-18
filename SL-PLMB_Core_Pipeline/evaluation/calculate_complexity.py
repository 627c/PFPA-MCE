"""
calculate_complexity.py - Top-Journal Level Model Complexity Analysis Tool (Revised Version)
"""
import sys
import torch
from thop import profile
sys.path.append("/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE")
from models.net_seg_only import ADRNet
from models.baselines import (
    UNet, 
    AttentionUNet, 
    UNetPlusPlus, 
    nnUNet, 
    SwinUNet, 
    MobileUNet
)
def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    print("="*70)
    print("🚀 PFPA-MCE Segmentation Model Complexity Analysis (Params & FLOPs)")
    print("="*70)
    print("Input size: 1 × 512 × 512 (single-channel grayscale image)")
    print("Calculation note: MACs = multiply-accumulate operations; top medical journals usually refer to MACs directly as FLOPs")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Standard input for segmentation task
    dummy_input = torch.randn(1, 1, 512, 512).to(device)
    
    models_dict = {
        'MobileUNet': MobileUNet(),
        'UNet': UNet(),
        'Attention-UNet': AttentionUNet(),
        'UNet++': UNetPlusPlus(),
        'Swin-UNet': SwinUNet(),
        'nnU-Net': nnUNet(),
        'ADR-Net (Ours)': ADRNet()
    }
    
    print(f"| {'Model':<20} | {'Params (M)':<12} | {'MACs (G)':<12} | {'FLOPs (G)':<12} |")
    print(f"|{'-'*22}|{'-'*14}|{'-'*14}|{'-'*14}|")
    
    results = {}
    for name, model in models_dict.items():
        model = model.to(device)
        model.eval()  # Inference mode, does not affect parameter count and computation cost
        
        # Suppress thop internal prints and gradient computation
        with torch.no_grad():
            try:
                macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
            except Exception as e:
                print(f"{name} calculation failed: {str(e)}")
                continue
        
        # Unit conversion
        params_m = params / 1e6  # Million parameters
        macs_g = macs / 1e9      # Billion multiply-accumulate operations
        flops_g = macs_g * 2     # Strict FLOPs = 2 × MACs
        
        results[name] = {
            'params_m': params_m,
            'macs_g': macs_g,
            'flops_g': flops_g
        }
        
        # Formatted output
        print(f"| {name:<20} | {params_m:>10.2f} M | {macs_g:>10.2f} G | {flops_g:>10.2f} G |")
        
        # Release GPU memory immediately to avoid OOM
        del model
        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()