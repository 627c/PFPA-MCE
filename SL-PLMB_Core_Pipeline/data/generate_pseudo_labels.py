"""
generate_pseudo_labels.py - GUI-free Interactive Annotator for Servers
Workflow:
1. The script automatically generates key frame preview images
2. Download the preview images to local via scp for inspection
3. Enter the number of the best frame in the terminal
4. The script generates the mask for that frame
5. Enter y to save, n to reselect, t to adjust threshold
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
# ===================== Configuration Area =====================
IMG_SIZE = (512, 512)
ROOT_DIR = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE"
CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints/best_seg_model.pth')
VIDEO_DIR = os.path.join(ROOT_DIR, 'raw_data/videos/')
MASK_OUTPUT_DIR = os.path.join(ROOT_DIR, 'raw_data/pseudo_masks/')
PREVIEW_DIR = os.path.join(ROOT_DIR, 'raw_data/preview_frames/')
# ==========================================
def keep_image_size_open(img):
    """Preprocessing function (unchanged)"""
    img = Image.fromarray(img).convert('L')
    ratio = min(IMG_SIZE[0] / img.size[0], IMG_SIZE[1] / img.size[1])
    new_w, new_h = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    new_img = Image.new('L', IMG_SIZE, 0)
    new_img.paste(img, ((IMG_SIZE[0] - new_w) // 2, (IMG_SIZE[1] - new_h) // 2))
    return np.array(new_img)
def predict_mask(frame, model, device, threshold=0.3):
    """Call AI to predict mask (unchanged)"""
    img_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        # TTA augmented prediction
        pred1 = model(img_t)
        pred2 = model(torch.flip(img_t, dims=[3]))
        pred_mask = (pred1 + torch.flip(pred2, dims=[3])) / 2.0
        pred_mask = torch.sigmoid(pred_mask)[0,0].cpu().numpy()
    
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return binary_mask
def generate_preview_grid(frames, video_name, num_frames=10):
    """Generate key frame preview grid image"""
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    
    # Uniformly sample 10 key frames
    sample_indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    
    # Create 2-row 5-column grid
    grid = []
    for i in range(2):
        row = []
        for j in range(5):
            idx = sample_indices[i*5 + j]
            frame = (frames[idx] * 255).astype(np.uint8)
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Add frame number label
            cv2.putText(frame_color, f"{idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            row.append(frame_color)
        grid.append(np.hstack(row))
    
    preview_img = np.vstack(grid)
    preview_path = os.path.join(PREVIEW_DIR, f"{os.path.splitext(video_name)[0]}_preview.png")
    cv2.imwrite(preview_path, preview_img)
    
    return preview_path, sample_indices
def generate_mask_preview(frame, mask, video_name, frame_idx, threshold):
    """Generate mask preview image"""
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
    print("GUI-free Interactive Mask Generator Starting...")
    print("="*60)
    
    # 1. Check paths
    print(f"\n Checking path configuration:")
    print(f"   Model path: {CHECKPOINT_PATH}")
    print(f"   Video directory: {VIDEO_DIR}")
    print(f"   Output directory: {MASK_OUTPUT_DIR}")
    print(f"   Preview directory: {PREVIEW_DIR}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n Error: model file not found at {CHECKPOINT_PATH}")
        return
    
    if not os.path.exists(VIDEO_DIR):
        print(f"\n Error: video directory not found at {VIDEO_DIR}")
        return
    
    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    
    # 2. Load model
    print(f"\n Loading segmentation model...")
    try:
        device = BASE_CONFIG['device']
        model = AttentionUNet().to(device)
        
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            best_dsc = checkpoint.get('best_dsc', 0.0)
            print(f" Model loaded successfully! Best validation DSC: {best_dsc:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f" Legacy model loaded successfully!")
        
        model.eval()
    except Exception as e:
        print(f"\n Model loading failed: {str(e)}")
        return
    
    # 3. Get video list
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi'))]
    
    if len(videos) == 0:
        print(f"\n Error: no video files found in {VIDEO_DIR}")
        return
    
    print(f"\n📹 Found {len(videos)} video files")
    
    # 4. Process each video
    for i, vid_name in enumerate(videos):
        mask_path = os.path.join(MASK_OUTPUT_DIR, os.path.splitext(vid_name)[0] + '.png')
        
        if os.path.exists(mask_path):
            print(f"\n [{i+1}/{len(videos)}] Skipped (already processed): {vid_name}")
            continue
        
        print(f"\n [{i+1}/{len(videos)}] Processing: {vid_name}")
        
        # Read video
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
                print(f"    Cannot read video or video is empty, skipped")
                continue
            
            print(f"    Successfully read video, total {len(frames)} frames")
        except Exception as e:
            print(f"    Video reading failed: {str(e)}")
            continue
        
        # Generate key frame preview
        preview_path, sample_indices = generate_preview_grid(frames, vid_name)
        print(f"\n    Key frame preview generated: {preview_path}")
        print(f"    Please run the following command in your local terminal to download the preview:")
        print(f"      scp stu1@your_server_ip:{preview_path} ./")
        print(f"    After viewing the preview, enter the number of the best frame (0-{len(frames)-1})")
        
        current_threshold = 0.3
        
        while True:
            # Get frame number input from user
            while True:
                user_input = input("\n   Please enter the best frame number (enter q to quit): ")
                
                if user_input.lower() == 'q':
                    print("\n Program exited")
                    return
                
                try:
                    frame_idx = int(user_input)
                    if 0 <= frame_idx < len(frames):
                        break
                    else:
                        print(f"    Frame number must be between 0 and {len(frames)-1}")
                except ValueError:
                    print("    Please enter a valid number")
            
            # Predict mask
            print(f"    AI is predicting mask for frame {frame_idx}...")
            mask = predict_mask(frames[frame_idx], model, device, current_threshold)
            
            # Generate mask preview
            mask_preview_path = generate_mask_preview(frames[frame_idx], mask, vid_name, frame_idx, current_threshold)
            print(f"    Mask preview generated: {mask_preview_path}")
            print(f"    Please download and inspect the mask preview")
            
            # Get user feedback
            while True:
                feedback = input("\n   Satisfied with the mask? (y=save, n=reselect frame, t=adjust threshold): ").lower()
                
                if feedback == 'y':
                    # Save mask
                    try:
                        cv2.imwrite(mask_path, mask)
                        print(f"    Mask saved: {mask_path}")
                        break
                    except Exception as e:
                        print(f"    Save failed: {str(e)}")
                elif feedback == 'n':
                    print("    Reselecting frame")
                    break
                elif feedback == 't':
                    # Adjust threshold
                    while True:
                        threshold_input = input(f"   Please enter new threshold (current: {current_threshold:.2f}, range 0.1-0.5): ")
                        try:
                            new_threshold = float(threshold_input)
                            if 0.1 <= new_threshold <= 0.5:
                                current_threshold = new_threshold
                                print(f"    Threshold adjusted to: {current_threshold:.2f}")
                                # Re-predict mask
                                mask = predict_mask(frames[frame_idx], model, device, current_threshold)
                                mask_preview_path = generate_mask_preview(frames[frame_idx], mask, vid_name, frame_idx, current_threshold)
                                print(f"    New mask preview generated: {mask_preview_path}")
                                break
                            else:
                                print("    Threshold must be between 0.1 and 0.5")
                        except ValueError:
                            print("    Please enter a valid number")
                else:
                    print("    Please enter y, n or t")
            
            if feedback == 'y':
                break
    
    print("\n" + "="*60)
    print(" All videos processed!")
    print(f" Results saved to: {MASK_OUTPUT_DIR}")
    print("="*60)
if __name__ == "__main__":
    main()