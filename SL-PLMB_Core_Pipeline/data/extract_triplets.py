"""
extract_triplets.py - Automatic extraction and export tool for physiological triplet frames from MCE videos
Function: input an .avi video, automatically identify Flash, accurately locate and export three high-definition ground truth grayscale images: Destroy, Mid, Plateau.
"""
import os
import cv2
import numpy as np

def auto_extract_mce_triplets(video_path, output_dir):
    """
    Automatically locate and save triplet grayscale images based on microbubble perfusion kinetics
    """
    if not os.path.exists(video_path):
        print(f"Error: input video file not found: {video_path}")
        return False

    # 1. Initialize video stream reader
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps) or fps > 100:
        fps = 30.0  # Fallback standard ultrasound frame rate
    
    raw_frames = []
    mean_intensities = []
    
    print(f"Reading video and calculating global acoustic intensity curve...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale for global mean intensity calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        raw_frames.append(frame)  # Keep original color frames for subsequent conversion
        mean_intensities.append(np.mean(gray))
        
    cap.release()
    
    total_frames = len(raw_frames)
    mean_intensities = np.array(mean_intensities)
    
    if total_frames < 15:
        print(f"Error: total video frames too short ({total_frames} frames), unable to perform physiological kinetic analysis.")
        return False

    print(f"Video read successfully, total {total_frames} frames | Detected video frame rate (FPS): {fps:.2f}")

    # =========================================================================
    # 2. Core kinetic algorithm localization: three key physiological landmarks
    # =========================================================================
    
    # Point 1: Find the absolute peak of the Flash spike
    # Clinically, due to high-energy pulse bubble destruction, the instrument screen produces an instantaneous full-white extremely bright physical interruption
    flash_spike_idx = np.argmax(mean_intensities)
    
    # Point 2: Find the Destroy frame (true end of bubble destruction)
    # The Flash spike itself is bright, but the moment when all microbubbles are destroyed after the flash is the physical trough of darkest myocardium
    # We search for the absolute minimum intensity within 2 seconds (2 * fps frames) after the Flash
    search_dark_limit = min(total_frames, flash_spike_idx + int(fps * 2))
    search_dark_zone = mean_intensities[flash_spike_idx : search_dark_limit]
    
    if len(search_dark_zone) == 0:
        destroy_idx = flash_spike_idx
    else:
        destroy_idx = flash_spike_idx + np.argmin(search_dark_zone)
        
    # Point 3: Find the Plateau frame (contrast reperfusion saturation plateau phase)
    # When microbubbles are fully replenished, the curve enters a long-term stable plateau. We use a 5-frame moving average to smooth the signal and find the late perfusion peak
    post_destroy_zone = mean_intensities[destroy_idx + 1:]
    if len(post_destroy_zone) < 5:
        plateau_idx = total_frames - 1
    else:
        smoothed_post_zone = np.convolve(post_destroy_zone, np.ones(5)/5, mode='same')
        plateau_idx = destroy_idx + 1 + np.argmax(smoothed_post_zone)
        
    # Safety boundary lock: prevent plateau frame from accidentally reaching the end of the sequence due to respiratory drift and interfering with observation
    plateau_idx = min(plateau_idx, total_frames - 1)
    if plateau_idx <= destroy_idx:
        plateau_idx = total_frames - 1

    # Point 4: Find the Mid frame (50% perfusion half-recovery point)
    # The brightness of this frame is exactly halfway between the fully dark destruction trough and the saturation plateau, representing the velocity curvature of blood flow reperfusion
    y_destroy = mean_intensities[destroy_idx]
    y_plateau = mean_intensities[plateau_idx]
    target_mid_intensity = y_destroy + (y_plateau - y_destroy) * 0.5
    
    # Search for the frame closest to 50% recovery brightness between Destroy and Plateau
    search_mid_zone = mean_intensities[destroy_idx : plateau_idx]
    if len(search_mid_zone) == 0:
        mid_idx = destroy_idx + (plateau_idx - destroy_idx) // 2
    else:
        mid_idx = destroy_idx + np.argmin(np.abs(search_mid_zone - target_mid_intensity))

    # =========================================================================
    # 3. Image export and localization report (modified to save as grayscale images)
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Construct standard top-journal format filenames for saving
    destroy_img_path = os.path.join(output_dir, f"{video_base_name}_triplet_1_destroy.png")
    mid_img_path = os.path.join(output_dir, f"{video_base_name}_triplet_2_mid.png")
    plateau_img_path = os.path.join(output_dir, f"{video_base_name}_triplet_3_plateau.png")
    
    # Only modify here: convert to grayscale before saving
    cv2.imwrite(destroy_img_path, cv2.cvtColor(raw_frames[destroy_idx], cv2.COLOR_BGR2GRAY))
    cv2.imwrite(mid_img_path, cv2.cvtColor(raw_frames[mid_idx], cv2.COLOR_BGR2GRAY))
    cv2.imwrite(plateau_img_path, cv2.cvtColor(raw_frames[plateau_idx], cv2.COLOR_BGR2GRAY))
    
    print("\n" + "="*60)
    print("MCE Physiological Timeline Triplet Frame Automatic Extraction Report")
    print("="*60)
    print(f"Analyzing case video: {video_base_name}.avi")
    print(f"Detected Flash spike mutation point: frame {flash_spike_idx}")
    print(f"[1/3] Exported Destroy frame (fully dark): frame {destroy_idx} (time: {destroy_idx/fps:.2f}s)")
    print(f"[2/3] Exported Mid frame     (50% perfusion):  frame {mid_idx} (time: {mid_idx/fps:.2f}s)")
    print(f"[3/3] Exported Plateau frame (fully filled): frame {plateau_idx} (time: {plateau_idx/fps:.2f}s)")
    print(f"High-definition ground truth grayscale images successfully exported to directory: {output_dir}")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    video_file_path = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/raw_data/videos/P005_A4C.avi"
    output_directory = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results/extracted_triplets/"
    
    auto_extract_mce_triplets(video_file_path, output_directory)