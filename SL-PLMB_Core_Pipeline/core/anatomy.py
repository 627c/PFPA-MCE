"""
anatomy.py - Ultimate Clinical Gold Standard Version (Polar Coordinate Left-Right Separation + Geodesic Independent Equidistant Segmentation)

"""
import numpy as np
import cv2
def get_aha_view_mapping():
    """
    7-segment mapping table for three standard views per ASE 2015
    Order: [left apex, left mid, left base, right base, right mid, right apex, apical cap]
    (Unified convention: left wall from bottom to top, right wall from top to bottom, with apical cap in between)
    """
    return {
        # A4C: left side is inferoseptal (14, 9, 3), right side is anterolateral (16, 12, 6)
        'A4C': [14, 9, 3, 6, 12, 16, 17], 
        # A2C: left side is inferior (15, 10, 4), right side is anterior (13, 7, 1)
        'A2C': [15, 10, 4, 1, 7, 13, 17],  
        # A3C: left side is inferolateral (15, 11, 5), right side is anteroseptal (14, 8, 2)
        'A3C': [15, 11, 5, 2, 8, 14, 17]   
    }
def generate_aha17_mask(mask, view_name):
    """
    AHA 17-segment generator based on polar coordinate separation and geodesic equal division
    """
    aha_map = np.zeros_like(mask, dtype=np.int32)
    binary_mask = (mask > 127).astype(np.uint8)
    
    view_segs = get_aha_view_mapping().get(view_name)
    if not view_segs or binary_mask.sum() < 100:
        return aha_map
        
    ys, xs = np.where(binary_mask > 0)
    coords = np.column_stack((ys, xs))  # (y, x)
    
    # ===================== Step 1: Robust Apex Localization =====================
    # Take the top 5% points (smallest y values) and compute their center as the apex
    min_y = np.min(ys)
    max_y = np.max(ys)
    top_thresh = min_y + (max_y - min_y) * 0.05
    apex_pts = coords[coords[:, 0] <= top_thresh]
    apex = np.mean(apex_pts, axis=0) # [y_apex, x_apex]
    
    # ===================== Step 2: Extract Apical Cap (S17) =====================
    # Use the apex as the center and take a certain physical radius as the apical cap
    # The radius is dynamically determined by the overall myocardial span to prevent deformation from stretching
    total_height = max_y - min_y
    apex_cap_radius = total_height * 0.18 # Approximately 18% of the height is the apical cap
    
    dist_to_apex = np.linalg.norm(coords - apex, axis=1)
    apex_cap_mask = dist_to_apex <= apex_cap_radius
    
    for pt in coords[apex_cap_mask]:
        aha_map[pt[0], pt[1]] = view_segs[6] # Fill S17
        
    # Exclude the apical cap, remaining points are used for left and right wall segmentation
    remain_coords = coords[~apex_cap_mask]
    if len(remain_coords) < 10:
        return aha_map
        
    # ===================== Step 3: Perfect Left-Right Separation via Polar Coordinates =====================
    # Use the apex as the pole and calculate the polar angle of all remaining points (relative to the downward vertical line)
    dy = remain_coords[:, 0] - apex[0]
    dx = remain_coords[:, 1] - apex[1]
    
    # Calculate angle with atan2, taking straight downward (dy>0, dx=0) as 0 degrees
    # dx < 0 (left side) has negative angles; dx > 0 (right side) has positive angles
    angles = np.arctan2(dx, dy)
    
    left_coords = remain_coords[angles < 0]
    right_coords = remain_coords[angles >= 0]
    
    # ===================== Step 4: Independent Three-Fold Division via Snake Geodesic =====================
    def partition_wall_geodesic(wall_coords, apex_point, num_segs=3):
        """Calculate curve length along the real myocardial contour and divide equally"""
        if len(wall_coords) < 10: return []
        
        # 4.1 Find path starting point: the point closest to the apex
        dists_to_apex = np.linalg.norm(wall_coords - apex_point, axis=1)
        start_idx = np.argmin(dists_to_apex)
        
        # 4.2 Snake nearest-neighbor sorting: connects points by physical topology without relying on Y-coordinates
        ordered_coords = []
        unvisited = list(range(len(wall_coords)))
        current_idx = start_idx
        
        # For acceleration, random downsampling skeletonization is applied if there are too many points (for distance calculation only)
        # For accuracy here, all points are retained and the nearest point is found each iteration
        ordered_coords.append(wall_coords[current_idx])
        unvisited.remove(current_idx)
        
        while unvisited:
            curr_pt = ordered_coords[-1]
            # Calculate distance from current point to all unvisited points
            rem_pts = wall_coords[unvisited]
            dists = np.sum((rem_pts - curr_pt)**2, axis=1) # Squared distance, avoids sqrt to improve speed
            next_idx_in_rem = np.argmin(dists)
            
            # If the nearest point is too far away (mask fracture), fall back to y-coordinate sorting directly
            if dists[next_idx_in_rem] > 400: # Empirical threshold 20^2
                break
                
            real_idx = unvisited[next_idx_in_rem]
            ordered_coords.append(wall_coords[real_idx])
            unvisited.remove(real_idx)
            
        # If there are remaining unconnected points due to fracture, append them sorted by y-coordinate in descending order
        if unvisited:
            rem_arr = wall_coords[unvisited]
            rem_arr = rem_arr[np.argsort(rem_arr[:, 0])] # Sort downward
            ordered_coords.extend(rem_arr)
            
        ordered_coords = np.array(ordered_coords)
        
        # 4.3 Calculate cumulative curve length along the real path
        diffs = np.diff(ordered_coords, axis=0)
        step_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_len = np.insert(np.cumsum(step_lengths), 0, 0)
        
        total_len = cumulative_len[-1]
        
        # 4.4 Perform equidistant segmentation based on total curve length
        cut_points = [total_len * i / num_segs for i in range(1, num_segs)]
        
        partitions = []
        start_idx = 0
        for cut in cut_points:
            end_idx = np.argmin(np.abs(cumulative_len - cut))
            partitions.append(ordered_coords[start_idx:end_idx])
            start_idx = end_idx
        partitions.append(ordered_coords[start_idx:])
        
        return partitions
    # Independent segmentation for left and right walls (returned list order: [apical segment, mid segment, basal segment])
    left_parts = partition_wall_geodesic(left_coords, apex, 3)
    right_parts = partition_wall_geodesic(right_coords, apex, 3)
    
    # ===================== Step 5: Mapping and Filling =====================
    # Order of view_segs: [left apex, left mid, left base, right base, right mid, right apex, apical cap]
    
    # Left wall (filled from apex to base)
    if len(left_parts) == 3:
        for pt in left_parts[0]: aha_map[pt[0], pt[1]] = view_segs[0] # Left apex
        for pt in left_parts[1]: aha_map[pt[0], pt[1]] = view_segs[1] # Left mid
        for pt in left_parts[2]: aha_map[pt[0], pt[1]] = view_segs[2] # Left base
        
    # Right wall (filled from apex to base, note the indices of the mapping table)
    if len(right_parts) == 3:
        for pt in right_parts[0]: aha_map[pt[0], pt[1]] = view_segs[5] # Right apex
        for pt in right_parts[1]: aha_map[pt[0], pt[1]] = view_segs[4] # Right mid
        for pt in right_parts[2]: aha_map[pt[0], pt[1]] = view_segs[3] # Right base
    return aha_map