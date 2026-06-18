"""
anatomy.py - 终极临床金标准版 (极坐标左右分离 + 测地线独立等距切分)
✅ 核心修复：彻底解决长轴歪斜导致的左右节段跨界问题 (如 S11 跑到左侧)。
✅ 核心修复：摒弃错误的 Y 轴排序，采用最近邻测地线追踪计算真实心肌曲线长度。
✅ 严格遵守 ASE 2015 AHA 17 节段指南。
"""
import numpy as np
import cv2

def get_aha_view_mapping():
    """
    ASE 2015 标准三个切面的 7 节段映射表
    顺序：[左心尖, 左中段, 左基底, 右基底, 右中段, 右心尖, 心尖帽]
    (这里统一按照：左壁从下到上，右壁从上到下，中间夹着心尖帽)
    """
    return {
        # A4C: 左侧是下间隔(Inferoseptal 14,9,3)，右侧是前侧壁(Anterolateral 16,12,6)
        'A4C': [14, 9, 3, 6, 12, 16, 17], 
        # A2C: 左侧是下壁(Inferior 15,10,4)，右侧是前壁(Anterior 13,7,1)
        'A2C': [15, 10, 4, 1, 7, 13, 17],  
        # A3C: 左侧是下侧壁(Inferolateral 15,11,5)，右侧是前间隔(Anteroseptal 14,8,2)
        'A3C': [15, 11, 5, 2, 8, 14, 17]   
    }

def generate_aha17_mask(mask, view_name):
    """
    基于极坐标分离与测地线等分的 AHA 17 节段生成器
    """
    aha_map = np.zeros_like(mask, dtype=np.int32)
    binary_mask = (mask > 127).astype(np.uint8)
    
    view_segs = get_aha_view_mapping().get(view_name)
    if not view_segs or binary_mask.sum() < 100:
        return aha_map
        
    ys, xs = np.where(binary_mask > 0)
    coords = np.column_stack((ys, xs))  # (y, x)
    
    # ===================== 步骤 1：鲁棒定位心尖 (Apex) =====================
    # 取最上方 (y最小) 的 5% 的点，计算它们的中心作为心尖
    min_y = np.min(ys)
    max_y = np.max(ys)
    top_thresh = min_y + (max_y - min_y) * 0.05
    apex_pts = coords[coords[:, 0] <= top_thresh]
    apex = np.mean(apex_pts, axis=0) # [y_apex, x_apex]
    
    # ===================== 步骤 2：提取心尖帽 (S17) =====================
    # 以心尖为圆心，取一定物理半径作为心尖帽
    # 这个半径动态取决于整个心肌的跨度，防止被拉伸变形
    total_height = max_y - min_y
    apex_cap_radius = total_height * 0.18 # 约 18% 高度为心尖帽
    
    dist_to_apex = np.linalg.norm(coords - apex, axis=1)
    apex_cap_mask = dist_to_apex <= apex_cap_radius
    
    for pt in coords[apex_cap_mask]:
        aha_map[pt[0], pt[1]] = view_segs[6] # 填充 S17
        
    # 剔除心尖帽，剩余点用于左右壁分段
    remain_coords = coords[~apex_cap_mask]
    if len(remain_coords) < 10:
        return aha_map
        
    # ===================== 步骤 3：极坐标完美左右分离 =====================
    # 以心尖为极点，计算所有剩余点的极角 (相对于向下的垂直线)
    dy = remain_coords[:, 0] - apex[0]
    dx = remain_coords[:, 1] - apex[1]
    
    # 用 atan2 计算角度，以正下方 (dy>0, dx=0) 为 0 度
    # dx < 0 (左侧) 角度为负；dx > 0 (右侧) 角度为正
    angles = np.arctan2(dx, dy)
    
    left_coords = remain_coords[angles < 0]
    right_coords = remain_coords[angles >= 0]
    
    # ===================== 步骤 4：贪吃蛇测地线独立三等分 =====================
    def partition_wall_geodesic(wall_coords, apex_point, num_segs=3):
        """沿着心肌真实轮廓计算曲线长度，并等分"""
        if len(wall_coords) < 10: return []
        
        # 4.1 寻找路径起点：距离心尖最近的那个点
        dists_to_apex = np.linalg.norm(wall_coords - apex_point, axis=1)
        start_idx = np.argmin(dists_to_apex)
        
        # 4.2 贪吃蛇最近邻排序：不依赖Y坐标，按物理拓扑连接点
        ordered_coords = []
        unvisited = list(range(len(wall_coords)))
        current_idx = start_idx
        
        # 为了加速计算，如果点太多，进行随机降采样骨架化 (仅用于算距离)
        # 这里为了精准，保留所有点，但每次找最近点
        ordered_coords.append(wall_coords[current_idx])
        unvisited.remove(current_idx)
        
        while unvisited:
            curr_pt = ordered_coords[-1]
            # 计算当前点到所有未访问点的距离
            rem_pts = wall_coords[unvisited]
            dists = np.sum((rem_pts - curr_pt)**2, axis=1) # 平方距离，避免开方提升速度
            next_idx_in_rem = np.argmin(dists)
            
            # 如果最近的点都离得很远(掩膜断裂)，直接按 y 排序兜底
            if dists[next_idx_in_rem] > 400: # 经验阈值 20^2
                break
                
            real_idx = unvisited[next_idx_in_rem]
            ordered_coords.append(wall_coords[real_idx])
            unvisited.remove(real_idx)
            
        # 如果有断裂没连完的，把剩下的按 y 坐标大到小排在后面
        if unvisited:
            rem_arr = wall_coords[unvisited]
            rem_arr = rem_arr[np.argsort(rem_arr[:, 0])] # 往下排
            ordered_coords.extend(rem_arr)
            
        ordered_coords = np.array(ordered_coords)
        
        # 4.3 沿着真实路径计算累计曲线长度
        diffs = np.diff(ordered_coords, axis=0)
        step_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_len = np.insert(np.cumsum(step_lengths), 0, 0)
        
        total_len = cumulative_len[-1]
        
        # 4.4 根据总曲线长度进行等距离切分
        cut_points = [total_len * i / num_segs for i in range(1, num_segs)]
        
        partitions = []
        start_idx = 0
        for cut in cut_points:
            end_idx = np.argmin(np.abs(cumulative_len - cut))
            partitions.append(ordered_coords[start_idx:end_idx])
            start_idx = end_idx
        partitions.append(ordered_coords[start_idx:])
        
        return partitions

    # 左右独立切分 (返回的列表顺序是：[心尖段, 中段, 基底段])
    left_parts = partition_wall_geodesic(left_coords, apex, 3)
    right_parts = partition_wall_geodesic(right_coords, apex, 3)
    
    # ===================== 步骤 5：映射与填色 =====================
    # view_segs 的顺序是: [左心尖, 左中段, 左基底, 右基底, 右中段, 右心尖, 心尖帽]
    
    # 左壁 (从心尖往基底填充)
    if len(left_parts) == 3:
        for pt in left_parts[0]: aha_map[pt[0], pt[1]] = view_segs[0] # 左心尖
        for pt in left_parts[1]: aha_map[pt[0], pt[1]] = view_segs[1] # 左中
        for pt in left_parts[2]: aha_map[pt[0], pt[1]] = view_segs[2] # 左基底
        
    # 右壁 (从心尖往基底填充，注意映射表的索引)
    if len(right_parts) == 3:
        for pt in right_parts[0]: aha_map[pt[0], pt[1]] = view_segs[5] # 右心尖
        for pt in right_parts[1]: aha_map[pt[0], pt[1]] = view_segs[4] # 右中
        for pt in right_parts[2]: aha_map[pt[0], pt[1]] = view_segs[3] # 右基底

    return aha_map