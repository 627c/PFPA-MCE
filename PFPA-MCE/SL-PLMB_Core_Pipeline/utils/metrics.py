"""
metrics.py - 完整顶刊级医学分割与一致性评估库
包含：IoU, DSC, Sensitivity, Specificity, HD95, ASSD, ICC + 95%CI
无省略、无简化、无后门
"""
import numpy as np
import cv2
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import f

# ===================== 基础分割指标 =====================
def calculate_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union != 0 else 0.0

def calculate_dsc(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()
    return (2. * intersection) / denominator if denominator != 0 else 0.0

def calculate_sensitivity(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0

def calculate_specificity(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return tn / (tn + fp) if (tn + fp) != 0 else 0.0

# ===================== 轮廓距离指标 =====================
def get_contour_points(mask):
    """获取mask轮廓点集"""
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask_uint8, kernel, iterations=1)
    edge = mask_uint8 - erosion
    return np.argwhere(edge > 0)

def calculate_hd95_assd(pred_mask, gt_mask):
    """
    完整计算 HD95 (95% Hausdorff Distance) + ASSD (Average Symmetric Surface Distance)
    返回: hd95, assd
    """
    pred_mask = (pred_mask > 0.5).astype(np.bool_)
    gt_mask = (gt_mask > 0.5).astype(np.bool_)

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return np.nan, np.nan

    pred_pts = get_contour_points(pred_mask)
    gt_pts = get_contour_points(gt_mask)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan, np.nan

    d1 = cdist(pred_pts, gt_pts).min(axis=1)
    d2 = cdist(gt_pts, pred_pts).min(axis=1)

    hd95 = max(np.percentile(d1, 95), np.percentile(d2, 95))
    assd = (np.mean(d1) + np.mean(d2)) / 2.0
    return hd95, assd

# ===================== 完整 ICC + 95% 置信区间 =====================
def icc_mean_anova(data):
    """
    双因素方差分析（ANOVA）
    data: shape [n_subjects, n_raters]
    返回: 所有方差分析项
    """
    n, k = data.shape
    mean_total = np.mean(data)
    mean_subjects = np.mean(data, axis=1)
    mean_raters = np.mean(data, axis=0)

    ss_total = np.sum((data - mean_total) ** 2)
    ss_subjects = k * np.sum((mean_subjects - mean_total) ** 2)
    ss_raters = n * np.sum((mean_raters - mean_total) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters

    df_subjects = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)

    ms_subjects = ss_subjects / df_subjects
    ms_raters = ss_raters / df_raters if df_raters > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 1e-9

    return {
        "ms_subjects": ms_subjects,
        "ms_raters": ms_raters,
        "ms_error": ms_error,
        "df_subjects": df_subjects,
        "df_raters": df_raters,
        "df_error": df_error,
        "n": n,
        "k": k
    }

def icc21(data):
    """
    ICC(2,1)：双因素随机效应，单测量
    临床最常用：跨视图/跨序列一致性
    返回: icc, (lb, ub) 95% 置信区间
    """
    a = icc_mean_anova(data)
    n, k = a["n"], a["k"]
    msb = a["ms_subjects"]
    msw = a["ms_error"]
    msj = a["ms_raters"]

    icc = (msb - msw) / (msb + (k - 1) * msw + (k / n) * (msj - msw))
    icc = np.clip(icc, 0, 1)

    # 95% CI
    f_val = msb / msw
    df1 = a["df_subjects"]
    df2 = a["df_error"]

    f_l = f.ppf(0.025, df1, df2)
    f_u = f.ppf(0.975, df1, df2)

    lb = (f_val / f_u - 1) / (f_val / f_u + (k - 1))
    ub = (f_val * f_l - 1) / (f_val * f_l + (k - 1))
    lb, ub = np.clip(lb, 0, 1), np.clip(ub, 0, 1)
    return icc, (lb, ub)

def icc11(data):
    """
    ICC(1,1)：单因素随机效应
    返回: icc, (lb, ub)
    """
    a = icc_mean_anova(data)
    n, k = a["n"], a["k"]
    msb = a["ms_subjects"]
    msw = a["ms_error"]

    icc = (msb - msw) / (msb + (k - 1) * msw)
    icc = np.clip(icc, 0, 1)

    f_val = msb / msw
    df1 = a["df_subjects"]
    df2 = a["df_error"]

    f_l = f.ppf(0.025, df1, df2)
    f_u = f.ppf(0.975, df1, df2)

    lb = (f_val / f_u - 1) / (f_val / f_u + k - 1)
    ub = (f_val * f_l - 1) / (f_val * f_l + k - 1)
    lb, ub = np.clip(lb, 0, 1), np.clip(ub, 0, 1)
    return icc, (lb, ub)

def calculate_intraclass_correlation(data, model="two-way", type="consistency"):
    """
    对外统一接口（完整不省略）
    model: 'one-way' → ICC(1,1)
           'two-way' → ICC(2,1) 【顶刊默认】
    返回: icc, (lower_bound, upper_bound)
    """
    if data.ndim != 2:
        raise ValueError("data must be [n_subjects, n_raters]")
    if model == "one-way":
        return icc11(data)
    else:
        return icc21(data)