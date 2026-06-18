"""
calculate_all_icc.py - 终极学术规范 ICC(2,1) 评估
✅ 手动实现标准 ICC(2,1) 算法，杜绝外部依赖报错。
✅ 严格 Listwise Deletion，剔除 0 值盲区，杜绝方差崩塌。
✅ 同步对比三大消融模型 (A, B, C)。
"""
import os
import json
import numpy as np
from scipy.stats import f as f_dist
from datetime import datetime
import sys

sys.path.append("/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE")
from config import PATHS

def compute_icc_2k1(matrix):
    """
    手动实现 ICC(2,1) 双向随机效应、绝对一致性。
    """
    n, k = matrix.shape
    if n < 2: return 0.0, 0.0, 0.0

    grand_mean = np.mean(matrix)
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)

    # Sum of Squares
    SST = np.sum((matrix - grand_mean)**2)
    SSR = k * np.sum((row_means - grand_mean)**2)
    SSC = n * np.sum((col_means - grand_mean)**2)
    SSE = SST - SSR - SSC

    # Mean Squares
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1)) if ((n - 1) * (k - 1)) > 0 else 1e-10

    # Calculate ICC(2,1)
    icc_num = MSR - MSE
    icc_den = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    icc = icc_num / icc_den if icc_den != 0 else 0.0

    # 95% CI calculation (McGraw & Wong 1996)
    F_j = MSC / MSE
    vn = (k - 1) * (n - 1)
    vd = n - 1
    v = vn * vd / (vn + vd) if (vn + vd) > 0 else 1e-10

    # Avoid division by zero in bounds
    if icc == 1.0 or MSR == 0:
        return max(0.0, float(icc)), 0.0, 1.0

    a = (k * icc) / (n * (1 - icc))
    b = 1 + (k * icc) / (1 - icc)
    v = (a * MSC + b * MSE)**2 / ((a * MSC)**2 / (k - 1) + (b * MSE)**2 / ((n - 1) * (k - 1)))
    
    F_l = f_dist.ppf(1 - 0.025, n - 1, v)
    F_u = f_dist.ppf(0.025, n - 1, v)

    L = (n * (MSR - F_l * MSE)) / (F_l * (k * MSC + (k * n - k - n) * MSE) + n * MSR)
    U = (n * (MSR - F_u * MSE)) / (F_u * (k * MSC + (k * n - k - n) * MSE) + n * MSR)

    # 限制物理范围 [0, 1]
    icc = max(0.0, min(1.0, float(icc)))
    L = max(0.0, min(1.0, float(L)))
    U = max(0.0, min(1.0, float(U)))
    
    return icc, L, U

def extract_valid_matrix(data, patient_list, level, metric, seg_key=None):
    """提取有效矩阵，剔除物理 0 值盲区"""
    matrix = []
    for pid in patient_list:
        try:
            if level == "global":
                a2c = float(data[pid]["A2C"]["global"].get(metric, 0.0))
                a3c = float(data[pid]["A3C"]["global"].get(metric, 0.0))
                a4c = float(data[pid]["A4C"]["global"].get(metric, 0.0))
            else:
                a2c = float(data[pid]["A2C"]["segments"].get(seg_key, {}).get(metric, 0.0))
                a3c = float(data[pid]["A3C"]["segments"].get(seg_key, {}).get(metric, 0.0))
                a4c = float(data[pid]["A4C"]["segments"].get(seg_key, {}).get(metric, 0.0))
            
            # 盲区保护：0 值转换为 NaN
            if a2c <= 1e-5: a2c = np.nan
            if a3c <= 1e-5: a3c = np.nan
            if a4c <= 1e-5: a4c = np.nan
            matrix.append([a2c, a3c, a4c])
        except KeyError:
            matrix.append([np.nan, np.nan, np.nan])
            
    matrix = np.array(matrix)
    valid_mask = ~np.isnan(matrix).any(axis=1)
    valid_matrix = matrix[valid_mask]
    return valid_matrix

def main():
    print("=" * 80)
    print("📊 跨视图 ICC(2,1) 一致性评估引擎")
    print("=" * 80)

    OUTPUT_DIR = os.path.join(PATHS["results_root"], "evaluation")
    
    data_paths = {
        "Method A (No TF)": os.path.join(OUTPUT_DIR, "baseline_A_no_tf.json"),
        "Method B (TF Single)": os.path.join(OUTPUT_DIR, "baseline_B_single_view.json"),
        "Method C (TF + Fused)": os.path.join(OUTPUT_DIR, "method_C_fused.json")
    }

    datasets = {}
    for name, path in data_paths.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                datasets[name] = json.load(f)

    if len(datasets) < 3:
        print("❌ 错误：必须跑完流水线生成三个 JSON 才能对比！")
        return

    common_patients = [pid for pid, views in datasets["Method C (TF + Fused)"].items() 
                       if all(v in views for v in ["A2C", "A3C", "A4C"])]
                       
    print(f"🔍 有效评估病例数：{len(common_patients)} 例")

    eval_metrics = [
        ("全局血容量 A", "global", "A", None),
        ("全局血流速度 β", "global", "beta", None),
        ("全局血流量 MBF", "global", "mbf", None),
        ("心尖帽血流量 Seg17", "segment", "mbf", "seg_17"),
    ]

    results = {}
    for metric_name, level, metric, seg_key in eval_metrics:
        results[metric_name] = {}
        for group_name, data in datasets.items():
            matrix = extract_valid_matrix(data, common_patients, level, metric, seg_key)
            icc, ci_l, ci_h = compute_icc_2k1(matrix)
            results[metric_name][group_name] = {"icc": icc, "ci_low": ci_l, "ci_high": ci_h, "n": len(matrix)}

    # 生成 Markdown
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(OUTPUT_DIR, f"ICC_Ablation_Report_{timestamp}.md")

    method_names = list(datasets.keys())
    md_content = f"""# 跨切面灌注参数一致性评估（ICC）
> **最大病例数**：{len(common_patients)} 例 (排除盲区的实际计算配对数见 N 值)
> **统计方法**：组内相关系数 ICC(2,1)，双向随机效应模型，绝对一致性

| 评估指标 | 有效配对数 (N) | Method A (No TF) | Method B (TF) | Method C (TF+Fused) | 相对提升(B至C) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for metric_name, res in results.items():
        row = f"| {metric_name} | {res[method_names[-1]]['n']} "
        for name in method_names:
            row += f"| {res[name]['icc']:.3f} ({res[name]['ci_low']:.2f}-{res[name]['ci_high']:.2f}) "
            
        base = res["Method B (TF Single)"]["icc"]
        ours = res["Method C (TF + Fused)"]["icc"]
        if base > 1e-4:
            improve = f"+{((ours - base) / base) * 100:.1f}%"
        else:
            improve = "N/A"
        row += f"| **{improve}** |\n"
        md_content += row

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✅ ICC 报告已生成: {md_path}")

if __name__ == "__main__":
    main()