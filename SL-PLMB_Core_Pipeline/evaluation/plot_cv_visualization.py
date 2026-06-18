import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append("/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE")
from config import PATHS

# AHA 17-segment bullseye plot configuration
AHA_RINGS = [
    (0.7, 1.0, 6, 0),    # Basal segments: 1-6
    (0.4, 0.7, 6, 6),    # Mid segments: 7-12
    (0.1, 0.4, 4, 12),   # Apical segments: 13-16
    (0.0, 0.1, 1, 16)    # Apical cap: 17
]

APICAL_ANGLES = [0.0, 3*np.pi/2, np.pi, np.pi/2]  # Angles of 4 apical segments

def plot_bullseye(values, ax, title, vmin=None, vmax=None, cmap="jet"):
    """Plot AHA 17-segment bullseye plot"""
    if vmin is None:
        vmin = np.min(values[values > 1e-5]) if np.any(values > 1e-5) else 0
    if vmax is None:
        vmax = np.max(values)
    
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Draw each ring
    for r_inner, r_outer, n_segs, start_idx in AHA_RINGS:
        if start_idx == 16:
            # Apical cap, draw circle separately
            theta = np.linspace(0, 2*np.pi, 100)
            r = np.full_like(theta, r_outer)
            ax.fill_between(theta, 0, r_outer, color=plt.cm.jet(norm(values[start_idx])), edgecolor="white", linewidth=1.2)
            ax.text(0, 0, "17", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        elif start_idx == 12:
            # Apical segments, 4 segments
            for i in range(n_segs):
                seg_idx = start_idx + i
                theta_mid = APICAL_ANGLES[i]
                theta_start = theta_mid - np.pi/4
                theta_end = theta_mid + np.pi/4
                theta = np.linspace(theta_start, theta_end, 50)
                ax.fill_between(theta, r_inner, r_outer, color=plt.cm.jet(norm(values[seg_idx])), edgecolor="white", linewidth=1.2)
                ax.text(theta_mid, (r_inner + r_outer)/2, str(seg_idx+1), ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        else:
            # Basal and mid segments, 6 segments
            theta_edges = np.linspace(0, 2*np.pi, n_segs + 1)
            for i in range(n_segs):
                seg_idx = start_idx + i
                theta = np.linspace(theta_edges[i], theta_edges[i+1], 50)
                ax.fill_between(theta, r_inner, r_outer, color=plt.cm.jet(norm(values[seg_idx])), edgecolor="white", linewidth=1.2)
                theta_mid = (theta_edges[i] + theta_edges[i+1]) / 2
                ax.text(theta_mid, (r_inner + r_outer)/2, str(seg_idx+1), ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    
    ax.set_title(title, fontsize=12, fontweight="bold", y=1.05)

def main():
    FIG_DIR = os.path.join(PATHS["results_root"], "figures_final", "cv_ablation")
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Load segment CV data
    seg_cv_results = np.load(os.path.join(FIG_DIR, "seg_cv_results.npy"), allow_pickle=True).item()
    method_names = list(seg_cv_results.keys())
    cv_b = seg_cv_results[method_names[1]] * 100  # Convert to percentage
    cv_c = seg_cv_results[method_names[2]] * 100
    cv_drop = cv_b - cv_c  # Reduction value

    # ===================== Figure 1: 17-segment CV comparison bar chart =====================
    print(" Plotting 17-segment CV comparison bar chart...")
    x = np.arange(17)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    rects1 = ax.bar(x - width/2, cv_b, width, label="Method B (TF Single)", color="#90CAF9", edgecolor="white")
    rects2 = ax.bar(x + width/2, cv_c, width, label="Method C (TF + Fused)", color="#EF5350", edgecolor="white")
    
    ax.set_xlabel("AHA 17 Myocardial Segments", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-view Coefficient of Variation CV (%)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-view CV Comparison of 17 Segments Before and After Fusion", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seg{i+1}" for i in range(17)])
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "seg_cv_barplot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ===================== Figure 2: CV bullseye plot (pre-fusion + post-fusion + reduction) =====================
    print(" Plotting CV bullseye plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection="polar"), dpi=300)
    
    # Unified color scale
    vmin = 0
    vmax = max(np.max(cv_b), np.max(cv_c))
    
    plot_bullseye(cv_b, axes[0], "Pre-fusion CV (%)", vmin=vmin, vmax=vmax)
    plot_bullseye(cv_c, axes[1], "Post-fusion CV (%)", vmin=vmin, vmax=vmax)
    plot_bullseye(cv_drop, axes[2], "CV Reduction (%)", vmin=0, vmax=np.max(cv_drop))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:2], fraction=0.02, pad=0.05)
    cbar.set_label("CV (%)", fontsize=10, fontweight="bold")
    
    sm2 = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=np.max(cv_drop)))
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=axes[2], fraction=0.02, pad=0.05)
    cbar2.set_label("CV Drop (%)", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cv_bullseye.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f" Visualization results saved to: {FIG_DIR}")
    print("   - seg_cv_barplot.png: 17-segment CV comparison bar chart")
    print("   - cv_bullseye.png: CV bullseye plot (pre-fusion / post-fusion / reduction)")

if __name__ == "__main__":
    main()