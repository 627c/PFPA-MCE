# PFPA-MCE: Pixel-Level Quantitative Microcirculation Perfusion Analysis

## 📌 Open Source Notice for Blind Review
To comply with double-blind peer review policies and protect unpublished contributions, the core algorithmic implementations of **MRA-Gate** (in ADR-Net) and **SL-PLMB** (fusion algorithm) have been temporarily withheld. 

All other components, including baseline models, STFT filtering, Wei's kinetic modeling, AHA 17-segment mapping, and evaluation scripts are fully open-sourced to demonstrate the clinical pipeline. The complete source code will be released upon paper acceptance.

## 📖 Overview
Myocardial Contrast Echocardiography (MCE) perfusion analysis suffers from large cross-view quantitative variations and the lack of ground-truth labels. We propose an **unsupervised cross-view perfusion consistency enhancement pipeline**, consisting of:
1. **ADR-Net**: A segmentation network tailored for acoustic dropout artifacts via Multi-scale Receptive Attention Gate (MRA-Gate).
2. **Time-Frequency Purification**: STFT-based signal filtering coupled with Wei's kinetic model for robust perfusion fitting.
3. **SL-PLMB**: A Subsegmental-Level Pixel-Level Memory Bank for unsupervised cross-view feature fusion, drastically reducing perfusion variance across standard views (A2C/A3C/A4C).

## 🗂️ Repository Structure
```text
PFPA-MCE/
├── SL-PLMB_Core_Pipeline/                   # Core method of the paper: Full unsupervised consistency enhancement pipeline
│   ├── config/                              # Global configuration and path settings
│   ├── models/                              # Segmentation models (core module withheld)
│   │   ├── net_seg_only.py                  # ADR-Net segmentation network
│   │   └── baselines.py                     # 7 classical segmentation baseline models
│   ├── core/                                # Core algorithm modules
│   │   ├── tac.py                           # TAC signal extraction and STFT time-frequency filtering
│   │   ├── perfusion.py                     # Wei's kinetic model perfusion parameter fitting
│   │   ├── anatomy.py                       # AHA 17-segment myocardial standard division
│   │   └── fusion.py                        # SL-PLMB cross-view fusion algorithm (core withheld)
│   ├── data/                                # Data preprocessing tools
│   │   ├── convert_masks.py                 # Mask format standardization
│   │   ├── split_seg_train_val.py           # Segmentation dataset train/val split
│   │   ├── extract_triplets.py              # Video key triplet frame extraction
│   │   └── generate_pseudo_labels.py        # Interactive pseudo-label generation tool
│   ├── utils/                               # Utility functions and metric calculation
│   │   ├── helpers.py                       # General helper functions
│   │   └── metrics.py                       # Full metrics: DSC, HD95, ICC(2,1), etc.
│   ├── evaluation/                          # Evaluation and visualization scripts
│   │   ├── evaluate_seg_baselines.py        # Segmentation full metric evaluation
│   │   ├── run_seg_ablation.py              # Segmentation module ablation experiment automation
│   │   ├── calculate_complexity.py          # Model parameter and computation statistics
│   │   ├── calculate_all_icc.py             # Cross-view consistency ICC evaluation
│   │   ├── generate_all_visualizations.py   # Academic visualization generation
│   │   └── plot_cv_visualization.py         # CV comparison visualization
│   └── run_final_clinical_pipeline.py       # End-to-end clinical pipeline main entry
├── Supervised_Extension/                    # Exploratory extension: Supervised perfusion regression baseline
│   ├── models/
│   │   └── net.py                           # Multi-task ADR-Net (segmentation + perfusion regression)
│   ├── data/
│   │   └── dataset.py                       # Spatiotemporal triplet dataset loader
│   ├── train.py                             # Multi-task joint training script
│   ├── evaluate.py                          # Perfusion regression accuracy evaluation
│   ├── split_train_val.py                   # Multi-task dataset train/val split
│   ├── run_ablation_studies.py              # Full pipeline ablation experiment automation
│   └── README.md                            # Detailed description of the extension scheme
├── README.md                                # English documentation (this file)
└── README_zh.md                             # Chinese documentation

```

## 📊 Quantitative Results (Real Data)

### 1. Robust Myocardial Segmentation

Evaluated on static MCE frames with expert annotations. ADR-Net achieves optimal boundary accuracy with extremely low parameters.

| Model | DSC ↑ | IoU ↑ | Sens ↑ | Spec ↑ | HD95 (px) ↓ | ASSD (px) ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Attention-UNet | 0.7517 | 0.6101 | 0.8768 | 0.9876 | 94.07 | 8.82 |
| UNet | 0.7701 | 0.6339 | 0.8512 | 0.9903 | 27.07 | 4.44 |
| UNet++ | 0.7737 | 0.6387 | 0.8647 | 0.9899 | 43.18 | 5.48 |
| nnU-Net | 0.7869 | 0.6556 | 0.8921 | 0.9897 | 28.88 | 4.45 |
| Swin-UNet | 0.7529 | 0.6111 | 0.8445 | 0.9892 | 21.95 | 4.17 |
| MobileUNet | 0.7562 | 0.6159 | 0.8227 | 0.9905 | 26.78 | 4.77 |
| **ADR-Net (Ours)** | **0.7857** | **0.6492** | **0.8782** | **0.9937** | **20.35** | **3.80** |

### 2. Unsupervised Cross-View Consistency

Evaluated using the Coefficient of Variation (CV) across multi-view sequences. Lower CV indicates higher clinical reproducibility.

| Evaluation Metric | Method A (No TF) Mean CV | Method B (TF Single) Mean CV | Method C (TF + Fused) Mean CV | Absolute Decrease (B→C) | Relative Decrease Rate (B→C) | 
| :--- | :---: | :---: | :---: | :---: | :---: | 
| Global Blood Volume A | 34.4% | 34.4% | 11.4% | 23.0% | 67.0% | 
| Global Blood Flow Velocity β | 54.2% | 54.4% | 19.0% | 35.4% | 65.0% | 
| Global Myocardial Blood Flow (MBF) | 61.8% | 62.0% | 19.7% | 42.3% | 68.2% | 
| Apical Cap Blood Flow (Seg17) | 79.4% | 79.5% | 22.1% | 57.4% | 72.2% |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/627c/PFPA-MCE.git

# Navigate to the core pipeline
cd PFPA-MCE/Core_Clinical_Pipeline

# Run the end-to-end evaluation (Ensure your paths in config.py are set)
python run_final_clinical_pipeline.py

```

*(Note: Due to ethical restrictions, clinical video sequences are not included. Please use your own `.avi` files placed in the `raw_data/` directory).*

## ✉️ Contact

Anqi Liu (1104220109@stu.jiangnan.edu.cn) - Jiangnan University
```
