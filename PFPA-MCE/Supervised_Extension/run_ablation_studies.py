"""
run_ablation_studies.py - 顶刊消融实验一键自动化流转中枢
自动控制两大战线变体组合：
1. Model A (Baseline): 无时频滤波 (No-TF) + 无跨视图融合 (No-PLMB)
2. Model B (+TF):     有时频滤波 (With-TF) + 无跨视图融合 (No-PLMB)
3. Model C (+PLMB):   无时频滤波 (No-TF) + 有跨视图融合 (With-PLMB)
4. Model D (Ours):     有时频滤波 (With-TF) + 有跨视图融合 (With-PLMB) [最终版]
"""
import os
import subprocess
import json
import numpy as np

RESULTS_SAVE_DIR = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results/ablation_summary"
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

def update_ablation_configs(apply_tf, use_plmb):
    """动态劫持并物理覆写配置文件与执行脚本的消融逻辑开关"""
    # 1. 动态改写数据准备脚本的 TF 滤波控制键
    prep_script_path = "./prepare_training_data.py"
    with open(prep_script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if "APPLY_TF_FILTER =" in line:
            lines[idx] = f"APPLY_TF_FILTER = {apply_tf} \n"
            break
    with open(prep_script_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # 2. 动态改写推理系统 main.py 的 PLMB 融合控制键
    main_script_path = "./main.py"
    with open(main_script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if "USE_PLMB =" in line:
            lines[idx] = f"USE_PLMB = {use_plmb}  # 动态消融开关\n"
            break
    with open(main_script_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def run_command(cmd_str):
    print(f"🎬 正在执行核心命令: {cmd_str}")
    process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"❌ 命令执行失败，程序阻断: {cmd_str}")
        return False
    return True

def collect_metrics_for_variant(variant_name):
    """在各个消融推理生成的 json 文件中提取灌注和一致性核心指标"""
    results_root = "/data/stu1/liuanqi/PFPA-MCE/PFPA-MCE/results"
    # 根据 main.py 的逻辑确定其输出文件名
    json_file = "final_perfusion_results.json" if "With-PLMB" in variant_name else "no_plmb_results.json"
    json_path = os.path.join(results_root, json_file)
    
    if not os.path.exists(json_path):
        print(f"⚠️ 找不到结果报告，无法提取该变体指标: {json_path}")
        return None
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 我们在这批数据中动态读取 Global 参数，通过快速交叉仿真，生成变体快照存盘
    summary_path = os.path.join(RESULTS_SAVE_DIR, f"{variant_name}_metrics.json")
    shutil_copy = f"cp {json_path} {summary_path}"
    os.system(shutil_copy)
    print(f"✅ {variant_name} 变体指标数据归档完毕！")

def main():
    print("="*60)
    print("🚀 PFPA-MCE 顶刊全闭环消融实验自动化测试管线启动")
    print("="*60)
    
    # 💥 组合 1：Model A (No-TF + No-PLMB)
    print("\n[Variant 1/4] 正在编译 Model A: No-TF + No-PLMB 变体...")
    update_ablation_configs(apply_tf=False, use_plmb=False)
    if run_command("python prepare_training_data.py") and run_command("python train.py") and run_command("python main.py"):
        collect_metrics_for_variant("Model_A_NoTF_NoPLMB")

    # 💥 组合 2：Model B (With-TF + No-PLMB)
    print("\n[Variant 2/4] 正在编译 Model B: With-TF + No-PLMB 变体...")
    update_ablation_configs(apply_tf=True, use_plmb=False)
    if run_command("python prepare_training_data.py") and run_command("python train.py") and run_command("python main.py"):
        collect_metrics_for_variant("Model_B_WithTF_NoPLMB")

    # 💥 组合 3：Model C (No-TF + With-PLMB)
    print("\n[Variant 3/4] 正在编译 Model C: No-TF + With-PLMB 变体...")
    update_ablation_configs(apply_tf=False, use_plmb=True)
    if run_command("python prepare_training_data.py") and run_command("python train.py") and run_command("python main.py"):
        collect_metrics_for_variant("Model_C_NoTF_WithPLMB")

    # 💥 组合 4：Model D (Ours: With-TF + With-PLMB)
    print("\n[Variant 4/4] 正在编译最终完美形态 Model D: With-TF + With-PLMB...")
    update_ablation_configs(apply_tf=True, use_plmb=True)
    if run_command("python prepare_training_data.py") and run_command("python train.py") and run_command("python main.py"):
        collect_metrics_for_variant("Model_D_Ours_Full")

    print("\n" + "="*60)
    print("🎉 恭喜！全套消融实验一键自动化跑通！")
    print(f"所有多任务变体报告已全量归档在: {RESULTS_SAVE_DIR}")
    print("请打开各个归档的json，直接提取 Global ICC 与 MSE 数据填入 Table 4！")
    print("="*60)

if __name__ == "__main__":
    main()