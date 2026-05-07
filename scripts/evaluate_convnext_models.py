# evaluate_convnext_models.py

import argparse
import os
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

# 从ultralytics导入必要的模块
from ultralytics import YOLO


# ------------------ 配置区 ------------------
# 请根据您的实际情况修改这些路径
# 实验结果的根目录
RUNS_DIR = './runs/train'
# 您要评估的实验名称模式，使用通配符 *
EXPERIMENT_PATTERN = 'baseline_yolo11_custom_ConvNeXt*'
# 修改配置区，支持多个实验模式
EXPERIMENT_PATTERNS = [
    'baseline_yolo11_custom_ConvNeXt*',
    'baseline_yolo11_test_lowlr*',
    'baseline_yolo11_custom_CBAM*'
]
# 数据集配置文件 .yaml 的路径
DATASET_YAML_PATH = '/home/user/PROJECT/pp/NEU-DET_YOLO_state_qmh/NEU-DET.yaml'
# 最终报告保存的文件名
OUTPUT_CSV_FILE = 'convnext_models_evaluation_summary.csv'


# -------------------------------------------

def get_model_profile(model_path, imgsz=640):
    """
    加载模型并计算其参数量和GFLOPs（优先使用 thop，失败则返回 N/A）。
    当可训练参数为0时，回退使用总参数量。
    """
    try:
        model = YOLO(model_path).model.eval()

        # 统计参数量：同时计算总参数与可训练参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))

        # 若可训练参数为0，回退为总参数
        chosen_params = trainable_params if trainable_params > 0 else total_params
        if trainable_params == 0 and total_params > 0:
            print("  [!] Detected all params frozen (requires_grad=False). Using total params instead.")

        params_m = round(chosen_params / 1e6, 2)

        # 计算GFLOPs（thop优先）
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)

        gflops = 'N/A'
        try:
            from thop import profile as thop_profile
            flops, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
            gflops = round(flops / 1e9, 2)
        except Exception as e:
            print(f"  [!] GFLOPs profiling unavailable: {e}")

        return params_m, gflops

    except Exception as e:
        print(f"  [!] Error profiling model {model_path}: {e}")
        return 'N/A', 'N/A'


def evaluate_experiment(exp_path: Path, dataset_yaml: str):
    """
    对单个实验目录进行完整的评估。

    Args:
        exp_path (Path): 实验目录的路径对象。
        dataset_yaml (str): 数据集配置文件的路径。

    Returns:
        dict: 包含该模型所有信息的字典，如果关键文件不存在则返回None。
    """
    experiment_name = exp_path.name
    print(f"--- Processing experiment: {experiment_name} ---")

    # 1. 定位关键文件
    model_path = exp_path / 'weights' / 'best.pt'
    args_path = exp_path / 'args.yaml'

    if not model_path.exists():
        print(f"  [!] Warning: 'best.pt' not found in {exp_path}. Skipping.")
        return None
    if not args_path.exists():
        print(f"  [!] Warning: 'args.yaml' not found in {exp_path}. Hyperparameters will be missing.")
        args_data = {}
    else:
        with open(args_path, 'r') as f:
            args_data = yaml.safe_load(f)

    # 2. 提取超参数
    imgsz = args_data.get('imgsz', 640)
    epochs = args_data.get('epochs', 'N/A')
    lr0 = args_data.get('lr0', 'N/A')
    optimizer = args_data.get('optimizer', 'N/A')
    batch_size = args_data.get('batch', 'N/A')

    # 3. 检查模型架构 (参数量, GFLOPs)
    print("  [*] Profiling model architecture...")
    params_m, gflops = get_model_profile(str(model_path), imgsz)
    print(f"  [*] Profiled: Params={params_m}M, GFLOPs={gflops}")

    # 4. 在数据集上进行验证
    print("  [*] Running validation on NEU-DET dataset...")
    try:
        # 加载模型用于验证
        model = YOLO(str(model_path))

        metrics = model.val(
            data=dataset_yaml,
            imgsz=imgsz,
            batch=batch_size,
            split='test',
            verbose=False
        )

        # 提取核心指标（使用平均P/R，避免索引问题）
        map50_95 = round(metrics.box.map, 4)   # mAP50-95
        map50 = round(metrics.box.map50, 4)    # mAP50
        precision = round(metrics.box.mp, 4)   # mean Precision
        recall = round(metrics.box.mr, 4)      # mean Recall

        print(f"  [*] Validation complete: mAP50-95={map50_95}, mAP50={map50}")

    except Exception as e:
        print(f"  [!] Error during validation for {experiment_name}: {e}")
        map50_95, map50, precision, recall = -1, -1, -1, -1

    # 5. 汇总结果
    result_dict = {
        'Experiment': experiment_name,
        'Params (M)': params_m if params_m != 'N/A' else -1,
        'GFLOPs': gflops if gflops != 'N/A' else -1,
        'mAP50-95': map50_95,
        'mAP50': map50,
        'Precision': precision,
        'Recall': recall,
        'Epochs': epochs,
        'ImgSize': imgsz,
        'BatchSize': batch_size,
        'Optimizer': optimizer,
        'Initial LR': lr0,
    }
    return result_dict


def main():
    """
    主函数，遍历所有实验并生成报告。
    """
    print("Starting evaluation of custom YOLO models...")

    # 检查数据集配置文件是否存在
    if not Path(DATASET_YAML_PATH).exists():
        print(f"[X] FATAL ERROR: Dataset config file not found at '{DATASET_YAML_PATH}'")
        print("Please update the DATASET_YAML_PATH variable in the script.")
        return

    # 查找所有匹配的实验目录
    base_path = Path(RUNS_DIR)

    experiment_paths = []
    for pattern in EXPERIMENT_PATTERNS:
        experiment_paths.extend(base_path.glob(pattern))

    experiment_paths = sorted(list(set(experiment_paths)))  # 去重并排序

    if not experiment_paths:
        print(f"[X] No experiment directories found matching pattern '{EXPERIMENT_PATTERN}' in '{RUNS_DIR}'")
        return

    print(f"Found {len(experiment_paths)} experiments to evaluate.")

    all_results = []
    # 使用tqdm创建进度条
    for exp_path in tqdm(experiment_paths, desc="Evaluating Experiments"):
        result = evaluate_experiment(exp_path, DATASET_YAML_PATH)
        if result:
            all_results.append(result)

    if not all_results:
        print("Evaluation finished, but no valid results were collected.")
        return

    # 7. 创建并展示最终的表格
    print("\n\n" + "=" * 20 + " FINAL EVALUATION SUMMARY " + "=" * 20)
    df = pd.DataFrame(all_results)

    df_display = df.copy()
    numeric_cols = ['Params (M)', 'GFLOPs', 'mAP50-95', 'mAP50', 'Precision', 'Recall']
    # 优化列的顺序
    column_order = [
        'Experiment', 'Params (M)', 'GFLOPs', 'mAP50-95', 'mAP50',
        'Precision', 'Recall', 'Epochs', 'ImgSize', 'BatchSize', 'Optimizer', 'Initial LR'
    ]
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: 'N/A' if x == -1 else x)
    # 排序：只对有效结果进行排序（mAP50-95 >= 0）
    valid_results = df[df['mAP50-95'] >= 0]
    failed_results = df[df['mAP50-95'] < 0]


    if not valid_results.empty:
        valid_sorted = valid_results.sort_values(by='mAP50-95', ascending=False).reset_index(drop=True)
        if not failed_results.empty:
            df_final = pd.concat([valid_sorted, failed_results], ignore_index=True)
        else:
            df_final = valid_sorted
    else:
        df_final = failed_results

    # 创建显示用的DataFrame
    df_display_final = df_final.copy()
    for col in numeric_cols:
        if col in df_display_final.columns:
            df_display_final[col] = df_display_final[col].apply(lambda x: 'N/A' if x == -1 else x)

    column_order = [
        'Experiment', 'Params (M)', 'GFLOPs', 'mAP50-95', 'mAP50',
        'Precision', 'Recall', 'Epochs', 'ImgSize', 'BatchSize', 'Optimizer', 'Initial LR'
    ]
    df_display_final = df_display_final[column_order]

    print(df_display_final.to_string())

    # 8. 保存到CSV文件
    try:
        df_display.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\n[✔] Summary successfully saved to '{OUTPUT_CSV_FILE}'")

        valid_count = len(valid_results)
        failed_count = len(failed_results)
        print(f"\n[ℹ] Summary: {valid_count} successful evaluations, {failed_count} failed (missing custom modules)")

    except Exception as e:
        print(f"\n[!] Error saving summary to CSV: {e}")


if __name__ == '__main__':
    main()
