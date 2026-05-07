#!/usr/bin/env python3
"""
ShapeIoU 快速状态检查脚本

使用方法:
    python scripts/check_shapeiou_status.py
"""

import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """检查文件是否存在"""
    path = Path(filepath)
    status = "✅" if path.exists() else "❌"
    print(f"{status} {description}: {filepath}")
    return path.exists()


def check_function_exists(module_path, function_name):
    """检查函数是否存在"""
    try:
        module_name = module_path.replace('/', '.').replace('.py', '')
        exec(f"from {module_name} import {function_name}")
        print(f"✅ 函数 {function_name} 存在于 {module_path}")
        return True
    except Exception as e:
        print(f"❌ 函数 {function_name} 不存在或导入失败: {e}")
        return False


def check_shapeiou_integration():
    """完整的 ShapeIoU 集成检查"""
    print("="*80)
    print("🎯 ShapeIoU 集成状态检查")
    print("="*80)

    all_checks = []

    # 1. 检查关键文件
    print("\n📁 检查关键文件:")
    print("-"*80)
    all_checks.append(check_file_exists("scripts/shapeiou.py", "ShapeIoU 原始实现"))
    all_checks.append(check_file_exists("ultralytics/utils/metrics.py", "Metrics 模块"))
    all_checks.append(check_file_exists("ultralytics/utils/loss.py", "Loss 模块"))
    all_checks.append(check_file_exists("scripts/train.py", "训练脚本"))
    all_checks.append(check_file_exists("ultralytics/utils/trainer_utils.py", "训练工具"))
    all_checks.append(check_file_exists("tests/test_shapeiou_integration.py", "集成测试"))

    # 2. 检查函数实现
    print("\n🔧 检查函数实现:")
    print("-"*80)
    all_checks.append(check_function_exists("ultralytics.utils.metrics", "bbox_shape_iou"))
    all_checks.append(check_function_exists("ultralytics.utils.loss", "BboxLoss"))
    all_checks.append(check_function_exists("ultralytics.utils.trainer_utils", "print_training_config"))

    # 3. 检查 BboxLoss 是否支持 ShapeIoU
    print("\n📊 检查 BboxLoss 对 ShapeIoU 的支持:")
    print("-"*80)
    try:
        from ultralytics.utils.loss import BboxLoss
        import torch

        # 尝试创建 ShapeIoU 损失
        bbox_loss = BboxLoss(16, iou_type="ShapeIoU")
        print(f"✅ BboxLoss 支持 ShapeIoU (iou_type={bbox_loss.iou_type})")
        all_checks.append(True)
    except Exception as e:
        print(f"❌ BboxLoss 不支持 ShapeIoU: {e}")
        all_checks.append(False)

    # 4. 检查 YOLO 配置系统是否注册了 iou_type
    print("\n⚙️  检查 YOLO 配置系统:")
    print("-"*80)
    try:
        from ultralytics.utils import DEFAULT_CFG_DICT
        if 'iou_type' in DEFAULT_CFG_DICT:
            print(f"✅ iou_type 已在配置系统中注册 (默认值: {DEFAULT_CFG_DICT['iou_type']})")
            all_checks.append(True)
        else:
            print("❌ iou_type 未在配置系统中注册")
            print("   请检查 ultralytics/cfg/default.yaml")
            all_checks.append(False)
    except Exception as e:
        print(f"❌ 无法检查配置系统: {e}")
        all_checks.append(False)

    # 5. 检查命令行参数
    print("\n📋 检查命令行参数:")
    print("-"*80)
    try:
        with open("scripts/train.py", "r") as f:
            content = f.read()
            if "--iou-type" in content or "--iou_type" in content:
                print("✅ train.py 支持 --iou-type 参数")
                all_checks.append(True)
            else:
                print("❌ train.py 不支持 --iou-type 参数")
                all_checks.append(False)
    except Exception as e:
        print(f"❌ 无法检查 train.py: {e}")
        all_checks.append(False)

    # 5. 运行简单的功能测试
    print("\n🧪 运行功能测试:")
    print("-"*80)
    try:
        import torch
        from ultralytics.utils.metrics import bbox_shape_iou

        # 创建测试数据
        box1 = torch.tensor([[100., 100., 200., 200.]])
        box2 = torch.tensor([[150., 150., 250., 250.]])

        # 计算 ShapeIoU
        iou = bbox_shape_iou(box1, box2, xywh=False)

        if iou.numel() > 0 and 0 <= iou.item() <= 1:
            print(f"✅ bbox_shape_iou 计算正常 (IoU = {iou.item():.4f})")
            all_checks.append(True)
        else:
            print(f"❌ bbox_shape_iou 返回异常值: {iou.item()}")
            all_checks.append(False)
    except Exception as e:
        print(f"❌ bbox_shape_iou 测试失败: {e}")
        all_checks.append(False)

    # 6. 检查配置打印功能
    print("\n🖨️  检查配置打印功能:")
    print("-"*80)
    try:
        import argparse
        from ultralytics.utils.trainer_utils import print_training_config

        # 测试参数缺失的情况
        args = argparse.Namespace(iou_type='ShapeIoU', epochs=100)
        print_training_config(args)

        print("✅ print_training_config 能够处理缺失参数")
        all_checks.append(True)
    except Exception as e:
        print(f"❌ print_training_config 失败: {e}")
        all_checks.append(False)

    # 8. 测试配置验证（最关键的测试）
    print("\n🔐 测试配置验证:")
    print("-"*80)
    try:
        from ultralytics.cfg import get_cfg

        # 测试 ShapeIoU 参数能否通过配置验证
        overrides = {
            'epochs': 10,
            'batch': 4,
            'iou_type': 'ShapeIoU',
            'name': 'test',
        }
        cfg = get_cfg(overrides=overrides)

        if hasattr(cfg, 'iou_type') and cfg.iou_type == 'ShapeIoU':
            print("✅ 配置验证通过，iou_type 参数可以正常使用")
            print(f"   验证值: {cfg.iou_type}")
            all_checks.append(True)
        else:
            print("❌ 配置验证失败，iou_type 参数无效")
            all_checks.append(False)
    except Exception as e:
        print(f"❌ 配置验证测试失败: {e}")
        all_checks.append(False)

    # 总结
    print("\n" + "="*80)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total * 100) if total > 0 else 0

    print(f"📈 检查结果: {passed}/{total} 通过 ({percentage:.1f}%)")

    if all(all_checks):
        print("🎉 所有检查通过！ShapeIoU 已完全集成并可以正常使用。")
        print("\n使用方法:")
        print("  python scripts/train.py --cfg data.yaml --epochs 100 --iou-type ShapeIoU")
        return 0
    else:
        print("⚠️  部分检查未通过，请查看上述详细信息。")
        return 1

    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        exit_code = check_shapeiou_integration()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  检查被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 检查过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

