# ShapeIoU 使用指南

## 📋 快速开始

### 1. 验证集成状态

```bash
# 快速检查 ShapeIoU 是否正确集成
python scripts/check_shapeiou_status.py
```

应该看到：
```
📈 检查结果: 13/13 通过 (100.0%)
🎉 所有检查通过！ShapeIoU 已完全集成并可以正常使用。
```

---

### 2. 基本使用

#### 2.1 使用默认的 CIoU（无需指定）
```bash
python scripts/train.py --cfg data.yaml --epochs 100
```

#### 2.2 使用 ShapeIoU
```bash
python scripts/train.py --cfg data.yaml --epochs 100 --iou-type ShapeIoU
```

#### 2.3 使用其他 IoU 类型
```bash
# GIoU
python scripts/train.py --cfg data.yaml --epochs 100 --iou-type GIoU

# DIoU
python scripts/train.py --cfg data.yaml --epochs 100 --iou-type DIoU

# 标准 IoU
python scripts/train.py --cfg data.yaml --epochs 100 --iou-type IoU
```

---

### 3. 完整训练示例

```bash
python scripts/train.py \
    --cfg datasets/coco8.yaml \
    --model yolo11n.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --iou-type ShapeIoU \
    --name shapeiou_experiment \
    --device 0 \
    --workers 8
```

---

### 4. 部分参数使用（现已修复）

✅ **现在可以安全使用部分参数，不会报错**：

```bash
# 只指定部分参数，其他使用默认值
python scripts/train.py --epochs 20 --name test_run

# 使用增强和加权数据加载器
python scripts/train.py --epochs 20 --augment --weighted-dataloader --name test_shapeiou

# 自定义 IoU 类型
python scripts/train.py --epochs 50 --iou-type ShapeIoU --name shapeiou_test
```

之前会报 `AttributeError: 'Namespace' object has no attribute 'data'` 的错误，现在已经修复。

---

## 🎯 训练时的输出

使用 ShapeIoU 训练时，会看到彩色高亮的配置信息：

```
================================================================================
🎯 YOLO11 Training Configuration
================================================================================

📁 Dataset & Model:
  Model:      yolo11n.pt
  Data:       coco8.yaml
  Task:       detect

🔧 Training Parameters:
  Epochs:     100
  Batch size: 16
  Image size: 640
  Device:     0

📊 Loss Configuration:
  IoU Type:   ShapeIoU (Shape-aware IoU)  ← 这里会显示你使用的 IoU 类型
  Box:        7.5
  Cls:        0.5
  DFL:        1.5

⚡ Optimizer:
  Optimizer:  SGD
  LR0:        0.01
  Momentum:   0.937
  Weight decay: 0.0005

================================================================================

Detection Loss initialized with IoU type: ShapeIoU  ← 确认损失函数使用 ShapeIoU

Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100      3.2G      0.856      1.234      0.567         32        640
  ...
```

---

## 🔍 不同 IoU 类型对比

### 支持的 IoU 类型

| IoU 类型 | 命令行参数 | 特点 | 适用场景 |
|---------|-----------|------|---------|
| **IoU** | `--iou-type IoU` | 标准交并比 | 基础检测 |
| **GIoU** | `--iou-type GIoU` | 考虑非重叠区域 | 改善无重叠惩罚 |
| **DIoU** | `--iou-type DIoU` | 考虑中心点距离 | 更快收敛 |
| **CIoU** | `--iou-type CIoU` (默认) | 考虑长宽比 | 综合性能好 |
| **ShapeIoU** | `--iou-type ShapeIoU` | 形状感知 | 对象形状敏感任务 |

### 实验对比建议

```bash
# 运行对比实验
python scripts/train.py --cfg data.yaml --epochs 100 --iou-type CIoU --name exp_ciou
python scripts/train.py --cfg data.yaml --epochs 100 --iou-type ShapeIoU --name exp_shapeiou

# 对比结果
python compare_experiments.py --exp1 exp_ciou --exp2 exp_shapeiou
```

---

## 🧪 测试和验证

### 运行完整集成测试
```bash
python tests/test_shapeiou_integration.py
```

预期输出：
```
🎯 ShapeIoU 集成验证测试
✅ ShapeIoU 函数测试通过！
✅ 损失函数集成测试通过！
✅ 命令行参数说明完成！
🎉 所有测试通过！ShapeIoU 已成功集成到 YOLO11 中！
```

### 快速状态检查
```bash
python scripts/check_shapeiou_status.py
```

预期输出：
```
📈 检查结果: 13/13 通过 (100.0%)
🎉 所有检查通过！ShapeIoU 已完全集成并可以正常使用。
```

---

## 🐛 故障排查

### 问题 1: AttributeError: 'Namespace' object has no attribute 'xxx'

**解决方案**: 已修复！`print_training_config()` 函数现在使用 `getattr()` 安全处理所有参数。

### 问题 2: 训练时看不到 IoU 类型提示

**检查**:
```bash
# 确保导入了配置打印函数
grep "print_training_config" train.py
```

如果没有，在 `scripts/train.py` 的 `main()` 函数开头添加：
```python
from ultralytics.utils.trainer_utils import print_training_config
print_training_config(args)
```

### 问题 3: ShapeIoU 计算结果异常

**检查**:
```bash
# 运行单元测试
python -c "
import torch
from ultralytics.utils.metrics import bbox_shape_iou

box1 = torch.tensor([[100., 100., 200., 200.]])
box2 = torch.tensor([[150., 150., 250., 250.]])
iou = bbox_shape_iou(box1, box2, xywh=False)
print(f'ShapeIoU: {iou.item():.4f}')
assert 0 <= iou.item() <= 1, 'IoU 应该在 [0, 1] 范围内'
print('✅ ShapeIoU 计算正常')
"
```

---

## 📊 性能优化建议

### 1. 根据数据集特点选择 IoU 类型

- **小目标较多**: 使用 `ShapeIoU` 或 `CIoU`
- **目标尺度变化大**: 使用 `CIoU`
- **需要快速收敛**: 使用 `DIoU`
- **基础检测**: 使用 `CIoU` (默认)

### 2. 调整损失权重

```bash
python scripts/train.py \
    --cfg data.yaml \
    --epochs 100 \
    --iou-type ShapeIoU \
    --box 7.5 \      # 边界框损失权重
    --cls 0.5 \      # 分类损失权重
    --dfl 1.5        # DFL 损失权重
```

### 3. 监控训练过程

```python
# 在训练过程中监控不同损失项
tensorboard --logdir runs/train
```

---

## 📚 相关文档

- [完整验证报告](SHAPEIOU_COMPLETE_VERIFICATION.md)
- [ShapeIoU 集成报告](SHAPEIOU_INTEGRATION_REPORT.md)
- [ShapeIoU 快速参考](SHAPEIOU_QUICK_REFERENCE.txt)
- [ShapeIoU 检查清单](SHAPEIOU_INTEGRATION_CHECKLIST.txt)

---

## ✅ 检查清单

使用 ShapeIoU 前，确保：

- [ ] 运行 `python scripts/check_shapeiou_status.py` 显示 100% 通过
- [ ] 运行 `python tests/test_shapeiou_integration.py` 全部测试通过
- [ ] 训练时能看到 IoU 类型的配置信息
- [ ] 可以使用 `--iou-type ShapeIoU` 参数
- [ ] 训练日志中显示 "Detection Loss initialized with IoU type: ShapeIoU"

---

## 🎉 总结

ShapeIoU 已完全集成到 YOLO11 中，你现在可以：

✅ 使用 `--iou-type ShapeIoU` 参数进行训练  
✅ 看到明确的 IoU 类型配置信息  
✅ 使用部分参数训练而不会报错  
✅ 与其他 IoU 类型轻松切换对比  
✅ 运行完整的测试验证集成状态  

**开始训练**:
```bash
python scripts/train.py --cfg data.yaml --epochs 100 --iou-type ShapeIoU --name my_experiment
```

祝训练顺利！🚀



