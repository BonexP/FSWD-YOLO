# 批量串行训练使用指南

## 📋 概述

`scripts/run_yolo_batch.sh` 脚本可以让你在**后台串行执行多个训练任务**，完美结合了：
- ✅ 后台运行（可关闭终端）
- ✅ 串行执行（避免 GPU 内存不足）
- ✅ 自动化（无需人工干预）

## 🚀 基本用法

```bash
./scripts/run_yolo_batch.sh <batch_name> <task1_args> -- <task2_args> [-- <task3_args> ...]
```

**重要规则：**
- 使用 `--` 分隔不同的训练任务
- 每个任务必须包含 `--name` 参数
- 任务会按顺序串行执行

## 📝 使用示例

### 示例 1：对比两个学习率

```bash
./scripts/run_yolo_batch.sh lr_compare \
    --name lr0.001 --lr0 0.001 --epochs 100 -- \
    --name lr0.0005 --lr0 0.0005 --epochs 100
```

**执行流程：**
1. 先训练 `lr0.001` (100 epochs)
2. 完成后自动开始训练 `lr0.0005` (100 epochs)
3. 所有日志保存到 `lr_compare.log`

### 示例 2：对比数据增强效果

```bash
./scripts/run_yolo_batch.sh augment_compare \
    --name baseline --epochs 200 -- \
    --name with_augment --augment --epochs 200
```

### 示例 3：对比不同 Batch Size

```bash
./scripts/run_yolo_batch.sh batch_size_study \
    --name bs16 --batch-size 16 --epochs 150 -- \
    --name bs32 --batch-size 32 --epochs 150 -- \
    --name bs64 --batch-size 64 --epochs 150
```

### 示例 4：完整的消融实验

```bash
./scripts/run_yolo_batch.sh ablation_study \
    --name baseline --epochs 200 -- \
    --name with_mixup --augment --mosaic 0.0 --mixup 0.3 --epochs 200 -- \
    --name with_mosaic --augment --mosaic 1.0 --mixup 0.0 --epochs 200 -- \
    --name full_augment --augment --epochs 200
```

### 示例 5：优化器对比

```bash
./scripts/run_yolo_batch.sh optimizer_compare \
    --name adam_001 --optimizer Adam --lr0 0.001 --epochs 150 -- \
    --name adam_0005 --optimizer Adam --lr0 0.0005 --epochs 150 -- \
    --name sgd_001 --optimizer SGD --lr0 0.001 --epochs 150
```

## 🔍 监控训练进度

### 查看实时日志
```bash
tail -f lr_compare.log
```

### 查看当前运行的训练任务
```bash
ps aux | grep train.py
```

### 查看 GPU 使用情况
```bash
watch -n 1 nvidia-smi
```

### 停止批量训练
```bash
# 方法1：使用脚本输出的 PID
kill <PID>

# 方法2：停止所有 train.py 进程
pkill -f "python scripts/train.py"
```

## ⚠️ 注意事项

1. **每个任务必须有 --name 参数**
   ```bash
   # ❌ 错误（缺少 --name）
   ./scripts/run_yolo_batch.sh test --epochs 100 -- --lr0 0.001
   
   # ✅ 正确
   ./scripts/run_yolo_batch.sh test --name exp1 --epochs 100 -- --name exp2 --lr0 0.001
   ```

2. **分隔符必须独立使用**
   ```bash
   # ❌ 错误（-- 连在参数后面）
   ./scripts/run_yolo_batch.sh test --name exp1 --epochs 100-- --name exp2
   
   # ✅ 正确（-- 前后有空格）
   ./scripts/run_yolo_batch.sh test --name exp1 --epochs 100 -- --name exp2
   ```

3. **确保 GPU 内存足够**
   - 虽然是串行执行，但要确保单个任务不会耗尽 GPU 内存
   - 如果内存不足，降低 `--batch-size`

4. **日志文件管理**
   - 所有任务共享同一个日志文件
   - 日志文件名为 `<batch_name>.log`
   - 建议定期清理旧日志

## 🆚 三种脚本对比

| 特性 | `scripts/run_yolo.sh` | `scripts/run_yolo_batch.sh` | 直接调用 `python scripts/train.py` |
|------|---------------|---------------------|----------------------------|
| 后台运行 | ✅ | ✅ | ❌ (需要手动 nohup) |
| 串行执行多任务 | ❌ | ✅ | ✅ (使用 &&) |
| 可关闭终端 | ✅ | ✅ | ❌ (除非用 nohup) |
| 适用场景 | 单次训练 | 批量对比实验 | 快速测试 |

## 💡 最佳实践

1. **规划实验名称**
   - 批量名称应体现实验主题：`lr_compare`, `augment_study`
   - 任务名称应体现参数差异：`lr0.001`, `bs16`, `with_augment`

2. **合理安排训练时长**
   - 对比实验建议先用较少 epochs 测试
   - 确认配置正确后再进行完整训练

3. **监控首个任务**
   ```bash
   # 启动批量训练后，立即查看日志
   ./scripts/run_yolo_batch.sh my_exp ... &
   tail -f my_exp.log
   ```

4. **保存实验记录**
   ```bash
   # 在日志中记录实验目的
   echo "实验目的：对比学习率 0.001 vs 0.0005" >> lr_compare.log
   ./scripts/run_yolo_batch.sh lr_compare ...
   ```

## 🐛 故障排查

### 问题：脚本立即退出
```bash
# 检查日志文件
cat <batch_name>.log

# 常见原因：
# - 数据集路径错误
# - 模型配置文件不存在
# - Python 环境问题
```

### 问题：第二个任务没有开始
```bash
# 检查第一个任务是否成功完成
grep "Training complete" <batch_name>.log

# 如果第一个任务失败，整个批量训练会终止
```

### 问题：无法找到日志文件
```bash
# 日志文件在脚本执行目录
ls -la *.log

# 查看最近修改的日志
ls -lt *.log | head
```

## 📊 结果分析

训练完成后，结果保存在：
```
runs/train/
├── lr0.001/          # 第一个任务结果
│   ├── weights/
│   ├── results.csv
│   └── ...
└── lr0.0005/         # 第二个任务结果
    ├── weights/
    ├── results.csv
    └── ...
```

使用以下命令对比结果：
```bash
# 对比最终指标
grep "mAP" runs/train/*/results.csv

# 使用 TensorBoard 可视化
tensorboard --logdir runs/train
```


