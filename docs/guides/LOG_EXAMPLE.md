# 日志输出示例

## 📄 完整的日志文件示例

假设运行命令：
```bash
./scripts/run_yolo_batch_v2.sh lr_compare \
    --name lr_high --lr0 0.001 --epochs 5 -- \
    --name lr_low --lr0 0.0005 --epochs 5
```

生成的日志文件 `lr_compare.log` 内容如下：

---

```
╔══════════════════════════════════════════════════════════════╗
║          批量训练开始: lr_compare
║          开始时间: 2025-11-12 14:30:26
║          任务总数: 2
╚══════════════════════════════════════════════════════════════╝


══════════════════════════════════════════════════════════════
🚀 [任务 1/2] 开始: lr_high
开始时间: 2025-11-12 14:30:26
任务参数: python scripts/train.py --name lr_high --lr0 0.001 --epochs 5
══════════════════════════════════════════════════════════════

YAML 文件正文如下：
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLO11 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)

Ultralytics YOLO11s summary: 295 layers, 9,458,752 parameters, 9,458,736 gradients, 21.7 GFLOPs

Transferred 475/475 items from pretrained weights
optimizer: Adam(lr=0.001, momentum=0.937) with parameter groups 88 weight(decay=0.0005), 102 weight(decay=0.0), 101 bias(decay=0.0)
train: Scanning /home/user/PROJECT/FSWD/FSW-MERGE/train/labels... 1200 images, 0 backgrounds, 0 corrupt: 100%|██████████| 1200/1200 [00:01<00:00, 823.45it/s]
train: New cache created: /home/user/PROJECT/FSWD/FSW-MERGE/train/labels.cache
val: Scanning /home/user/PROJECT/FSWD/FSW-MERGE/valid/labels... 300 images, 0 backgrounds, 0 corrupt: 100%|██████████| 300/300 [00:00<00:00, 876.32it/s]
val: New cache created: /home/user/PROJECT/FSWD/FSW-MERGE/valid/labels.cache

Plotting labels to runs/train/lr_high/labels.jpg...
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/train/lr_high
Starting training for 5 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/5      3.45G      1.234      0.567      1.123         45        640: 100%|██████████| 75/75 [00:25<00:00,  2.95it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.12it/s]
                   all        300        450      0.654      0.723      0.701      0.456

        2/5      3.45G      0.987      0.432      0.956         45        640: 100%|██████████| 75/75 [00:24<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.23it/s]
                   all        300        450      0.712      0.765      0.748      0.512

        3/5      3.45G      0.856      0.378      0.834         45        640: 100%|██████████| 75/75 [00:24<00:00,  3.08it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.18it/s]
                   all        300        450      0.743      0.789      0.776      0.545

        4/5      3.45G      0.765      0.334      0.745         45        640: 100%|██████████| 75/75 [00:24<00:00,  3.05it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.15it/s]
                   all        300        450      0.768      0.805      0.795      0.568

        5/5      3.45G      0.698      0.301      0.678         45        640: 100%|██████████| 75/75 [00:24<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.20it/s]
                   all        300        450      0.785      0.818      0.809      0.583

5 epochs completed in 0.035 hours.
Optimizer stripped from runs/train/lr_high/weights/last.pt, 19.2MB
Optimizer stripped from runs/train/lr_high/weights/best.pt, 19.2MB

Validating runs/train/lr_high/weights/best.pt...
Ultralytics YOLO11s summary: 295 layers, 9,458,752 parameters, 0 gradients, 21.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:05<00:00,  3.45it/s]
                   all        300        450      0.785      0.818      0.809      0.583
Speed: 0.4ms preprocess, 2.8ms inference, 0.0ms loss, 1.2ms postprocess per image
Results saved to runs/train/lr_high
Training complete. Results saved to: runs/train/lr_high

══════════════════════════════════════════════════════════════
✅ [任务 1/2] 完成: lr_high
结束时间: 2025-11-12 14:33:45
耗时: 199 秒 (3 分钟)
══════════════════════════════════════════════════════════════


══════════════════════════════════════════════════════════════
🚀 [任务 2/2] 开始: lr_low
开始时间: 2025-11-12 14:33:45
任务参数: python scripts/train.py --name lr_low --lr0 0.0005 --epochs 5
══════════════════════════════════════════════════════════════

YAML 文件正文如下：
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLO11 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
...

Ultralytics YOLO11s summary: 295 layers, 9,458,752 parameters, 9,458,736 gradients, 21.7 GFLOPs

Transferred 475/475 items from pretrained weights
optimizer: Adam(lr=0.0005, momentum=0.937) with parameter groups 88 weight(decay=0.0005), 102 weight(decay=0.0), 101 bias(decay=0.0)
train: Scanning /home/user/PROJECT/FSWD/FSW-MERGE/train/labels.cache... 1200 images, 0 backgrounds, 0 corrupt: 100%|██████████| 1200/1200 [00:00<00:00]
val: Scanning /home/user/PROJECT/FSWD/FSW-MERGE/valid/labels.cache... 300 images, 0 backgrounds, 0 corrupt: 100%|██████████| 300/300 [00:00<00:00]

Plotting labels to runs/train/lr_low/labels.jpg...
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/train/lr_low
Starting training for 5 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/5      3.45G      1.456      0.678      1.234         45        640: 100%|██████████| 75/75 [00:25<00:00,  2.96it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.15it/s]
                   all        300        450      0.623      0.698      0.678      0.431

        2/5      3.45G      1.123      0.512      1.045         45        640: 100%|██████████| 75/75 [00:24<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.21it/s]
                   all        300        450      0.689      0.745      0.723      0.489

        3/5      3.45G      0.934      0.445      0.912         45        640: 100%|██████████| 75/75 [00:24<00:00,  3.07it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.19it/s]
                   all        300        450      0.721      0.771      0.756      0.523

        4/5      3.45G      0.832      0.389      0.823         45        640: 100%|██████████| 75/75 [00:24<00:00,  3.06it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.17it/s]
                   all        300        450      0.748      0.791      0.778      0.551

        5/5      3.45G      0.756      0.345      0.751         45        640: 100%|██████████| 75/75 [00:24<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:03<00:00,  5.18it/s]
                   all        300        450      0.769      0.804      0.793      0.568

5 epochs completed in 0.035 hours.
Optimizer stripped from runs/train/lr_low/weights/last.pt, 19.2MB
Optimizer stripped from runs/train/lr_low/weights/best.pt, 19.2MB

Validating runs/train/lr_low/weights/best.pt...
Ultralytics YOLO11s summary: 295 layers, 9,458,752 parameters, 0 gradients, 21.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:05<00:00,  3.46it/s]
                   all        300        450      0.769      0.804      0.793      0.568
Speed: 0.4ms preprocess, 2.8ms inference, 0.0ms loss, 1.2ms postprocess per image
Results saved to runs/train/lr_low
Training complete. Results saved to: runs/train/lr_low

══════════════════════════════════════════════════════════════
✅ [任务 2/2] 完成: lr_low
结束时间: 2025-11-12 14:37:04
耗时: 199 秒 (3 分钟)
══════════════════════════════════════════════════════════════


╔══════════════════════════════════════════════════════════════╗
║          批量训练结束: lr_compare
║          结束时间: 2025-11-12 14:37:04
║          状态: ✅ 全部完成
║          总耗时: 398 秒 (6 分钟)
╚══════════════════════════════════════════════════════════════╝
```

---

## 🔍 日志分析命令

### 1. 快速查看任务进度
```bash
grep "🚀\|✅\|❌" lr_compare.log
```

**输出：**
```
🚀 [任务 1/2] 开始: lr_high
✅ [任务 1/2] 完成: lr_high
🚀 [任务 2/2] 开始: lr_low
✅ [任务 2/2] 完成: lr_low
```

### 2. 查看每个任务的耗时
```bash
grep "耗时" lr_compare.log
```

**输出：**
```
耗时: 199 秒 (3 分钟)
耗时: 199 秒 (3 分钟)
总耗时: 398 秒 (6 分钟)
```

### 3. 查看最终 mAP 结果
```bash
grep "mAP50-95):" lr_compare.log | grep "all"
```

**输出：**
```
                   all        300        450      0.785      0.818      0.809      0.583
                   all        300        450      0.769      0.804      0.793      0.568
```

### 4. 提取特定任务的日志
```bash
# 提取任务1的日志
sed -n '/🚀 \[任务 1\/2\]/,/✅ \[任务 1\/2\]/p' lr_compare.log > task1.log

# 提取任务2的日志
sed -n '/🚀 \[任务 2\/2\]/,/✅ \[任务 2\/2\]/p' lr_compare.log > task2.log
```

### 5. 实时监控训练进度
```bash
# 实时查看日志
tail -f lr_compare.log

# 只看重要信息
tail -f lr_compare.log | grep --line-buffered "Epoch\|mAP\|🚀\|✅"
```

### 6. 查看错误信息
```bash
grep -i "error\|fail\|exception" lr_compare.log
```

---

## 📊 对比：原版 vs 增强版

### 原版日志（scripts/run_yolo_batch.sh）
```
=========================================
[开始时间] 2025-11-12 14:30:26
批量训练：lr_compare
=========================================

YAML 文件正文如下：
...
Training complete. Results saved to: runs/train/lr_high

YAML 文件正文如下：
...
Training complete. Results saved to: runs/train/lr_low

=========================================
[结束时间] 2025-11-12 14:37:04
✅ 批量训练全部完成：lr_compare
=========================================
```

**问题：**
- ❌ 很难找到任务边界
- ❌ 不知道当前执行到哪个任务
- ❌ 没有单个任务的耗时统计

### 增强版日志（scripts/run_yolo_batch_v2.sh）
```
╔══════════════════════════════════════════════════════════════╗
║          批量训练开始: lr_compare
║          开始时间: 2025-11-12 14:30:26
╚══════════════════════════════════════════════════════════════╝

══════════════════════════════════════════════════════════════
🚀 [任务 1/2] 开始: lr_high
开始时间: 2025-11-12 14:30:26
══════════════════════════════════════════════════════════════

...任务1的输出...

══════════════════════════════════════════════════════════════
✅ [任务 1/2] 完成: lr_high
耗时: 199 秒 (3 分钟)
══════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════
🚀 [任务 2/2] 开始: lr_low
开始时间: 2025-11-12 14:33:45
══════════════════════════════════════════════════════════════

...任务2的输出...

══════════════════════════════════════════════════════════════
✅ [任务 2/2] 完成: lr_low
耗时: 199 秒 (3 分钟)
══════════════════════════════════════════════════════════════

╔══════════════════════════════════════════════════════════════╗
║          批量训练结束: lr_compare
║          总耗时: 398 秒 (6 分钟)
╚══════════════════════════════════════════════════════════════╝
```

**优势：**
- ✅ 明显的任务分隔符
- ✅ 显示任务进度（1/2, 2/2）
- ✅ 记录每个任务耗时
- ✅ 易于搜索和提取

---

## 🎯 推荐使用方式

### 日常训练：使用增强版
```bash
./scripts/run_yolo_batch_v2.sh my_experiment \
    --name task1 --lr0 0.001 -- \
    --name task2 --lr0 0.0005

# 在另一个终端实时监控
tail -f my_experiment.log | grep --line-buffered "🚀\|✅\|Epoch\|mAP"
```

### 快速分析结果
```bash
# 查看所有任务是否完成
grep "✅\|❌" my_experiment.log

# 对比所有任务的最终 mAP
grep "mAP50-95):" my_experiment.log | grep "all"

# 查看总耗时
grep "总耗时" my_experiment.log
```


