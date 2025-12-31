# YOLO11s 消融实验 - 模型组合总结

## 概述

本文档总结了为消融实验创建的所有YOLO11s模型配置文件及其训练命令。消融实验旨在系统地评估不同模块组合对模型性能的影响。

## 实验设计

### 关键组件

1. **主干网络 (Backbone) 模块**
   - `C3k2Ghost`: 使用Ghost卷积的轻量级主干
   - `C3k2GhostSimAMinner`: Ghost卷积 + SimAM注意力机制的主干

2. **注意力 (Attention) 模块**
   - `C2PSFCA`: Position-Sensitive Feature Channel Attention

3. **颈部网络 (Neck) 模块**
   - `VoVCsingle`: VoVGSCSPC模块（用于特征融合层）

4. **损失函数 (Loss Function)**
   - `CIoU` (默认基线)
   - `ShapeIoU` (通过 `--iou-type ShapeIoU` 参数指定)

### 实验策略

消融实验包含以下组合：
- 基线模型（无改进）
- 单一组件改进
- 两个组件组合
- 三个组件全部组合
- 每个配置都测试两种损失函数（CIoU和ShapeIoU）

---

## 模型配置文件列表

### 1. 基线模型
| 模型文件 | 描述 | 主干 | 注意力 | 颈部 |
|---------|------|------|--------|------|
| `yolo11s.yaml` | YOLO11s基线模型 | C3k2 | C2PSA | C3k2 |

### 2. 单组件改进模型

#### 2.1 仅主干网络改进
| 模型文件 | 描述 | 主干 | 注意力 | 颈部 |
|---------|------|------|--------|------|
| `yolo11s_C3k2Ghost.yaml` | Ghost主干 | C3k2Ghost | C2PSA | C3k2 |
| `yolo11s_C3k2GhostSimAMinner.yaml` | Ghost+SimAM主干 | C3k2GhostSimAMinner | C2PSA | C3k2 |

#### 2.2 仅注意力机制改进
| 模型文件 | 描述 | 主干 | 注意力 | 颈部 |
|---------|------|------|--------|------|
| `yolo11s_C2PSFCA.yaml` | FCA注意力 | C3k2 | C2PSFCA | C3k2 |

#### 2.3 仅颈部网络改进
| 模型文件 | 描述 | 主干 | 注意力 | 颈部 |
|---------|------|------|--------|------|
| `yolo11s_VoVCsingle.yaml` | VoV颈部 | C3k2 | C2PSA | VoVGSCSPC |

### 3. 两组件组合模型

#### 3.1 主干 + 注意力
| 模型文件 | 描述 | 主干 | 注意力 | 颈部 |
|---------|------|------|--------|------|
| `yolo11s_C3k2Ghost_C2PSFCA.yaml` | Ghost主干 + FCA注意力 | C3k2Ghost | C2PSFCA | C3k2 |
| `yolo11s_C3k2GhostSimAMinner_C2PSFCA.yaml` | Ghost+SimAM主干 + FCA注意力 | C3k2GhostSimAMinner | C2PSFCA | C3k2 |

#### 3.2 主干 + 颈部
| 模型文件 | 描述 | 主干 | 注意力 | 颈部 |
|---------|------|------|--------|------|
| `yolo11s_C3k2Ghost_VoVCsingle.yaml` | Ghost主干 + VoV颈部 | C3k2Ghost | C2PSA | VoVGSCSPC |
| `yolo11s_C3k2GhostSimAMinner_VoVCsingle.yaml` | Ghost+SimAM主干 + VoV颈部 | C3k2GhostSimAMinner | C2PSA | VoVGSCSPC |

#### 3.3 注意力 + 颈部
| 模型文件 | 描述 | 主干 | 注意力 | 颈部 |
|---------|------|------|--------|------|
| `yolo11s_C2PSFCA_VoVCsingle.yaml` | FCA注意力 + VoV颈部 | C3k2 | C2PSFCA | VoVGSCSPC |

### 4. 三组件全组合模型
| 模型文件 | 描述 | 主干 | 注意力 | 颈部 |
|---------|------|------|--------|------|
| `yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml` | Ghost + FCA + VoV | C3k2Ghost | C2PSFCA | VoVGSCSPC |
| `yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml` | Ghost+SimAM + FCA + VoV | C3k2GhostSimAMinner | C2PSFCA | VoVGSCSPC |

---

## 训练命令示例

### 基本训练命令格式

```bash
python train.py \
    --cfg /path/to/data.yaml \
    --model ultralytics/cfg/models/11/<model_file>.yaml \
    --name <experiment_name> \
    --epochs 300 \
    --batch-size 16 \
    --iou-type <CIoU|ShapeIoU> \
    [其他参数...]
```

### 详细训练命令

#### 1. 基线模型训练

```bash
# CIoU损失函数
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s.yaml \
    --name baseline_yolo11s_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# ShapeIoU损失函数
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s.yaml \
    --name baseline_yolo11s_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU
```

#### 2. 单组件改进模型训练

```bash
# C3k2Ghost主干 + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost.yaml \
    --name c3k2ghost_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C3k2Ghost主干 + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost.yaml \
    --name c3k2ghost_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU

# C3k2GhostSimAMinner主干 + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner.yaml \
    --name c3k2ghostsimaminner_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C3k2GhostSimAMinner主干 + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner.yaml \
    --name c3k2ghostsimaminner_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU

# C2PSFCA注意力 + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C2PSFCA.yaml \
    --name c2psfca_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C2PSFCA注意力 + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C2PSFCA.yaml \
    --name c2psfca_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU

# VoVCsingle颈部 + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_VoVCsingle.yaml \
    --name vovcsingle_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# VoVCsingle颈部 + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_VoVCsingle.yaml \
    --name vovcsingle_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU
```

#### 3. 两组件组合模型训练

```bash
# C3k2Ghost + C2PSFCA + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA.yaml \
    --name c3k2ghost_c2psfca_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C3k2Ghost + C2PSFCA + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA.yaml \
    --name c3k2ghost_c2psfca_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU

# C3k2GhostSimAMinner + C2PSFCA + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA.yaml \
    --name c3k2ghostsimam_c2psfca_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C3k2GhostSimAMinner + C2PSFCA + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA.yaml \
    --name c3k2ghostsimam_c2psfca_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU

# C3k2Ghost + VoVCsingle + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_VoVCsingle.yaml \
    --name c3k2ghost_vovcsingle_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C3k2Ghost + VoVCsingle + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_VoVCsingle.yaml \
    --name c3k2ghost_vovcsingle_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU

# C3k2GhostSimAMinner + VoVCsingle + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_VoVCsingle.yaml \
    --name c3k2ghostsimam_vovcsingle_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C3k2GhostSimAMinner + VoVCsingle + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_VoVCsingle.yaml \
    --name c3k2ghostsimam_vovcsingle_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU

# C2PSFCA + VoVCsingle + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C2PSFCA_VoVCsingle.yaml \
    --name c2psfca_vovcsingle_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C2PSFCA + VoVCsingle + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C2PSFCA_VoVCsingle.yaml \
    --name c2psfca_vovcsingle_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU
```

#### 4. 三组件全组合模型训练

```bash
# C3k2Ghost + C2PSFCA + VoVCsingle + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml \
    --name c3k2ghost_c2psfca_vovcsingle_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C3k2Ghost + C2PSFCA + VoVCsingle + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml \
    --name c3k2ghost_c2psfca_vovcsingle_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU

# C3k2GhostSimAMinner + C2PSFCA + VoVCsingle + CIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml \
    --name c3k2ghostsimam_c2psfca_vovcsingle_ciou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU

# C3k2GhostSimAMinner + C2PSFCA + VoVCsingle + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml \
    --name c3k2ghostsimam_c2psfca_vovcsingle_shapeiou \
    --epochs 300 \
    --batch-size 16 \
    --iou-type ShapeIoU
```

---

## 使用批量训练脚本

如果需要连续训练多个模型，可以使用 `run_yolo_batch_v2.sh` 脚本：

### 示例：对比两种主干网络

```bash
./run_yolo_batch_v2.sh backbone_comparison \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost.yaml \
    --name c3k2ghost_ciou \
    --iou-type CIoU \
    --epochs 300 -- \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner.yaml \
    --name c3k2ghostsimam_ciou \
    --iou-type CIoU \
    --epochs 300
```

### 示例：测试所有单组件改进（使用ShapeIoU）

```bash
./run_yolo_batch_v2.sh single_component_shapeiou \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost.yaml \
    --name c3k2ghost_shapeiou \
    --iou-type ShapeIoU \
    --epochs 300 -- \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner.yaml \
    --name c3k2ghostsimam_shapeiou \
    --iou-type ShapeIoU \
    --epochs 300 -- \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C2PSFCA.yaml \
    --name c2psfca_shapeiou \
    --iou-type ShapeIoU \
    --epochs 300 -- \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_VoVCsingle.yaml \
    --name vovcsingle_shapeiou \
    --iou-type ShapeIoU \
    --epochs 300
```

### 示例：完整消融实验（两组件组合）

```bash
./run_yolo_batch_v2.sh ablation_two_component \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA.yaml \
    --name ghost_fca_ciou \
    --iou-type CIoU -- \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_VoVCsingle.yaml \
    --name ghost_vov_ciou \
    --iou-type CIoU -- \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C2PSFCA_VoVCsingle.yaml \
    --name fca_vov_ciou \
    --iou-type CIoU
```

---

## 消融实验矩阵

下表展示了完整的消融实验设计：

| 实验编号 | 主干网络 | 注意力机制 | 颈部网络 | 损失函数 | 模型文件 | 实验名称 |
|---------|---------|-----------|---------|---------|---------|---------|
| 1 | C3k2 | C2PSA | C3k2 | CIoU | yolo11s.yaml | baseline_ciou |
| 2 | C3k2 | C2PSA | C3k2 | ShapeIoU | yolo11s.yaml | baseline_shapeiou |
| 3 | C3k2Ghost | C2PSA | C3k2 | CIoU | yolo11s_C3k2Ghost.yaml | c3k2ghost_ciou |
| 4 | C3k2Ghost | C2PSA | C3k2 | ShapeIoU | yolo11s_C3k2Ghost.yaml | c3k2ghost_shapeiou |
| 5 | C3k2GhostSimAMinner | C2PSA | C3k2 | CIoU | yolo11s_C3k2GhostSimAMinner.yaml | c3k2ghostsimam_ciou |
| 6 | C3k2GhostSimAMinner | C2PSA | C3k2 | ShapeIoU | yolo11s_C3k2GhostSimAMinner.yaml | c3k2ghostsimam_shapeiou |
| 7 | C3k2 | C2PSFCA | C3k2 | CIoU | yolo11s_C2PSFCA.yaml | c2psfca_ciou |
| 8 | C3k2 | C2PSFCA | C3k2 | ShapeIoU | yolo11s_C2PSFCA.yaml | c2psfca_shapeiou |
| 9 | C3k2 | C2PSA | VoVGSCSPC | CIoU | yolo11s_VoVCsingle.yaml | vovcsingle_ciou |
| 10 | C3k2 | C2PSA | VoVGSCSPC | ShapeIoU | yolo11s_VoVCsingle.yaml | vovcsingle_shapeiou |
| 11 | C3k2Ghost | C2PSFCA | C3k2 | CIoU | yolo11s_C3k2Ghost_C2PSFCA.yaml | ghost_fca_ciou |
| 12 | C3k2Ghost | C2PSFCA | C3k2 | ShapeIoU | yolo11s_C3k2Ghost_C2PSFCA.yaml | ghost_fca_shapeiou |
| 13 | C3k2GhostSimAMinner | C2PSFCA | C3k2 | CIoU | yolo11s_C3k2GhostSimAMinner_C2PSFCA.yaml | ghostsimam_fca_ciou |
| 14 | C3k2GhostSimAMinner | C2PSFCA | C3k2 | ShapeIoU | yolo11s_C3k2GhostSimAMinner_C2PSFCA.yaml | ghostsimam_fca_shapeiou |
| 15 | C3k2Ghost | C2PSA | VoVGSCSPC | CIoU | yolo11s_C3k2Ghost_VoVCsingle.yaml | ghost_vov_ciou |
| 16 | C3k2Ghost | C2PSA | VoVGSCSPC | ShapeIoU | yolo11s_C3k2Ghost_VoVCsingle.yaml | ghost_vov_shapeiou |
| 17 | C3k2GhostSimAMinner | C2PSA | VoVGSCSPC | CIoU | yolo11s_C3k2GhostSimAMinner_VoVCsingle.yaml | ghostsimam_vov_ciou |
| 18 | C3k2GhostSimAMinner | C2PSA | VoVGSCSPC | ShapeIoU | yolo11s_C3k2GhostSimAMinner_VoVCsingle.yaml | ghostsimam_vov_shapeiou |
| 19 | C3k2 | C2PSFCA | VoVGSCSPC | CIoU | yolo11s_C2PSFCA_VoVCsingle.yaml | fca_vov_ciou |
| 20 | C3k2 | C2PSFCA | VoVGSCSPC | ShapeIoU | yolo11s_C2PSFCA_VoVCsingle.yaml | fca_vov_shapeiou |
| 21 | C3k2Ghost | C2PSFCA | VoVGSCSPC | CIoU | yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml | ghost_fca_vov_ciou |
| 22 | C3k2Ghost | C2PSFCA | VoVGSCSPC | ShapeIoU | yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml | ghost_fca_vov_shapeiou |
| 23 | C3k2GhostSimAMinner | C2PSFCA | VoVGSCSPC | CIoU | yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml | ghostsimam_fca_vov_ciou |
| 24 | C3k2GhostSimAMinner | C2PSFCA | VoVGSCSPC | ShapeIoU | yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml | ghostsimam_fca_vov_shapeiou |

**总共24个实验配置**

---

## 建议的实验顺序

### 阶段1：基线和单组件（6个实验）
评估每个单独组件的影响

1. 基线 + CIoU
2. 基线 + ShapeIoU
3. C3k2Ghost + CIoU
4. C3k2GhostSimAMinner + CIoU
5. C2PSFCA + CIoU
6. VoVCsingle + CIoU

### 阶段2：最佳单组件 + 损失函数（4个实验）
选择阶段1中表现最好的2个单组件，测试ShapeIoU

7. 最佳组件1 + ShapeIoU
8. 最佳组件2 + ShapeIoU

### 阶段3：两组件组合（6-8个实验）
基于阶段1和2的结果，选择最有前途的组合

### 阶段4：三组件全组合（2-4个实验）
最终测试最佳配置的全组合

---

## 数据增强配置

如果需要启用数据增强，在训练命令中添加 `--augment` 参数：

```bash
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml \
    --name ghost_fca_vov_ciou_augmented \
    --epochs 300 \
    --batch-size 16 \
    --iou-type CIoU \
    --augment \
    --mosaic 1.0 \
    --mixup 0.2 \
    --hsv-h 0.015 \
    --hsv-s 0.7 \
    --hsv-v 0.4
```

---

## 加权数据加载器

如果存在类别不平衡问题，可以使用加权数据加载器：

```bash
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s.yaml \
    --name baseline_weighted \
    --epochs 300 \
    --batch-size 16 \
    --weighted-dataloader
```

---

## 结果分析

训练完成后，建议记录以下指标进行对比：

1. **精度指标**
   - mAP@0.5
   - mAP@0.5:0.95
   - Precision
   - Recall

2. **效率指标**
   - Parameters (参数量)
   - GFLOPs (计算复杂度)
   - Inference time (推理时间)
   - FPS (帧率)

3. **训练指标**
   - Training time per epoch
   - Loss convergence
   - GPU memory usage

4. **各类别性能**
   - Per-class AP
   - Confusion matrix

---

## 快速开始示例

### 快速测试一个配置

```bash
# 测试Ghost主干 + FCA注意力 + VoV颈部 + ShapeIoU
python train.py \
    --cfg /home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml \
    --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml \
    --name quick_test \
    --epochs 50 \
    --batch-size 16 \
    --iou-type ShapeIoU
```

### 运行完整消融实验

创建一个脚本 `run_full_ablation.sh`:

```bash
#!/bin/bash

DATA_CFG="/home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml"
EPOCHS=300
BATCH_SIZE=16

# 基线
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s.yaml --name baseline_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s.yaml --name baseline_shapeiou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type ShapeIoU

# 单组件 - 主干
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost.yaml --name c3k2ghost_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner.yaml --name c3k2ghostsimam_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU

# 单组件 - 注意力
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s_C2PSFCA.yaml --name c2psfca_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU

# 单组件 - 颈部
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s_VoVCsingle.yaml --name vovcsingle_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU

# 两组件组合
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA.yaml --name ghost_fca_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_VoVCsingle.yaml --name ghost_vov_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s_C2PSFCA_VoVCsingle.yaml --name fca_vov_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU

# 三组件全组合
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml --name ghost_fca_vov_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU
python train.py --cfg $DATA_CFG --model ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml --name ghostsimam_fca_vov_ciou --epochs $EPOCHS --batch-size $BATCH_SIZE --iou-type CIoU

echo "消融实验完成！"
```

然后运行：

```bash
chmod +x run_full_ablation.sh
./run_full_ablation.sh
```

---

## 注意事项

1. **路径配置**：请根据实际情况修改数据配置文件路径 `/home/user/FSW-AUG/FSW-MERGE_augmented_double/data.yaml`

2. **批量大小**：根据GPU显存调整 `--batch-size`，建议值：
   - 8GB GPU: batch-size 8-16
   - 12GB GPU: batch-size 16-32
   - 24GB GPU: batch-size 32-64

3. **训练时长**：每个实验大约需要几小时到一天的训练时间（取决于数据集大小和GPU性能）

4. **实验管理**：建议使用统一的实验管理工具（如TensorBoard或WandB）来跟踪所有实验结果

5. **模型验证**：在正式开始长时间训练前，建议先运行几个epoch验证模型配置正确

---

## 文件清单

### 新创建的模型配置文件

所有模型文件位于 `ultralytics/cfg/models/11/` 目录：

1. ✅ `yolo11s_C3k2Ghost_C2PSFCA.yaml`
2. ✅ `yolo11s_C3k2Ghost_VoVCsingle.yaml`
3. ✅ `yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml`
4. ✅ `yolo11s_C3k2GhostSimAMinner_C2PSFCA.yaml`
5. ✅ `yolo11s_C3k2GhostSimAMinner_VoVCsingle.yaml`
6. ✅ `yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml`
7. ✅ `yolo11s_C2PSFCA_VoVCsingle.yaml`

### 已存在的模型配置文件

8. ✅ `yolo11s.yaml` (基线)
9. ✅ `yolo11s_C3k2Ghost.yaml`
10. ✅ `yolo11s_C3k2GhostSimAMinner.yaml`
11. ✅ `yolo11s_C2PSFCA.yaml`
12. ✅ `yolo11s_VoVCsingle.yaml`

---

## 联系与支持

如有问题或需要进一步的帮助，请参考项目文档或联系项目维护者。

---

**文档创建日期**: 2025-12-31  
**版本**: 1.0  
**作者**: GitHub Copilot Workspace Agent
