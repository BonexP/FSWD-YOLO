# NewConcat YAML 快速参考 / Quick Reference

## 语法 / Syntax
```yaml
- [[from_layers], repeats, NewConcat, [out_channels, kernel_size, use_dw, post_fusion]]
```

## 参数 / Parameters
| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
|-----------------|------------|-----------------|-------------------|
| `out_channels` | int | **必需 / Required** | 输出通道数 / Output channels |
| `kernel_size` | int | 3 | 卷积核大小 / Kernel size |
| `use_dw` | bool | True | 深度可分离卷积 / Depthwise conv |
| `post_fusion` | bool | False | 后融合处理 / Post-fusion |

## 快速示例 / Quick Examples

### 1️⃣ 默认配置（最常用）
```yaml
- [[-1, 13], 1, NewConcat, [512]]
```
→ `out_channels=512, kernel_size=3, use_dw=True, post_fusion=False`

### 2️⃣ 大卷积核
```yaml
- [[-1, 13], 1, NewConcat, [512, 5]]
```
→ `kernel_size=5`，其他默认

### 3️⃣ 标准卷积（不用 DW）
```yaml
- [[-1, 13], 1, NewConcat, [512, 3, False]]
```
→ `use_dw=False`，更强特征

### 4️⃣ 完整配置
```yaml
- [[-1, 13], 1, NewConcat, [512, 3, True, True]]
```
→ 所有参数显式指定

## 场景推荐 / Recommended Scenarios

| 场景 | 配置 | 原因 |
|-----|------|-----|
| 🚀 边缘设备 | `[512]` | 最高效率 |
| ⚡ 实时检测 | `[512]` 或 `[512, 3]` | 平衡速度精度 |
| 🎯 高精度 | `[512, 3, False]` | 更强特征表达 |
| 🔥 多分支融合(4+) | `[512, 3, True, True]` | 更好梯度流 |
| 📏 大目标 | `[512, 5]` | 更大感受野 |

## 完整示例 / Complete Example
```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, NewConcat, [512]]              # P4: 默认配置
  - [-1, 2, C3k2, [512, False]]
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, NewConcat, [256, 5]]           # P3: 大卷积核
  - [-1, 2, C3k2, [256, False]]
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, NewConcat, [512, 3, False]]   # P4 head: 标准卷积
  - [-1, 2, C3k2, [512, False]]
  
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, NewConcat, [1024, 3, True, True]]  # P5: 后融合
  - [-1, 2, C3k2, [1024, True]]
```

## 测试命令 / Test Commands
```bash
# 测试基础解析
python test_newconcat_parsing.py

# 测试高级配置
python test_advanced_newconcat.py

# 加载模型测试
python -c "from ultralytics import YOLO; model = YOLO('ultralytics/cfg/models/11/yolo11s_NewConcat2.yaml'); print('Success!')"
```

## 相关文档 / Related Docs
- 📖 详细指南: `NEWCONCAT_YAML_GUIDE.md`
- 🔧 修复说明: `NEWCONCAT_PARAMETER_PARSING_FIX.md`
- 💻 模块代码: `ultralytics/nn/modules/NewConcat.py`
- ⚙️ 解析代码: `ultralytics/nn/tasks.py` (line ~1768)

