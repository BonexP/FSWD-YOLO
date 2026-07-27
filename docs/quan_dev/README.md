# FSWD-YOLO ONNX 与 Vitis AI 检查指南

这套工具用于完成两项彼此独立的工作：在训练主机上导出并验证 ONNX；在 Vitis AI 主机上针对明确的 DPU target 检查 PyTorch 图。脚本只检测依赖，不调用 `pip`、`conda` 或其他安装命令。

## 主机分工

推荐在保存训练环境和 `best.pt` 的代码主机上导出 ONNX。该步骤需要 PyTorch、仓库内的 Ultralytics 和 ONNX，不需要 Vitis AI。

Vitis AI 主机负责运行 Inspector。请在该主机克隆本仓库并切换到与导出主机相同的提交，因为 FSWD-YOLO 包含自定义模型模块；随后把 `best.pt` 传到该主机。Inspector 读取的是 `.pt`，不是已导出的 `.onnx`。

```text
训练/代码主机: best.pt -> export_fswd_onnx.py -> ONNX + report
                         |
                         +-> 将 best.pt 和相同代码提交交给 Vitis AI 主机

Vitis AI 主机: best.pt + target -> inspect_fswd_vitis.py -> Inspector 结果 + manifest
```

板卡尚未确定时，可以先完成 ONNX 导出和图检查，但不能运行有意义的 Inspector 检查，也不能据此确认 DPU 兼容性。

## 1. 导出主机

从仓库根目录先确认解释器和依赖。下面的命令不会安装任何内容：

```bash
python -c "import sys, torch, onnx; print(sys.executable); print(torch.__version__, onnx.__version__)"
python scripts/export_fswd_onnx.py --help
```

`pip show ultralytics` 可能显示未安装，这是源码检出模式下的正常现象。部署脚本会把仓库根目录加入 Python 搜索路径，直接使用当前仓库的 `ultralytics/`。它还会在导入 Ultralytics 前把 PyTorch 1.10 的不可哈希 `TorchVersion` 规范化为普通版本字符串，避免版本检查缓存报 `TypeError: unhashable type: 'TorchVersion'`。

执行固定 batch 1、FP32、静态输入、无 NMS、无 simplify 的 ONNX 导出：

```bash
python scripts/export_fswd_onnx.py \
  --weights /data/weights/best.pt \
  --output /data/deploy/fswd-yolo.onnx \
  --imgsz 640 \
  --opset 13
```

工具要求 `onnx>=1.12.0,<1.18.0`，与本仓库导出依赖范围一致。它会运行 `onnx.checker.check_model`，并生成：

- `/data/deploy/fswd-yolo.onnx`
- `/data/deploy/fswd-yolo.onnx.report.json`

报告包含输入/输出名称、类型和形状、opset、算子计数、SHA256、Git 提交、包版本和执行状态。`ReduceMean`、`ReduceSum`、`Pow`、`Div`、`MatMul`、`Softmax`、`Expand`、`Transpose` 会列入 `review_operations`，表示需要结合 target 继续审查，不表示一定不支持。

安装了 ONNX Runtime 时，可以增加确定性数值对齐：

```bash
python scripts/export_fswd_onnx.py \
  --weights /data/weights/best.pt \
  --output /data/deploy/fswd-yolo.onnx \
  --verify-runtime \
  --overwrite
```

默认比较阈值为 `rtol=1e-4`、`atol=1e-5`。输出或报告已存在时，工具会拒绝执行，除非明确传入 `--overwrite`。

## 2. Vitis AI 主机

进入已激活的 Vitis AI 3.5 PyTorch/NNDCT 环境，并在 FSWD-YOLO 仓库根目录运行：

```bash
python -c "import sys, torch, pytorch_nndct; print(sys.executable); print(torch.__version__)"
python -c "import ultralytics; print(ultralytics.__file__)"
python scripts/inspect_fswd_vitis.py --help
```

在确定板卡、DPU 架构和匹配的 Inspector target 后执行：

```bash
python scripts/inspect_fswd_vitis.py \
  --weights /workspace/weights/best.pt \
  --target ACTUAL_TARGET \
  --output-dir /workspace/inspect_fswd
```

`--target` 没有默认值，必须替换 `ACTUAL_TARGET`。请从所用板卡的平台文件、DPU fingerprint 或对应版本的 [AMD Vitis AI 文档](https://docs.amd.com/r/zh-CN/ug1414-vitis-ai/Vitis-AI-%E5%AE%B9%E5%99%A8) 获取匹配值，不要使用其他板卡示例中的 target。

默认输入为 `[1, 3, 640, 640]` 的 CPU FP32 tensor，`--verbose-level` 默认为 `2`，图像格式默认为 SVG。结果目录内会生成 `inspection_manifest.json`，记录 checkpoint 哈希、target、输入形状、Git/运行时信息及成功或失败详情。已有 manifest 需要 `--overwrite`。

Vitis AI 3.5 的 Inspector 报告代码没有为 3D permute `(0, 2, 1)` 提供布局说明，可能在编译完成后以 `KeyError: (0, 2, 1)` 退出。包装器仅在检测到该版本的危险直接索引实现时安装报告层兼容方法：已知布局保留原说明，未知布局记录原始 permutation order。manifest 的 `compatibility.vitis_35_permute_report_patch_applied` 会说明本次是否应用补丁。该兼容处理不改变模型图、DPU 分区或编译结果。

默认的 `--activation-experiment none` 是原模型审计。包装器只在内存中把 `nn.SiLU.inplace` 设为 `False`，不再绑定 `C2f.forward_split()`，也不会修改 checkpoint 或训练模型定义。脚本使用同一个确定性输入比较准备前后的原始预测；shape 或 `torch.allclose(rtol=1e-5, atol=1e-6)` 不通过时会停止检查。manifest 的 `model_preparation` 会记录 `mode: original_model_audit`、准备计数和最大/平均绝对误差。

Inspector 完成后，脚本会解析本次生成的 `inspect_*.txt`，按节点、算子和原因去重，并把以下内容写入 `inspection_summary`：唯一 CPU finding 数、分类计数、各类算子计数、直接阻塞算子的示例，以及 `requires_cpu_fallback`。顶层 `status: ok` 只表示脚本和 Inspector 成功执行；如果 `requires_cpu_fallback: true`，就不能解释为模型能够全图运行在 DPU 上。不要用 `grep -c` 的原始次数作为节点数，因为同一报告行的节点名、算子列和原因列可能多次出现相同算子字符串。

先用独立目录重新建立原模型审计基线：

```bash
python scripts/inspect_fswd_vitis.py \
  --weights /workspace/best.pt \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir /workspace/inspect_fswd_b4096_audit \
  --activation-experiment none \
  --overwrite
```

为了判断替换激活函数能否减少直接阻塞，可以分别运行两个显式的非等价图实验：

```bash
python scripts/inspect_fswd_vitis.py \
  --weights /workspace/best.pt \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir /workspace/inspect_fswd_b4096_hardswish \
  --activation-experiment hardswish \
  --overwrite

python scripts/inspect_fswd_vitis.py \
  --weights /workspace/best.pt \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir /workspace/inspect_fswd_b4096_hardsigmoid \
  --activation-experiment hardsigmoid \
  --overwrite
```

这两个实验只在内存中替换 SiLU 的前向运算。数值不一致是预期结果，脚本会记录 `prediction_comparison` 的漂移但仍运行 Inspector；输出 shape 改变仍会中止。实验成功只说明相应图的 target 分区情况，不说明原 checkpoint 保持了检测精度，也不会产生可部署 checkpoint。最终采用任何非等价替换前都必须重新训练或微调并验证精度。

比较三个 manifest 的 `inspection_summary.direct_blockers`、`operators_by_category` 和 `category_counts`，并对照 Inspector 控制台的 device/DPU subgraph 数量。只有直接阻塞算子减少且 DPU 分区更连续，才能把某个激活函数列为后续训练候选；这仍不等于量化、编译或板上运行已经通过。

## 3. DPU 候选模型的训练前筛选

原始 `fswd-yolo.yaml` 和 `C2PSFCA` 保持不变。DPU 候选使用独立的 `ultralytics/cfg/models/11/fswd-yolo-dpu.yaml`：全局卷积激活为 Hardswish，C2f 类模块使用独立投影替代通道切片，`C2PSFCADPU` 仍保留 keep、spatial 和 channel 三分支。Detect 的卷积预测头保留在候选图中，DFL、框解码、sigmoid 和 NMS 留在主机端。

先在具有完整 PyTorch/Ultralytics/thop 环境的代码主机运行资源门槛检查：

```bash
python scripts/profile_fswd_dpu_candidate.py \
  --baseline-config ultralytics/cfg/models/11/fswd-yolo.yaml \
  --candidate-config ultralytics/cfg/models/11/fswd-yolo-dpu.yaml \
  --imgsz 640 \
  --output /data/deploy/fswd-dpu-profile.json \
  --overwrite
```

通过条件为候选参数量不超过原模型的 110%，并且候选 GFLOPs 不高于原模型。缺少 `thop` 或 FLOP 计算返回 0 时工具会失败，不会把未知结果当作通过。

如果需要从现有 `best.pt` 初始化可训练候选，执行：

```bash
python scripts/initialize_fswd_dpu.py \
  --source-checkpoint /data/weights/best.pt \
  --model-config ultralytics/cfg/models/11/fswd-yolo-dpu.yaml \
  --output /data/deploy/fswd-yolo-dpu-init.pt \
  --report /data/deploy/fswd-yolo-dpu-init.migration.json \
  --imgsz 640 \
  --overwrite
```

迁移工具会把原 C2f 融合投影的输出通道复制到独立投影，记录 C2PSFCA 重设计中新增和有意不映射的参数，并验证候选模型的 raw Detect 输出经过原生主机解码后与其常规 FP32 推理一致。源 checkpoint 不会被修改。生成的初始化 checkpoint 采用了新的激活函数和注意力结构，因此在重新训练或微调并验证之前不具备精度结论。

在 Vitis AI 3.5 主机上可以先检查随机权重候选图。这个步骤只判断图结构和 target 分区，不判断精度：

```bash
python scripts/inspect_fswd_vitis.py \
  --model-config ultralytics/cfg/models/11/fswd-yolo-dpu.yaml \
  --target DPUCZDX8G_ISA1_B4096 \
  --raw-detect-output \
  --output-dir /workspace/inspect_fswd_dpu_candidate \
  --overwrite
```

也可以检查已初始化或后续训练得到的 DPU checkpoint：

```bash
python scripts/inspect_fswd_vitis.py \
  --weights /workspace/fswd-yolo-dpu-init.pt \
  --target DPUCZDX8G_ISA1_B4096 \
  --raw-detect-output \
  --output-dir /workspace/inspect_fswd_dpu_checkpoint \
  --overwrite
```

`inspection_manifest.json` 的 `input.model_source.weight_state` 会区分 `random_initialization` 与 `trained`，`input.output_contract` 应为 `raw_detect_feature_maps`。检查报告：

```bash
python -m json.tool /workspace/inspect_fswd_dpu_candidate/inspection_manifest.json
grep -n "assigned to CPU\|can't be converted to XIR" /workspace/inspect_fswd_dpu_candidate/inspect_*.txt
```

训练前图门槛要求 backbone、neck、`C2PSFCADPU`、卷积 Detect 头和三张 raw 输出中没有直接不支持或被迫分配到 CPU 的节点，并且 Inspector 控制台至少报告一个 DPU subgraph。`DPUCZDX8G_ISA1_B4096` 目前仅用于软件筛选；确定板卡后仍须使用该平台实际 `arch.json` 或 fingerprint 重新量化、编译和验证。

## Vitis 环境依赖原则

不要在 Vitis AI 环境中使用 `conda install timm` 或 `conda install onnx`。Conda 求解可能替换 AMD 容器预装的 PyTorch/NNDCT 组合，使 `pytorch_nndct` 再次不可导入。

如果已确认当前容器只缺少 FSWD 模型定义所需的轻量包，可由操作者手工执行：

```bash
python -m pip install --no-deps einops==0.8.0 timm==0.6.7
python -c "import torch, pytorch_nndct; print(torch.__version__, 'NNDCT OK')"
```

只有确实要在 Vitis 主机上读取 ONNX 时，才单独安装兼容版本，并立即回归检查 NNDCT：

```bash
python -m pip install --no-deps onnx==1.14.0
python -c "import torch, pytorch_nndct, onnx; print(torch.__version__, onnx.__version__, 'NNDCT OK')"
```

这些是人工环境准备示例，不会被工具自动执行。若第一条导入检查失败，应先恢复 AMD 提供的 Vitis AI 环境，不要继续运行 Inspector。

## 如何解释结果

ONNX checker 通过只说明模型符合 ONNX 格式规则；ONNX Runtime 数值对齐只说明 CPU 参考执行接近 PyTorch。二者都不能证明模型可在某个 AMD DPU 上完整部署。

板卡确定后的最小判定链为：使用真实 target 运行 Inspector，处理不支持或分配到 CPU 的算子，再使用该平台对应的量化器和编译器验证。量化精度、编译成功、DPU/CPU 分区和板上推理结果才构成最终部署证据。
