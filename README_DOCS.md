# 📚 YOLO11 训练脚本文档索引

## 🎯 快速导航

### 核心脚本
- **scripts/train.py** - 主训练脚本
- **scripts/run_yolo.sh** - 单次训练启动脚本（后台运行）
- **scripts/run_yolo_batch.sh** - 批量串行训练脚本（基础版）
- **scripts/run_yolo_batch_v2.sh** - 批量串行训练脚本（增强日志版）⭐ 推荐

### 文档列表

#### 🚀 使用指南
1. **docs/guides/BATCH_TRAINING_EXAMPLES.md** - 批量训练使用示例
   - 基本用法
   - 5个实际示例
   - 监控和故障排查
   - 最佳实践

#### 📝 日志相关
2. **docs/guides/LOGGING_SUMMARY.md** ⭐ - 日志控制快速参考
   - 核心问题快速答案
   - 实用命令速查
   - 常见问题解答

3. **docs/guides/LOGGING_EXPLAINED.md** - 日志机制详细解析
   - 日志流向总览
   - 逐层分析
   - 改进建议

4. **docs/guides/REDIRECTION_EXPLAINED.md** - 重定向机制图解
   - 命令分解
   - 文件描述符
   - 实验演示

5. **docs/guides/LOG_EXAMPLE.md** - 实际日志输出示例
   - 完整日志示例
   - 日志分析命令
   - 版本对比

#### 🔍 代码审查
6. **docs/reports/CODE_REVIEW_SUMMARY.md** - 代码审查报告
   - 代码质量评估
   - 潜在改进点
   - 使用场景对比

---

## 📖 阅读路径推荐

### 路径 1：快速上手（5分钟）
```
1. docs/guides/LOGGING_SUMMARY.md (快速了解日志控制)
   ↓
2. docs/guides/BATCH_TRAINING_EXAMPLES.md (学习基本用法)
   ↓
3. 开始使用！
```

### 路径 2：深入理解（15分钟）
```
1. docs/guides/LOGGING_SUMMARY.md (概览)
   ↓
2. docs/guides/LOGGING_EXPLAINED.md (详细机制)
   ↓
3. docs/guides/REDIRECTION_EXPLAINED.md (重定向原理)
   ↓
4. docs/guides/LOG_EXAMPLE.md (实际示例)
   ↓
5. 完全掌握！
```

### 路径 3：代码审查（10分钟）
```
1. docs/reports/CODE_REVIEW_SUMMARY.md (整体评估)
   ↓
2. 查看脚本源码
   ↓
3. 根据建议改进
```

---

## 🎯 按需查找

### 我想知道...

#### "如何使用批量训练脚本？"
→ 查看 **docs/guides/BATCH_TRAINING_EXAMPLES.md**

#### "日志是怎么控制的？"
→ 查看 **docs/guides/LOGGING_SUMMARY.md** (快速答案)
→ 查看 **docs/guides/LOGGING_EXPLAINED.md** (详细解析)

#### "重定向是什么原理？"
→ 查看 **docs/guides/REDIRECTION_EXPLAINED.md**

#### "日志文件长什么样？"
→ 查看 **docs/guides/LOG_EXAMPLE.md**

#### "代码有没有问题？"
→ 查看 **docs/reports/CODE_REVIEW_SUMMARY.md**

#### "如何对比两个超参数？"
→ 查看 **docs/guides/BATCH_TRAINING_EXAMPLES.md** 示例1

#### "如何实时监控训练？"
→ 查看 **docs/guides/LOGGING_SUMMARY.md** 实用命令部分

#### "为什么 && 不能串行执行？"
→ 查看 **docs/reports/CODE_REVIEW_SUMMARY.md** 已知问题部分

---

## 🛠️ 脚本功能对比

| 功能 | scripts/run_yolo.sh | scripts/run_yolo_batch.sh | scripts/run_yolo_batch_v2.sh |
|------|-------------|-------------------|----------------------|
| 单次训练 | ✅ | ❌ | ❌ |
| 批量训练 | ❌ | ✅ | ✅ |
| 串行执行 | ❌ | ✅ | ✅ |
| 后台运行 | ✅ | ✅ | ✅ |
| 日志分隔符 | N/A | ❌ | ✅⭐ |
| 任务耗时 | ❌ | ❌ | ✅⭐ |
| 进度显示 | ❌ | ❌ | ✅⭐ |
| 推荐度 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 推荐使用场景

- **单次长时间训练** → `scripts/run_yolo.sh`
- **批量对比实验** → `scripts/run_yolo_batch_v2.sh` ⭐
- **快速测试** → 直接调用 `python scripts/train.py`
- **并行训练（多GPU）** → 手动后台启动多个 `scripts/run_yolo.sh`

---

## 📋 快速命令参考

### 启动训练
```bash
# 单次训练
./scripts/run_yolo.sh my_exp --augment --epochs 200

# 批量对比（推荐）
./scripts/run_yolo_batch_v2.sh compare \
    --name baseline --epochs 100 -- \
    --name optimized --augment --lr0 0.001 --epochs 100
```

### 监控训练
```bash
# 实时查看日志
tail -f my_exp.log

# 查看任务进度
grep "🚀\|✅" my_exp.log

# 检查进程
ps aux | grep train.py

# 查看GPU使用
watch -n 1 nvidia-smi
```

### 停止训练
```bash
# 使用PID停止
kill <PID>

# 停止所有训练
pkill -f "python scripts/train.py"
```

### 分析结果
```bash
# 查看最终mAP
grep "mAP50-95):" my_exp.log | grep "all"

# 查看耗时
grep "耗时" my_exp.log

# 提取特定任务日志
sed -n '/🚀 \[任务 1/,/✅ \[任务 1/p' my_exp.log > task1.log
```

---

## 🎓 核心概念速记

### 1. 为什么 `&&` 不能直接串行训练？
```bash
./scripts/run_yolo.sh task1 && ./scripts/run_yolo.sh task2
#             脚本使用 nohup & 立即返回
#             ↓
#          shell 认为 task1 已完成
#             ↓
#          立即启动 task2
#             ↓
#          结果：并行运行！❌
```

**解决方案：**
```bash
./scripts/run_yolo_batch.sh batch_name \
    --name task1 ... -- \
    --name task2 ...
# 将所有任务用 && 连接后再放入 nohup
# 结果：串行运行！✅
```

### 2. 日志重定向原理
```bash
command > file.log 2>&1 &
#       │          │      │
#       │          │      └─ 后台运行
#       │          └──────── stderr → stdout
#       └─────────────────── stdout → 文件

结果：所有输出都进入 file.log
```

### 3. nohup 的作用
```bash
# 没有 nohup
command &
# 关闭终端 → 进程终止 ❌

# 有 nohup
nohup command &
# 关闭终端 → 进程继续 ✅
```

---

## 🔧 故障排查清单

### 问题：脚本启动后立即退出
```bash
# 检查日志
cat <batch_name>.log

# 常见原因：
# - 数据集路径错误
# - Python环境问题
# - 参数格式错误
```

### 问题：看不到日志输出
```bash
# 检查文件是否存在
ls -lh *.log

# 实时查看
tail -f <batch_name>.log

# 检查进程
ps aux | grep train.py
```

### 问题：GPU内存不足
```bash
# 降低batch size
./scripts/run_yolo_batch_v2.sh exp \
    --name task1 --batch-size 8 -- \
    --name task2 --batch-size 8

# 或确保串行执行（一次只运行一个）
```

### 问题：无法找到分隔符
```bash
# 使用增强版脚本
./scripts/run_yolo_batch_v2.sh ...

# 搜索分隔符
grep "🚀\|✅" experiment.log
```

---

## 📞 获取帮助

### 查看脚本帮助
```bash
./scripts/run_yolo_batch.sh --help
./scripts/run_yolo_batch_v2.sh --help
```

### 查看文档
```bash
# 查看索引（本文件）
cat README_DOCS.md

# 查看快速参考
cat docs/guides/LOGGING_SUMMARY.md

# 查看详细解析
cat docs/guides/LOGGING_EXPLAINED.md
```

---

## 🎯 最后的建议

### 日常使用推荐
1. **对比实验** → 使用 `scripts/run_yolo_batch_v2.sh`
2. **单次训练** → 使用 `scripts/run_yolo.sh`
3. **快速测试** → 直接 `python scripts/train.py`

### 最佳实践
1. ✅ 使用有意义的实验名称
2. ✅ 训练前先用小 epochs 测试
3. ✅ 使用 `tail -f` 实时监控
4. ✅ 定期清理旧日志
5. ✅ 记录实验配置和结果

### 进阶技巧
1. 使用 TensorBoard 可视化结果
2. 结合 `watch nvidia-smi` 监控GPU
3. 使用 `tmux` 管理多个监控窗口
4. 编写自动分析脚本提取关键指标

---

## 🌟 总结

你现在拥有：
- ✅ 3个训练脚本（单次、批量基础、批量增强）
- ✅ 6个详细文档（使用、日志、审查）
- ✅ 完整的日志控制机制理解
- ✅ 丰富的使用示例和最佳实践

**开始你的高效训练之旅吧！** 🚀

---

## 📝 文档更新日期
2025-11-12

## 🔖 版本
v1.0

## 👤 维护者
GitHub Copilot

