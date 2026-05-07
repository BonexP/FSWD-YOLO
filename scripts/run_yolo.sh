#!/usr/bin/env bash
# run_yolo.sh - YOLO 模型训练启动脚本
# 用法：./run_yolo.sh <name> [其他 train.py 参数...]

set -euo pipefail

# 显示帮助文档

show_help() {
    cat << EOF
YOLO 模型训练脚本

用法：
    $0 <name> [选项...]

必需参数：
    name                训练任务名称（会自动添加 --name 前缀）

可选参数：
    --cfg PATH          数据集配置文件路径 (默认: /home/user/PROJECT/FSWD/FSW-MERGE/data.yaml)
    --model PATH        模型配置文件路径 (默认: ultralytics/cfg/models/11/yolo11s.yaml)
    --epochs N          训练轮数 (默认: 300)
    --batch-size N      批次大小 (默认: 16)
    --img-size N        输入图片大小 (默认: 640)
    --optimizer TYPE    优化器 (SGD/Adam, 默认: Adam)
    --lr0 FLOAT         初始学习率 (默认: 0.001)
    --weight-decay FLOAT 权重衰减 (默认: 0.0005)
    --momentum FLOAT    优化器动量 (默认: 0.937)
    --project PATH      实验结果保存目录 (默认: runs/train)

数据增强参数：
    --augment           启用数据增强总开关
    --auto-augment STR  自动增强策略 (默认: randaugment)
    --mosaic FLOAT      mosaic 增强概率 (默认: 1.0)
    --mixup FLOAT       mixup 增强概率 (默认: 0.2)
    --hsv-h FLOAT       HSV 色调增强 (默认: 0.015)
    --hsv-s FLOAT       HSV 饱和度增强 (默认: 0.7)
    --hsv-v FLOAT       HSV 明度增强 (默认: 0.4)
    --translate FLOAT   平移增强幅度 (默认: 0.1)
    --scale FLOAT       尺度缩放增强 (默认: 0.5)
    --fliplr FLOAT      水平翻转概率 (默认: 0.5)
    --erasing FLOAT     随机擦除概率 (默认: 0.4)
    --warmup-epochs N   预热轮数 (默认: 5)
    --close-mosaic N    关闭 mosaic 的轮数 (默认: 10)

示例：
    # 基础训练（不启用增强）
    $0 baseline_exp

    # 启用数据增强的训练
    $0 augment_exp --augment

    # 启用增强并微调参数
    $0 custom_exp --augment --mosaic 0.8 --mixup 0.1 --epochs 200

    # 完整自定义
    $0 full_custom --augment --batch-size 32 --lr0 0.0005 --optimizer SGD

注意：
    - 训练将在后台运行，日志保存到 <name>.log
    - 可以安全关闭终端，训练会继续进行
    - 使用 tail -f <name>.log 查看实时日志


EOF
}

# 检查是否至少有一个参数（name）
if [[ $# -lt 1 ]]; then

    show_help
    exit 1
fi

# 检查 -h 或 --help
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# 获取第一个参数作为训练名称
NAME=$1
shift  # 移除第一个参数，剩下的都是 train.py 的参数

# 设置日志文件路径
LOG_FILE="./${NAME}.log"

# 构建完整的 python 命令
# $@ 会展开为所有剩余的命令行参数
TRAIN_CMD="python train.py --name \"${NAME}\" $@"

# 打印即将执行的命令（方便调试）
echo "========================================="
echo "[$(date '+%F %T')] 启动 YOLO 训练"
echo "========================================="
echo "训练名称：${NAME}"
echo "日志文件：${LOG_FILE}"
echo "执行命令：${TRAIN_CMD}"
echo "========================================="

# 启动训练（后台运行）
nohup python train.py --name "${NAME}" "$@" > "${LOG_FILE}" 2>&1 &

# 获取进程 PID
PID=$!

# 打印成功信息
echo "✅ 训练已在后台启动"
echo "   进程 PID：${PID}"
echo "   查看日志：tail -f ${LOG_FILE}"
echo "   停止训练：kill ${PID}"
echo "========================================="

# 等待 1 秒后检查进程是否还在运行
sleep 1
if ps -p ${PID} > /dev/null 2>&1; then
    echo "✅ 进程运行正常"
else
    echo "❌ 进程启动失败，请检查日志文件：${LOG_FILE}"
    exit 1
fi
