#!/usr/bin/env bash
# run_yolo_batch.sh - YOLO 模型批量串行训练脚本
# 用法：./run_yolo_batch.sh <batch_name> <task1_args> -- <task2_args> [-- <task3_args> ...]

set -euo pipefail

# 显示帮助文档
show_help() {
    cat << EOF
YOLO 模型批量串行训练脚本

用法：
    $0 <batch_name> <task1_args> -- <task2_args> [-- <task3_args> ...]

参数说明：
    batch_name          批量训练的总名称（用于日志文件命名）
    task_args           每个训练任务的参数（与 train.py 参数一致）
    --                  分隔符，用于区分不同的训练任务

示例：
    # 对比两个学习率
    $0 lr_compare \\
        --name lr0.001 --lr0 0.001 --epochs 100 -- \\
        --name lr0.0005 --lr0 0.0005 --epochs 100

    # 对比增强与不增强
    $0 augment_compare \\
        --name baseline --epochs 200 -- \\
        --name with_augment --augment --epochs 200

    # 三个任务串行
    $0 three_tasks \\
        --name task1 --batch-size 16 -- \\
        --name task2 --batch-size 32 -- \\
        --name task3 --batch-size 64

注意：
    - 所有训练将串行执行（第一个完成后才开始第二个）
    - 整个批量训练在后台运行，可以安全关闭终端
    - 统一日志文件：<batch_name>.log
    - 每个任务必须包含 --name 参数
    - 使用 tail -f <batch_name>.log 查看实时日志

EOF
}

# 检查参数
if [[ $# -lt 3 ]]; then
    show_help
    exit 1
fi

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# 获取批量训练名称
BATCH_NAME=$1
shift

# 设置日志文件
LOG_FILE="./${BATCH_NAME}.log"

# 解析所有任务参数（用 -- 分隔）
TASKS=()
CURRENT_TASK=""

for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        if [[ -n "$CURRENT_TASK" ]]; then
            TASKS+=("$CURRENT_TASK")
            CURRENT_TASK=""
        fi
    else
        if [[ -n "$CURRENT_TASK" ]]; then
            CURRENT_TASK="$CURRENT_TASK $arg"
        else
            CURRENT_TASK="$arg"
        fi
    fi
done

# 添加最后一个任务
if [[ -n "$CURRENT_TASK" ]]; then
    TASKS+=("$CURRENT_TASK")
fi

# 检查是否至少有一个任务
if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "❌ 错误：未找到任何训练任务"
    show_help
    exit 1
fi

# 构建串行训练命令（用 && 连接）
TRAIN_COMMANDS=""
for i in "${!TASKS[@]}"; do
    TASK_NUM=$((i + 1))
    TASK_ARGS="${TASKS[$i]}"

    # 验证每个任务是否包含 --name 参数
    if [[ ! "$TASK_ARGS" =~ --name ]]; then
        echo "❌ 错误：任务 $TASK_NUM 缺少 --name 参数"
        echo "   任务参数：$TASK_ARGS"
        exit 1
    fi

    # 构建命令
    if [[ $i -eq 0 ]]; then
        TRAIN_COMMANDS="python scripts/train.py $TASK_ARGS"
    else
        TRAIN_COMMANDS="$TRAIN_COMMANDS && python scripts/train.py $TASK_ARGS"
    fi
done

# 打印执行信息
echo "========================================="
echo "[$(date '+%F %T')] 启动批量串行训练"
echo "========================================="
echo "批量名称：${BATCH_NAME}"
echo "训练任务数：${#TASKS[@]}"
echo "日志文件：${LOG_FILE}"
echo ""
echo "训练任务列表："
for i in "${!TASKS[@]}"; do
    TASK_NUM=$((i + 1))
    echo "  [$TASK_NUM] python scripts/train.py ${TASKS[$i]}"
done
echo ""
echo "完整命令："
echo "  $TRAIN_COMMANDS"
echo "========================================="

# 创建临时脚本文件来执行串行训练
TEMP_SCRIPT="/tmp/yolo_batch_${BATCH_NAME}_$$.sh"
cat > "$TEMP_SCRIPT" << EOFSCRIPT
#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "[开始时间] \$(date '+%F %T')"
echo "批量训练：${BATCH_NAME}"
echo "========================================="

# 执行所有训练任务
$TRAIN_COMMANDS

EXIT_CODE=\$?

echo ""
echo "========================================="
echo "[结束时间] \$(date '+%F %T')"
if [[ \$EXIT_CODE -eq 0 ]]; then
    echo "✅ 批量训练全部完成：${BATCH_NAME}"
else
    echo "❌ 批量训练失败，退出码：\$EXIT_CODE"
fi
echo "========================================="

exit \$EXIT_CODE
EOFSCRIPT

chmod +x "$TEMP_SCRIPT"

# 使用 nohup 在后台运行，执行完成后自动删除临时脚本
nohup bash -c "$TEMP_SCRIPT && rm -f $TEMP_SCRIPT || (rm -f $TEMP_SCRIPT; exit 1)" > "${LOG_FILE}" 2>&1 &

# 获取进程 PID
PID=$!

# 打印成功信息
echo ""
echo "✅ 批量训练已在后台启动"
echo "   进程 PID：${PID}"
echo "   查看日志：tail -f ${LOG_FILE}"
echo "   停止训练：kill ${PID}"
echo "   任务顺序：串行执行（前一个完成后才开始下一个）"
echo "========================================="

# 等待 1 秒后检查进程是否还在运行
sleep 1
if ps -p ${PID} > /dev/null 2>&1; then
    echo "✅ 批量训练进程运行正常"
    echo ""
    echo "💡 提示："
    echo "   - 所有任务将串行执行"
    echo "   - 可以安全关闭终端"
    echo "   - 使用 'ps aux | grep train.py' 查看当前运行的任务"
else
    echo "❌ 进程启动失败，请检查日志文件：${LOG_FILE}"
    exit 1
fi

