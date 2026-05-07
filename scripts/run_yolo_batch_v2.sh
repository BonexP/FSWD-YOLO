#!/usr/bin/env bash
# run_yolo_batch_v2.sh - YOLO 模型批量串行训练脚本（增强日志版本）
# 改进：为每个任务添加明显的日志分隔符，便于查找和调试

set -euo pipefail

# 显示帮助文档
show_help() {
    cat << EOF
YOLO 模型批量串行训练脚本 (增强日志版本)

用法：
    $0 <batch_name> <task1_args> -- <task2_args> [-- <task3_args> ...]

参数说明：
    batch_name          批量训练的总名称（用于日志文件命名）
    task_args           每个训练任务的参数（与 train.py 参数一致）
    --                  分隔符，用于区分不同的训练任务

日志增强功能：
    ✅ 每个任务都有明显的开始/结束标记
    ✅ 记录每个任务的执行时间
    ✅ 显示任务名称和参数
    ✅ 更易于查找和调试

示例：
    # 对比两个学习率
    $0 lr_compare \\
        --name lr0.001 --lr0 0.001 --epochs 100 -- \\
        --name lr0.0005 --lr0 0.0005 --epochs 100

    # 查看日志中的任务边界
    grep "🚀\|✅\|❌" lr_compare.log

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


# 替换命令中的 TOTAL_TASKS
TOTAL_TASKS=${#TASKS[@]}
# 构建串行训练命令（用 && 连接，添加日志分隔符）
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

    # 提取任务名称（用于日志标记）
    TASK_NAME=$(echo "$TASK_ARGS" | grep -oP '(?<=--name )[^ ]+' || echo "task_$TASK_NUM")

    # 构建带分隔符的任务命令
    TASK_CMD="echo ''; echo '══════════════════════════════════════════════════════════════'; echo '🚀 [任务 $TASK_NUM/$TOTAL_TASKS] 开始: $TASK_NAME'; echo '开始时间: '\$(date '+%F %T'); echo '任务参数: python scripts/train.py $TASK_ARGS'; echo '══════════════════════════════════════════════════════════════'; echo ''"

    # 添加训练命令，记录开始时间
    TASK_CMD="$TASK_CMD && TASK_START=\$(date +%s) && python scripts/train.py $TASK_ARGS"

    # 添加任务完成标记，计算耗时
    TASK_CMD="$TASK_CMD && TASK_END=\$(date +%s) && TASK_DURATION=\$((TASK_END - TASK_START)) && echo '' && echo '══════════════════════════════════════════════════════════════' && echo '✅ [任务 $TASK_NUM/$TOTAL_TASKS] 完成: $TASK_NAME' && echo '结束时间: '\$(date '+%F %T') && echo '耗时: '\$TASK_DURATION' 秒 ('\$((TASK_DURATION / 60))' 分钟)' && echo '══════════════════════════════════════════════════════════════' && echo ''"

    # 连接到总命令
    if [[ $i -eq 0 ]]; then
        TRAIN_COMMANDS="$TASK_CMD"
    else
        TRAIN_COMMANDS="$TRAIN_COMMANDS && $TASK_CMD"
    fi
done

#TRAIN_COMMANDS="${TRAIN_COMMANDS//\$TOTAL_TASKS/$TOTAL_TASKS}"

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
    TASK_NAME=$(echo "${TASKS[$i]}" | grep -oP '(?<=--name )[^ ]+' || echo "task_$TASK_NUM")
    echo "  [$TASK_NUM] $TASK_NAME"
    echo "      python scripts/train.py ${TASKS[$i]}"
done
echo "========================================="

# 创建临时脚本文件来执行串行训练
TEMP_SCRIPT="/tmp/yolo_batch_${BATCH_NAME}_$$.sh"
cat > "$TEMP_SCRIPT" << EOFSCRIPT
#!/usr/bin/env bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          批量训练开始: ${BATCH_NAME}"
echo "║          开始时间: \$(date '+%F %T')"
echo "║          任务总数: ${TOTAL_TASKS}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 记录批量训练开始时间
BATCH_START=\$(date +%s)

# 执行所有训练任务
$TRAIN_COMMANDS

EXIT_CODE=\$?

# 计算批量训练总耗时
BATCH_END=\$(date +%s)
BATCH_DURATION=\$((BATCH_END - BATCH_START))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          批量训练结束: ${BATCH_NAME}"
echo "║          结束时间: \$(date '+%F %T')"
if [[ \$EXIT_CODE -eq 0 ]]; then
    echo "║          状态: ✅ 全部完成"
else
    echo "║          状态: ❌ 失败 (退出码: \$EXIT_CODE)"
fi
echo "║          总耗时: \$BATCH_DURATION 秒 (\$((BATCH_DURATION / 60)) 分钟)"
echo "╚══════════════════════════════════════════════════════════════╝"

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
echo "   日志文件：${LOG_FILE}"
echo ""
echo "📊 日志查看命令："
echo "   实时监控：tail -f ${LOG_FILE}"
echo "   查看进度：grep '🚀\\|✅\\|❌' ${LOG_FILE}"
echo "   查看错误：grep -i 'error\\|失败' ${LOG_FILE}"
echo ""
echo "🛑 停止训练："
echo "   kill ${PID}"
echo "========================================="

# 等待 1 秒后检查进程是否还在运行
sleep 1
if ps -p ${PID} > /dev/null 2>&1; then
    echo "✅ 批量训练进程运行正常"
    echo ""
    echo "💡 提示："
    echo "   - 所有任务将串行执行"
    echo "   - 可以安全关闭终端"
    echo "   - 每个任务都有明显的日志分隔符"
    echo "   - 使用 'ps aux | grep train.py' 查看当前任务"
else
    echo "❌ 进程启动失败，请检查日志文件：${LOG_FILE}"
    exit 1
fi

