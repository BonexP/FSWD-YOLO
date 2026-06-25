#!/usr/bin/env bash
# Revision 70/15/15 fixed-split training launcher.
#
# Run from the FSWD-YOLO repository root on the A100 host:
#   bash scripts/revision_train_70_15_15.sh
#
# Common overrides:
#   TASK_SET=main SEEDS="0 1 2" DEVICE=0 bash scripts/revision_train_70_15_15.sh
#   TASK_SET=workset SEEDS="0 1 2 3 4" DEVICE=0 bash scripts/revision_train_70_15_15.sh
#   TASK_SET=cherry-pick SEEDS="0 1 2" DEVICE=0 bash scripts/revision_train_70_15_15.sh
#   TASK_SET=ablation SEEDS=0 bash scripts/revision_train_70_15_15.sh
#   DRY_RUN=1 bash scripts/revision_train_70_15_15.sh

set -euo pipefail

DATA="${DATA:-/home/user/PROJECT/FSWD/FSW-MERGE_revision_dataset_augmented_double/data.yaml}"
TASK_SET="${TASK_SET:-main}"
SEEDS="${SEEDS:-0 1 2}"
PROJECT="${PROJECT:-runs/revision_70_15_15_${TASK_SET}}"
BATCH_NAME="${BATCH_NAME:-revision_70_15_15_${TASK_SET}}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${WORKERS:-16}"
CACHE="${CACHE:-ram}"
OPTIMIZER="${OPTIMIZER:-Adam}"
LR0="${LR0:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
MOMENTUM="${MOMENTUM:-0.937}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-10}"
AUGMENT="${AUGMENT:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f scripts/train.py ]]; then
    echo "ERROR: run this script from the FSWD-YOLO repository root." >&2
    exit 1
fi

if [[ ! -f scripts/run_yolo_batch_v2.sh ]]; then
    echo "ERROR: missing scripts/run_yolo_batch_v2.sh." >&2
    exit 1
fi

if [[ ! -f "${DATA}" ]]; then
    echo "ERROR: dataset YAML not found: ${DATA}" >&2
    exit 1
fi

case "${TASK_SET}" in
    main|workset|cherry-pick|ablation|all)
        ;;
    *)
        echo "ERROR: TASK_SET must be one of: main, workset, cherry-pick, ablation, all." >&2
        exit 1
        ;;
esac

COMMON_ARGS=(
    --cfg "${DATA}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --img-size "${IMG_SIZE}"
    --device "${DEVICE}"
    --workers "${WORKERS}"
    --optimizer "${OPTIMIZER}"
    --lr0 "${LR0}"
    --weight-decay "${WEIGHT_DECAY}"
    --momentum "${MOMENTUM}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --close-mosaic "${CLOSE_MOSAIC}"
    --project "${PROJECT}"
)

if [[ -n "${CACHE}" ]]; then
    COMMON_ARGS+=(--cache "${CACHE}")
fi

if [[ "${AUGMENT}" == "1" ]]; then
    COMMON_ARGS+=(--augment)
fi

TASKS=()

join_args() {
    local out=""
    local arg
    for arg in "$@"; do
        if [[ -z "${out}" ]]; then
            out="${arg}"
        else
            out="${out} ${arg}"
        fi
    done
    printf '%s' "${out}"
}

model_exists() {
    local model_cfg="$1"
    [[ -f "${model_cfg}" ]]
}

add_task() {
    local name="$1"
    local model_cfg="$2"
    local seed="$3"
    local iou_type="$4"
    local weighted="$5"

    if ! model_exists "${model_cfg}"; then
        echo "ERROR: model config not found: ${model_cfg}" >&2
        exit 1
    fi

    local args=("${COMMON_ARGS[@]}")
    args+=(
        --model "${model_cfg}"
        --name "${name}_seed${seed}"
        --seed "${seed}"
        --iou-type "${iou_type}"
    )
    if [[ "${weighted}" == "1" ]]; then
        args+=(--weighted-dataloader)
    fi
    TASKS+=("$(join_args "${args[@]}")")
}

add_main_tasks_for_seed() {
    local seed="$1"
    add_task "main_yolov8s" "ultralytics/cfg/models/v8/yolov8s.yaml" "${seed}" "CIoU" "1"
    add_task "main_yolov9s" "ultralytics/cfg/models/v9/yolov9s.yaml" "${seed}" "CIoU" "1"
    add_task "main_yolov10s" "ultralytics/cfg/models/v10/yolov10s.yaml" "${seed}" "CIoU" "1"
    add_task "main_yolo11s" "ultralytics/cfg/models/11/yolo11s.yaml" "${seed}" "CIoU" "1"
    add_task "main_fswd_yolo" "ultralytics/cfg/models/11/fswd-yolo.yaml" "${seed}" "ShapeIoU" "1"
}

add_workset_tasks_for_seed() {
    local seed="$1"
    add_task "main_yolov8s" "ultralytics/cfg/models/v8/yolov8s.yaml" "${seed}" "CIoU" "1"
    add_task "main_yolov9s" "ultralytics/cfg/models/v9/yolov9s.yaml" "${seed}" "CIoU" "1"
    add_task "main_fswd_yolo" "ultralytics/cfg/models/11/fswd-yolo.yaml" "${seed}" "ShapeIoU" "1"
}

add_cherry_pick_tasks_for_seed() {
    local seed="$1"
    add_workset_tasks_for_seed "${seed}"
}

add_ablation_tasks_for_seed() {
    local seed="$1"
    add_task "abl_ghost_fca_ciou" \
        "ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA.yaml" "${seed}" "CIoU" "1"
    add_task "abl_ghostsimam_fca_ciou" \
        "ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA.yaml" "${seed}" "CIoU" "1"
    add_task "abl_ghost_vov_ciou" \
        "ultralytics/cfg/models/11/yolo11s_C3k2Ghost_VoVCsingle.yaml" "${seed}" "CIoU" "1"
    add_task "abl_ghostsimam_vov_ciou" \
        "ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_VoVCsingle.yaml" "${seed}" "CIoU" "1"
    add_task "abl_fca_vov_ciou" \
        "ultralytics/cfg/models/11/yolo11s_C2PSFCA_VoVCsingle.yaml" "${seed}" "CIoU" "1"
    add_task "abl_ghost_fca_vov_ciou" \
        "ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml" "${seed}" "CIoU" "1"
    add_task "abl_ghost_fca_vov_shapeiou" \
        "ultralytics/cfg/models/11/yolo11s_C3k2Ghost_C2PSFCA_VoVCsingle.yaml" "${seed}" "ShapeIoU" "1"
    add_task "abl_ghostsimam_fca_vov_ciou_noweighted" \
        "ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml" "${seed}" "CIoU" "0"
    add_task "abl_ghostsimam_fca_vov_ciou" \
        "ultralytics/cfg/models/11/yolo11s_C3k2GhostSimAMinner_C2PSFCA_VoVCsingle.yaml" "${seed}" "CIoU" "1"
}

for seed in ${SEEDS}; do
    if [[ "${TASK_SET}" == "main" || "${TASK_SET}" == "all" ]]; then
        add_main_tasks_for_seed "${seed}"
    fi
    if [[ "${TASK_SET}" == "workset" ]]; then
        add_workset_tasks_for_seed "${seed}"
    fi
    if [[ "${TASK_SET}" == "cherry-pick" ]]; then
        add_cherry_pick_tasks_for_seed "${seed}"
    fi
    if [[ "${TASK_SET}" == "ablation" || "${TASK_SET}" == "all" ]]; then
        add_ablation_tasks_for_seed "${seed}"
    fi
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "ERROR: no training tasks were generated." >&2
    exit 1
fi

mkdir -p "${PROJECT}"

echo "============================================================"
echo "Revision fixed-split training launcher"
echo "============================================================"
echo "Dataset YAML : ${DATA}"
echo "Task set     : ${TASK_SET}"
echo "Seeds        : ${SEEDS}"
echo "Project      : ${PROJECT}"
echo "Batch name   : ${BATCH_NAME}"
echo "Device       : ${DEVICE}"
echo "Epochs       : ${EPOCHS}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Image size   : ${IMG_SIZE}"
echo "Augment      : ${AUGMENT}"
echo "Dry run      : ${DRY_RUN}"
echo "Task count   : ${#TASKS[@]}"
echo "============================================================"

cmd=(bash scripts/run_yolo_batch_v2.sh "${BATCH_NAME}")
for idx in "${!TASKS[@]}"; do
    if [[ "${idx}" != "0" ]]; then
        cmd+=(--)
    fi
    # run_yolo_batch_v2.sh expects each task as normal shell tokens. The script
    # below intentionally relies on word splitting for generated task strings.
    # shellcheck disable=SC2206
    task_tokens=(${TASKS[$idx]})
    cmd+=("${task_tokens[@]}")
done

printf 'Launch command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1, command was not launched."
    exit 0
fi

"${cmd[@]}"
