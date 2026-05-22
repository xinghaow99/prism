#!/bin/bash
set -euo pipefail

# VBench video-generation evaluation entrypoint for HunyuanVideo.
#
# Usage:
#   bash scripts/vbench.sh
#   GPUS=8 ATTN_MODE=prism THRESHOLD=0.95 bash scripts/vbench.sh
#   ATTN_MODE=flash PROMPT_START=0 PROMPT_END=100 bash scripts/vbench.sh
#
# Environment variables:
#   GPUS              - number of GPUs for prompt sharding (default: 1)
#   ATTN_MODE         - flash, xattn, or prism (default: prism)
#   THRESHOLD         - top-p threshold for sparse attention (default: 0.95)
#   STRIDE            - XAttention estimation stride (default: 8)
#   WARMUP_STEPS      - dense warmup denoising steps (default: 5)
#   BLOCK_SIZE        - sparse attention block size (default: 128)
#   PROMPT_FILE       - prompt file path, relative to repo root (default: eval/HunyuanVideo/all_dimension_longer.txt)
#   PROMPT_START      - start prompt index, inclusive (default: 0)
#   PROMPT_END        - end prompt index, exclusive (default: -1, meaning all)
#   NUM_PROMPTS       - optional cap after PROMPT_START when PROMPT_END is unset
#   SEED              - random seed (default: 0)
#   NUM_SEED          - number of seeds per prompt (default: 1)
#   VIDEO_HEIGHT      - video height in pixels (default: 720)
#   VIDEO_WIDTH       - video width in pixels (default: 1280)
#   VIDEO_LENGTH      - number of frames (default: 129)
#   INFER_STEPS       - denoising steps (default: 50)
#   MODEL_BASE        - HunyuanVideo checkpoint root (default: eval/HunyuanVideo/ckpts)
#   DIT_WEIGHT        - DiT checkpoint path (default: $MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt)
#   SAVE_PATH         - output directory (default: results/vbench)
#   LOG_DIR           - per-GPU log directory for GPUS>1 (default: logs/vbench)

export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

GPUS=${GPUS:-1}
ATTN_MODE=${ATTN_MODE:-${PATCH_TYPE:-prism}}
THRESHOLD=${THRESHOLD:-0.95}
STRIDE=${STRIDE:-8}
WARMUP_STEPS=${WARMUP_STEPS:-5}
BLOCK_SIZE=${BLOCK_SIZE:-128}

PROMPT_FILE=${PROMPT_FILE:-"eval/HunyuanVideo/all_dimension_longer.txt"}
PROMPT_START=${PROMPT_START:-0}
PROMPT_END=${PROMPT_END:--1}
NUM_PROMPTS=${NUM_PROMPTS:-}
SEED=${SEED:-0}
NUM_SEED=${NUM_SEED:-1}

VIDEO_HEIGHT=${VIDEO_HEIGHT:-720}
VIDEO_WIDTH=${VIDEO_WIDTH:-1280}
VIDEO_LENGTH=${VIDEO_LENGTH:-129}
INFER_STEPS=${INFER_STEPS:-50}

MODEL_BASE=${MODEL_BASE:-"eval/HunyuanVideo/ckpts"}
DIT_WEIGHT=${DIT_WEIGHT:-"${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"}

SAVE_PATH=${SAVE_PATH:-"results/vbench"}
LOG_DIR=${LOG_DIR:-"logs/vbench"}

if [ "$ATTN_MODE" != "flash" ] && [ "$ATTN_MODE" != "xattn" ] && [ "$ATTN_MODE" != "prism" ]; then
    echo "Unsupported ATTN_MODE: $ATTN_MODE. Expected one of: flash, xattn, prism." >&2
    exit 1
fi

case "$PROMPT_FILE" in
    /*) PROMPT_FILE_ABS="$PROMPT_FILE" ;;
    *) PROMPT_FILE_ABS="$ROOT_DIR/$PROMPT_FILE" ;;
esac
if [ ! -f "$PROMPT_FILE_ABS" ]; then
    echo "Prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

if [ "$PROMPT_END" -lt 0 ]; then
    TOTAL_PROMPTS=$(grep -cve '^[[:space:]]*$' "$PROMPT_FILE_ABS")
    if [ -n "$NUM_PROMPTS" ]; then
        PROMPT_END=$((PROMPT_START + NUM_PROMPTS))
        if [ "$PROMPT_END" -gt "$TOTAL_PROMPTS" ]; then
            PROMPT_END=$TOTAL_PROMPTS
        fi
    else
        PROMPT_END=$TOTAL_PROMPTS
    fi
fi

if [ "$PROMPT_END" -le "$PROMPT_START" ]; then
    echo "Invalid prompt range: PROMPT_START=$PROMPT_START PROMPT_END=$PROMPT_END" >&2
    exit 1
fi

COMMON_ARGS=(
    --attn "$ATTN_MODE"
    --threshold "$THRESHOLD"
    --stride "$STRIDE"
    --attn_warmup_steps "$WARMUP_STEPS"
    --block_size "$BLOCK_SIZE"
    --prompt_file "$PROMPT_FILE"
    --save_path "$SAVE_PATH"
    --seed "$SEED"
    --num_seed "$NUM_SEED"
    --video_size "$VIDEO_HEIGHT" "$VIDEO_WIDTH"
    --video_length "$VIDEO_LENGTH"
    --infer_steps "$INFER_STEPS"
    --model_base "$MODEL_BASE"
    --dit_weight "$DIT_WEIGHT"
)

echo "================================================================"
echo "VBench Video-Generation Evaluation"
echo "Attention Mode: $ATTN_MODE"
echo "Threshold: $THRESHOLD"
echo "Stride: $STRIDE"
echo "Warmup Steps: $WARMUP_STEPS"
echo "Block Size: $BLOCK_SIZE"
echo "Prompt Range: $PROMPT_START to $PROMPT_END"
echo "GPUs: $GPUS"
echo "Video: ${VIDEO_HEIGHT}x${VIDEO_WIDTH}, ${VIDEO_LENGTH} frames, ${INFER_STEPS} steps"
echo "Model Base: $MODEL_BASE"
echo "DiT Weight: $DIT_WEIGHT"
echo "Save Path: $SAVE_PATH"
echo "================================================================"

if [ "$GPUS" -gt 1 ]; then
    mkdir -p "$LOG_DIR"

    TOTAL_RANGE=$((PROMPT_END - PROMPT_START))
    PER_GPU=$(((TOTAL_RANGE + GPUS - 1) / GPUS))
    PIDS=()

    for GPU in $(seq 0 $((GPUS - 1))); do
        START=$((PROMPT_START + GPU * PER_GPU))
        END=$((START + PER_GPU))

        if [ "$START" -ge "$PROMPT_END" ]; then
            continue
        fi
        if [ "$END" -gt "$PROMPT_END" ]; then
            END=$PROMPT_END
        fi

        LOG_FILE="$LOG_DIR/${ATTN_MODE}_thr${THRESHOLD}_gpu${GPU}.log"
        echo "GPU $GPU: prompts $START-$END -> $LOG_FILE"
        CUDA_VISIBLE_DEVICES=$GPU python3 eval/run_vbench.py \
            "${COMMON_ARGS[@]}" \
            --prompt_start "$START" \
            --prompt_end "$END" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done

    FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            FAILED=$((FAILED + 1))
        fi
    done

    if [ "$FAILED" -gt 0 ]; then
        echo "$FAILED VBench worker(s) failed. Check $LOG_DIR for details." >&2
        exit 1
    fi
else
    python3 eval/run_vbench.py \
        "${COMMON_ARGS[@]}" \
        --prompt_start "$PROMPT_START" \
        --prompt_end "$PROMPT_END"
fi

echo "================================================================"
echo "VBench video-generation evaluation completed successfully."
echo "================================================================"
