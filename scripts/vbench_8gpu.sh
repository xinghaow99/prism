#!/bin/bash
set -euo pipefail

# Run VBench evaluation on 8 GPUs in parallel.
# Each GPU processes a slice of prompts for one attention mode.
#
# Usage:
#   bash scripts/vbench_8gpu.sh              # dense + xattn, 200 prompts
#   NUM_PROMPTS=945 bash scripts/vbench_8gpu.sh  # full benchmark

source /mnt/bn/llm-liufangxu/wangxinghao_workspace/.bashrc
conda activate prism

cd /mnt/bn/llm-liufangxu/wangxinghao_workspace/projects/sparse_attention/prism_anonymous

export PYTORCH_ALLOC_CONF=expandable_segments:True

NUM_GPUS=${NUM_GPUS:-8}
NUM_PROMPTS=${NUM_PROMPTS:-200}
ATTN_MODE=${ATTN_MODE:-"both"}  # "flash", "xattn", or "both"
THRESHOLD=${THRESHOLD:-0.9}
STRIDE=${STRIDE:-8}
WARMUP_STEPS=${WARMUP_STEPS:-5}

CKPT_DIR="eval/HunyuanVideo/ckpts"
SAVE_PATH=${SAVE_PATH:-"results/vbench"}
LOG_DIR="logs/vbench"
mkdir -p "$LOG_DIR"

COMMON_ARGS="--save_path $SAVE_PATH --seed 0 \
    --model_base $CKPT_DIR \
    --dit_weight $CKPT_DIR/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --video_size 720 1280 --video_length 129 --infer_steps 50"

run_mode() {
    local mode=$1
    local extra_args=""
    if [ "$mode" = "xattn" ] || [ "$mode" = "prism" ]; then
        extra_args="--threshold $THRESHOLD --stride $STRIDE --attn_warmup_steps $WARMUP_STEPS"
    fi

    echo "=== Starting $mode on $NUM_GPUS GPUs ($NUM_PROMPTS prompts) ==="

    local per_gpu=$(( (NUM_PROMPTS + NUM_GPUS - 1) / NUM_GPUS ))
    local pids=()

    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        local start=$((gpu * per_gpu))
        local end=$(( start + per_gpu ))
        if [ $end -gt $NUM_PROMPTS ]; then end=$NUM_PROMPTS; fi
        if [ $start -ge $NUM_PROMPTS ]; then continue; fi

        echo "  GPU $gpu: prompts $start-$end"
        CUDA_VISIBLE_DEVICES=$gpu python eval/run_vbench.py \
            --attn "$mode" $extra_args \
            --prompt_start $start --prompt_end $end \
            $COMMON_ARGS \
            > "$LOG_DIR/${mode}_thr${THRESHOLD}_gpu${gpu}.log" 2>&1 &
        pids+=($!)
    done

    echo "  Waiting for $mode jobs (${#pids[@]} GPUs)..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: PID $pid failed"
            failed=$((failed + 1))
        fi
    done
    echo "=== $mode done ($failed failures) ==="
}

echo "================================================================"
echo "VBench 8-GPU Benchmark"
echo "Prompts: $NUM_PROMPTS | GPUs: $NUM_GPUS | Mode: $ATTN_MODE"
echo "Threshold: $THRESHOLD | Stride: $STRIDE | Warmup: $WARMUP_STEPS"
echo "================================================================"

if [ "$ATTN_MODE" = "flash" ] || [ "$ATTN_MODE" = "both" ]; then
    run_mode flash
fi

if [ "$ATTN_MODE" = "xattn" ] || [ "$ATTN_MODE" = "both" ]; then
    run_mode xattn
fi

if [ "$ATTN_MODE" = "prism" ]; then
    run_mode prism
fi

echo ""
echo "=== Results ==="
find "$SAVE_PATH" -name "*.mp4" | wc -l
echo "videos generated. Logs in $LOG_DIR/"
