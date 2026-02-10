#!/bin/bash

GPUS=${GPUS:-8}
MODEL=${MODEL:-"Qwen/Qwen3-VL-8B-Instruct"}
TASKS=${TASKS:-"longvideobench_val_v"}
BATCH_SIZE=${BATCH_SIZE:-1}

# Vision Specific Configuration
MAX_PIXELS=${MAX_PIXELS:-327680}
MIN_PIXELS=${MIN_PIXELS:-327680}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-512}
FPS=${FPS:-1}

# lmms-eval specific model type
LMMS_MODEL=${LMMS_MODEL:-"qwen3_vl"}

LOG_SAMPLES=${LOG_SAMPLES:-"false"}
EXTRA_ARGS=""
if [ "$LOG_SAMPLES" == "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --log_samples"
fi

MODEL_ARGS="pretrained=$MODEL,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,max_num_frames=$MAX_NUM_FRAMES"
if [ -n "$FPS" ]; then
    MODEL_ARGS="$MODEL_ARGS,fps=$FPS"
fi

export MODEL_ID=$MODEL

export PATCH_TYPE="prism" # none, minference, flexprefill, xattn, prism
export COLLECT_DENSITY=${COLLECT_DENSITY:-"true"}

export LOW_FREQ_DIM=${LOW_FREQ_DIM:-96}
export HIGH_FREQ_DIM=${HIGH_FREQ_DIM:-64}
export BLOCK_SIZE=${BLOCK_SIZE:-128}
export LOW_FREQ_THRESHOLD=${LOW_FREQ_THRESHOLD:-0.93}
export HIGH_FREQ_THRESHOLD=${HIGH_FREQ_THRESHOLD:-0.93}
export CALIBRATE=${CALIBRATE:-"true"}

export USE_TRITON_SELECT=${USE_TRITON_SELECT:-"true"}
export USE_TRITON_LOGITS=${USE_TRITON_LOGITS:-"true"}

# Handle output naming
TASK_TAG=${TASKS//,/+}
OUTPUT_NAME="${MODEL##*/}_${PATCH_TYPE}"
OUTPUT_DIR=${OUTPUT_DIR:-"results/${TASK_TAG}/${OUTPUT_NAME}"}

echo "Running LongVideoBench evaluation with Prism Attention"
echo "Model: $MODEL"
echo "Tasks: $TASKS"
echo "Output Directory: $OUTPUT_DIR"

if [ $GPUS -gt 1 ]; then
    accelerate launch --main_process_port ${PORT0:-29500} --multi_gpu --num_processes $GPUS eval/run_lmms_eval.py \
        --model $LMMS_MODEL \
        --model_args "${MODEL_ARGS}" \
        --tasks $TASKS \
        --batch_size $BATCH_SIZE \
        $EXTRA_ARGS \
        --output_path "$OUTPUT_DIR"
else
    python3 eval/run_lmms_eval.py \
        --model $LMMS_MODEL \
        --model_args "${MODEL_ARGS}" \
        --tasks $TASKS \
        --batch_size $BATCH_SIZE \
        $EXTRA_ARGS \
        --output_path "$OUTPUT_DIR"
fi
