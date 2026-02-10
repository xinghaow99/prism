#!/bin/bash


GPUS=${GPUS:-1}
MODEL=${MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
TASKS=${TASKS:-"pg19"}
BATCH_SIZE=${BATCH_SIZE:-1}

CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-"4096 8192 16384 32768 65536 131072"}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-"false"}
LOG_SAMPLES=${LOG_SAMPLES:-"false"}
ENABLE_THINKING=${ENABLE_THINKING:-"false"}

EXTRA_ARGS=""
if [ "$APPLY_CHAT_TEMPLATE" == "true" ]; then
    EXTRA_ARGS="--apply_chat_template"
fi
if [ "$LOG_SAMPLES" == "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --log_samples"
fi

BASE_MODEL_ARGS="pretrained=$MODEL,trust_remote_code=True"
if [ "$MODEL" == "Qwen/Qwen3-8B" ] || [[ "$MODEL" == *"Qwen3"* ]]; then
    BASE_MODEL_ARGS="$BASE_MODEL_ARGS,enable_thinking=$ENABLE_THINKING"
fi

export MODEL_ID=$MODEL

export PATCH_TYPE="prism" # none, minference, flexprefill, xattn, prism
export COLLECT_DENSITY=${COLLECT_DENSITY:-"true"}

export LOW_FREQ_DIM=${LOW_FREQ_DIM:-96}
export HIGH_FREQ_DIM=${HIGH_FREQ_DIM:-64}
export BLOCK_SIZE=${BLOCK_SIZE:-128}
export LOW_FREQ_THRESHOLD=${LOW_FREQ_THRESHOLD:-0.92}
export HIGH_FREQ_THRESHOLD=${HIGH_FREQ_THRESHOLD:-0.92}
export CALIBRATE=${CALIBRATE:-"true"}

export USE_TRITON_SELECT=${USE_TRITON_SELECT:-"true"}
export USE_TRITON_LOGITS=${USE_TRITON_LOGITS:-"true"}

OUTPUT_ROOT=${OUTPUT_DIR:-"results/${TASKS}"}
OUTPUT_NAME="${MODEL##*/}_${PATCH_TYPE}"

echo "Running PG19 evaluation with Prism Attention"
echo "Model: $MODEL"
echo "Tasks: $TASKS"
echo "Context Lengths: $CONTEXT_LENGTHS"
echo "Output Root: $OUTPUT_ROOT"

for MAX_LENGTH in $CONTEXT_LENGTHS; do
    echo "=========================================================="
    echo "Evaluating with Context Length: $MAX_LENGTH"
    echo "=========================================================="

    MODEL_ARGS="${BASE_MODEL_ARGS},max_length=$MAX_LENGTH"

    CUR_OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_NAME}/Len${MAX_LENGTH}"

    if [ $GPUS -gt 1 ]; then
        accelerate launch --main_process_port ${PORT0:-29500} --multi_gpu --num_processes $GPUS eval/run_lm_eval.py \
            --model hf \
            --model_args "${MODEL_ARGS}" \
            --tasks $TASKS \
            --batch_size $BATCH_SIZE \
            $EXTRA_ARGS \
            --output_path "$CUR_OUTPUT_DIR"
    else
        python3 eval/run_lm_eval.py \
            --model hf \
            --model_args "${MODEL_ARGS}" \
            --tasks $TASKS \
            --batch_size $BATCH_SIZE \
            $EXTRA_ARGS \
            --output_path "$CUR_OUTPUT_DIR"
    fi

    echo "Finished evaluation for length $MAX_LENGTH. Results saved in $CUR_OUTPUT_DIR"
done
