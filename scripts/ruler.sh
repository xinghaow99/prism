#!/bin/bash

# Configuration for RULER evaluation with Prism Attention
NUM_SAMPLES=${NUM_SAMPLES:-100}
GPUS=${GPUS:-1}
# Default RULER lengths
MAX_SEQ_LENGTHS=${MAX_SEQ_LENGTHS:-"4096,8192,16384,32768,65536,131072"}

MODEL=${MODEL:-"Qwen/Qwen3-8B"} # Should be YaRN-extrapolated
# MODEL=${MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
TASKS=${TASKS:-"ruler"}
BATCH_SIZE=${BATCH_SIZE:-1}
# For RULER, we usually want a large max_length in model_args to accommodate the longest test sequence
MAX_LENGTH=${MAX_LENGTH:-131072}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-"true"}
LOG_SAMPLES=${LOG_SAMPLES:-"false"}
ENABLE_THINKING=${ENABLE_THINKING:-"false"}

EXTRA_ARGS=""
if [ "$APPLY_CHAT_TEMPLATE" == "true" ]; then
    EXTRA_ARGS="--apply_chat_template"
fi
if [ "$LOG_SAMPLES" == "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --log_samples"
fi

MODEL_ARGS="pretrained=$MODEL,trust_remote_code=True,max_length=$MAX_LENGTH"
if [ "$MODEL" == "Qwen/Qwen3-8B" ] || [[ "$MODEL" == *"Qwen3"* ]]; then
    MODEL_ARGS="$MODEL_ARGS,enable_thinking=$ENABLE_THINKING"
fi

export MODEL_ID=$MODEL

export PATCH_TYPE="prism"
export COLLECT_DENSITY=${COLLECT_DENSITY:-"true"}

# Prism Attention Hyperparameters
export LOW_FREQ_DIM=${LOW_FREQ_DIM:-96}
export HIGH_FREQ_DIM=${HIGH_FREQ_DIM:-64}
export BLOCK_SIZE=${BLOCK_SIZE:-128}
export LOW_FREQ_THRESHOLD=${LOW_FREQ_THRESHOLD:-0.95}
export HIGH_FREQ_THRESHOLD=${HIGH_FREQ_THRESHOLD:-0.95}
export CALIBRATE=${CALIBRATE:-"true"}

export USE_TRITON_SELECT=${USE_TRITON_SELECT:-"true"}
export USE_TRITON_LOGITS=${USE_TRITON_LOGITS:-"true"}

# Ensure comma separation if spaces are used
MAX_SEQ_LENGTHS_JSON="[${MAX_SEQ_LENGTHS// /,}]"

OUTPUT_NAME="${MODEL##*/}_${PATCH_TYPE}"
OUTPUT_DIR=${OUTPUT_DIR:-"results/${TASKS}/${OUTPUT_NAME}"}

echo "Running evaluation with Prism Attention"
echo "Model: $MODEL"
echo "Tasks: $TASKS"
echo "Max Seq Lengths: $MAX_SEQ_LENGTHS"
echo "Output Directory: $OUTPUT_DIR"

if [ $GPUS -gt 1 ]; then
    accelerate launch --main_process_port ${PORT0:-29500} --multi_gpu --num_processes $GPUS eval/run_lm_eval.py \
        --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks $TASKS \
        --batch_size $BATCH_SIZE \
        --metadata "{\"num_samples\": $NUM_SAMPLES, \"max_seq_lengths\": $MAX_SEQ_LENGTHS_JSON}" \
        $EXTRA_ARGS \
        --output_path "$OUTPUT_DIR"
else
    python3 eval/run_lm_eval.py \
        --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks $TASKS \
        --batch_size $BATCH_SIZE \
        --metadata "{\"num_samples\": $NUM_SAMPLES, \"max_seq_lengths\": $MAX_SEQ_LENGTHS_JSON}" \
        $EXTRA_ARGS \
        --output_path "$OUTPUT_DIR"
fi
