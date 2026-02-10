#!/bin/bash
set -euo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"


MODEL_ID=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
SEQ_LENS=${2:-"8192,16384,32768,65536,131072"}
BATCH_SIZE=${3:-1}
LOW_FREQ_DIM=${5:-96}
HIGH_FREQ_DIM=${6:-64}
BLOCK_SIZE=${7:-128}
LOW_FREQ_THRESHOLD=${8:-0.95}
HIGH_FREQ_THRESHOLD=${9:-0.95}
CALIBRATE=${CALIBRATE:-"true"}
USE_TRITON_LOGITS=${USE_TRITON_LOGITS:-"true"}
USE_TRITON_SELECT=${39:-"true"}

DATA_PATH=${13:-"data/ruler_qa_squad_32k_0.json"}
DATA_IDX=${14:-"0"}

FLEX_PREFILL_GAMMA=${30:-0.95}
FLEX_PREFILL_TAU=${31:-0.1}
FLEX_PREFILL_MIN_BUDGET=${32:-""}
FLEX_PREFILL_MAX_BUDGET=${33:-""}
FLEX_PREFILL_GQA_INTERLEAVE=${34:-"false"}

MINFERENCE_VERTICAL_SIZE=${35:-512}
MINFERENCE_SLASH_SIZE=${36:-2048}
MINFERENCE_ADAPTIVE_BUDGET=${37:-""}

ATTN_ONLY_LAYER_IDX=${41:-"12"}
NO_DENSITY=${44:-"true"}
NO_SELECT_TIME=${45:-"true"}

INCLUDE_FLASH=${48:-"true"}
INCLUDE_MINFERENCE=${49:-"true"}
INCLUDE_FLEXPREFILL=${50:-"true"}
INCLUDE_XATTENTION=${51:-"true"}
INCLUDE_PRISM=${52:-"true"}

XATTN_STRIDE=${XATTN_STRIDE:-8}
XATTN_THRESHOLD=${XATTN_THRESHOLD:-0.9}
FORCE_SINK=${FORCE_SINK:-"true"}
FORCE_RECENT=${FORCE_RECENT:-"true"}

echo "================================================================"
echo "Starting Efficiency Comparison Benchmark"
echo "Model: $MODEL_ID"
echo "Sequence Lengths: $SEQ_LENS"
echo "Batch Size: $BATCH_SIZE"
echo "Data Path: $DATA_PATH"
echo "Data Index: $DATA_IDX"
echo "Block Size: $BLOCK_SIZE"
echo "Low Freq Dim: $LOW_FREQ_DIM"
echo "High Freq Dim: $HIGH_FREQ_DIM"
echo "Low Freq Threshold: $LOW_FREQ_THRESHOLD"
echo "High Freq Threshold: $HIGH_FREQ_THRESHOLD"
echo "Force Sink: $FORCE_SINK"
echo "Force Recent: $FORCE_RECENT"
echo "Calibrate: $CALIBRATE"
echo "Use Triton Select: $USE_TRITON_SELECT"
echo "Use Triton Logits: $USE_TRITON_LOGITS"
echo "XAttention Stride: $XATTN_STRIDE"
echo "XAttention Threshold: $XATTN_THRESHOLD"
echo "FlexPrefill Gamma: $FLEX_PREFILL_GAMMA"
echo "FlexPrefill Tau: $FLEX_PREFILL_TAU"
echo "FlexPrefill Min Budget: $FLEX_PREFILL_MIN_BUDGET"
echo "FlexPrefill Max Budget: $FLEX_PREFILL_MAX_BUDGET"
echo "FlexPrefill GQA Interleave: $FLEX_PREFILL_GQA_INTERLEAVE"
echo "Minference Vertical Size: $MINFERENCE_VERTICAL_SIZE"
echo "Minference Slash Size: $MINFERENCE_SLASH_SIZE"
echo "Minference Adaptive Budget: $MINFERENCE_ADAPTIVE_BUDGET"
echo "Attention Only Layer Idx: $ATTN_ONLY_LAYER_IDX"
echo "Disable Density: $NO_DENSITY"
echo "Disable Select Time: $NO_SELECT_TIME"
echo "Include FlashAttention: $INCLUDE_FLASH"
echo "Include Minference: $INCLUDE_MINFERENCE"
echo "Include FlexPrefill: $INCLUDE_FLEXPREFILL"
echo "Include XAttention: $INCLUDE_XATTENTION"
echo "Include Prism: $INCLUDE_PRISM"
echo "================================================================"

ARGS=(
    --model_id "$MODEL_ID"
    --seq_lens "$SEQ_LENS"
    --batch_size "$BATCH_SIZE"
    --data_path "$DATA_PATH"
    --data_idx "$DATA_IDX"
    --block_size "$BLOCK_SIZE"
    --low_freq_dim "$LOW_FREQ_DIM"
    --high_freq_dim "$HIGH_FREQ_DIM"
    --low_freq_threshold "$LOW_FREQ_THRESHOLD"
    --high_freq_threshold "$HIGH_FREQ_THRESHOLD"
    --xattn_stride "$XATTN_STRIDE"
    --xattn_threshold "$XATTN_THRESHOLD"
    --flexprefill_gamma "$FLEX_PREFILL_GAMMA"
    --flexprefill_tau "$FLEX_PREFILL_TAU"
    --minference_vertical_size "$MINFERENCE_VERTICAL_SIZE"
    --minference_slash_size "$MINFERENCE_SLASH_SIZE"
    --num_warmup 3
    --num_runs 5
)

if [ "$FORCE_SINK" = "true" ]; then
    ARGS+=(--force_sink)
else
    ARGS+=(--no_force_sink)
fi
if [ "$FORCE_RECENT" = "true" ]; then
    ARGS+=(--force_recent)
else
    ARGS+=(--no_force_recent)
fi
if [ "$CALIBRATE" = "true" ]; then
    ARGS+=(--calibrate)
else
    ARGS+=(--no_calibrate)
fi
if [ "$USE_TRITON_SELECT" = "true" ]; then
    ARGS+=(--use_triton_select)
else
    ARGS+=(--no_use_triton_select)
fi
if [ "$USE_TRITON_LOGITS" = "true" ]; then
    ARGS+=(--use_triton_logits)
else
    ARGS+=(--no_use_triton_logits)
fi

if [ -n "$FLEX_PREFILL_MIN_BUDGET" ]; then
    ARGS+=(--flexprefill_min_budget "$FLEX_PREFILL_MIN_BUDGET")
fi
if [ -n "$FLEX_PREFILL_MAX_BUDGET" ]; then
    ARGS+=(--flexprefill_max_budget "$FLEX_PREFILL_MAX_BUDGET")
fi
if [ "$FLEX_PREFILL_GQA_INTERLEAVE" = "true" ]; then
    ARGS+=(--flexprefill_gqa_interleave)
fi
if [ -n "$MINFERENCE_ADAPTIVE_BUDGET" ]; then
    ARGS+=(--minference_adaptive_budget "$MINFERENCE_ADAPTIVE_BUDGET")
fi
if [ -n "$ATTN_ONLY_LAYER_IDX" ]; then
    ARGS+=(--attn_only_layer_idx "$ATTN_ONLY_LAYER_IDX")
fi
if [ "$NO_DENSITY" = "true" ]; then
    ARGS+=(--no_density)
fi
if [ "$NO_SELECT_TIME" = "true" ]; then
    ARGS+=(--no_select_time)
fi

METHODS=""
if [ "$INCLUDE_FLASH" = "true" ]; then
    METHODS="FlashAttention"
fi
if [ "$INCLUDE_MINFERENCE" = "true" ]; then
    if [ -n "$METHODS" ]; then METHODS="$METHODS,Minference"; else METHODS="Minference"; fi
fi
if [ "$INCLUDE_FLEXPREFILL" = "true" ]; then
    if [ -n "$METHODS" ]; then METHODS="$METHODS,FlexPrefill"; else METHODS="FlexPrefill"; fi
fi
if [ "$INCLUDE_XATTENTION" = "true" ]; then
    if [ -n "$METHODS" ]; then METHODS="$METHODS,XAttention"; else METHODS="XAttention"; fi
fi
if [ "$INCLUDE_PRISM" = "true" ]; then
    if [ -n "$METHODS" ]; then METHODS="$METHODS,Prism"; else METHODS="Prism"; fi
fi
if [ -n "$METHODS" ]; then
    ARGS+=(--methods "$METHODS")
fi

python3 eval/efficiency_comparison.py "${ARGS[@]}"
echo "================================================================"
echo "Benchmark completed successfully."
