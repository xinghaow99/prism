#!/bin/bash
set -euo pipefail

# VBench evaluation script for HunyuanVideo with sparse attention.
#
# Usage:
#   ATTN_MODE=xattn THRESHOLD=0.95 bash scripts/vbench.sh
#   ATTN_MODE=flash bash scripts/vbench.sh   # dense baseline
#
# Environment variables:
#   ATTN_MODE       - flash (dense) or xattn (default: flash)
#   THRESHOLD       - top-p threshold for sparse attention (default: 0.95)
#   STRIDE          - XAttention estimation stride (default: 8)
#   WARMUP_STEPS    - dense warmup steps (default: 5)
#   BLOCK_SIZE      - sparse attention block size (default: 128)
#   VIDEO_HEIGHT    - video height in pixels (default: 720)
#   VIDEO_WIDTH     - video width in pixels (default: 1280)
#   VIDEO_LENGTH    - number of frames (default: 129)
#   INFER_STEPS     - denoising steps (default: 50)
#   NUM_PROMPTS     - max prompts to process (default: all)
#   PROMPT_START    - start prompt index (default: 0)
#   PROMPT_END      - end prompt index (default: -1, meaning all)
#   SEED            - random seed (default: 0)
#   MODEL_BASE      - path to model checkpoints (default: ckpts)
#   DIT_WEIGHT      - path to DiT weights
#   SAVE_PATH       - output directory (default: results/vbench)

export PYTORCH_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

# Attention configuration
ATTN_MODE=${ATTN_MODE:-"flash"}
THRESHOLD=${THRESHOLD:-0.9}  # Match XAttention repo default
STRIDE=${STRIDE:-8}
WARMUP_STEPS=${WARMUP_STEPS:-5}
BLOCK_SIZE=${BLOCK_SIZE:-128}

# Video configuration
VIDEO_HEIGHT=${VIDEO_HEIGHT:-720}
VIDEO_WIDTH=${VIDEO_WIDTH:-1280}
VIDEO_LENGTH=${VIDEO_LENGTH:-129}
INFER_STEPS=${INFER_STEPS:-50}

# Prompt range
PROMPT_START=${PROMPT_START:-0}
PROMPT_END=${PROMPT_END:--1}
SEED=${SEED:-0}

# Model paths
MODEL_BASE=${MODEL_BASE:-"ckpts"}
DIT_WEIGHT=${DIT_WEIGHT:-"ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"}

# Output
SAVE_PATH=${SAVE_PATH:-"results/vbench"}

echo "================================================================"
echo "VBench Evaluation — HunyuanVideo"
echo "Attention Mode: $ATTN_MODE"
echo "Threshold: $THRESHOLD"
echo "Stride: $STRIDE"
echo "Warmup Steps: $WARMUP_STEPS"
echo "Block Size: $BLOCK_SIZE"
echo "Video Size: ${VIDEO_HEIGHT}x${VIDEO_WIDTH}, ${VIDEO_LENGTH} frames"
echo "Inference Steps: $INFER_STEPS"
echo "Prompt Range: $PROMPT_START to $PROMPT_END"
echo "Seed: $SEED"
echo "Model Base: $MODEL_BASE"
echo "Save Path: $SAVE_PATH"
echo "================================================================"

python3 eval/run_vbench.py \
    --attn "$ATTN_MODE" \
    --threshold "$THRESHOLD" \
    --stride "$STRIDE" \
    --attn_warmup_steps "$WARMUP_STEPS" \
    --block_size "$BLOCK_SIZE" \
    --video_size "$VIDEO_HEIGHT" "$VIDEO_WIDTH" \
    --video_length "$VIDEO_LENGTH" \
    --infer_steps "$INFER_STEPS" \
    --prompt_start "$PROMPT_START" \
    --prompt_end "$PROMPT_END" \
    --seed "$SEED" \
    --model_base "$MODEL_BASE" \
    --dit_weight "$DIT_WEIGHT" \
    --save_path "$SAVE_PATH"

echo "================================================================"
echo "VBench evaluation completed successfully."
echo "================================================================"
