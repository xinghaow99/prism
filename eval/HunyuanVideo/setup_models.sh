#!/bin/bash
set -euo pipefail

# Download all HunyuanVideo model weights.
# Run from eval/HunyuanVideo/ directory.
# Total download: ~50GB (DiT ~25GB, text encoder ~16GB, CLIP ~1.5GB, VAE ~1GB)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

echo "=== Step 1/4: Download HunyuanVideo DiT + VAE ==="
hf download tencent/HunyuanVideo --local-dir ./ckpts

echo "=== Step 2/4: Download LLaVA text encoder ==="
hf download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava-llama-3-8b-v1_1-transformers

echo "=== Step 3/4: Preprocess text encoder ==="
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
    --input_dir ckpts/llava-llama-3-8b-v1_1-transformers \
    --output_dir ckpts/text_encoder

echo "=== Step 4/4: Download CLIP text encoder ==="
hf download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2

echo "=== Done! Model structure: ==="
find ckpts -maxdepth 3 -type d
