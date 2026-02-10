# Prism: Spectral-Aware Block-Sparse Attention
This repository contains the official implementation of *Prism: Spectral-Aware Block-Sparse Attention*.

## Overview
Prism is a training-free method to accelerate long-context LLM pre-filling. It addresses the "blind spot" in standard mean pooling caused by Rotary Positional Embeddings (RoPE) by disentangling attention into high-frequency and low-frequency bands.

**Key Features:**
*   **Dual-Band Importance Estimation:** Separates semantic (low-freq) and positional (high-freq) signals.
*   **Energy-Based Calibration:** Restores attenuated signals automatically.
*   **Speed:** Up to **5.1× speedup** on 128K context with negligible accuracy loss.
*   **Implementation:** purely block-level operations with custom Triton kernels.

## Repository Structure
*   `prism/`
    *   `prism/prism.py`: Core implementation of Prism.
    *   `prism/kernels/`: Custom Triton kernels for efficient block importance estimation with Prism and block-sparse attention.
*   `eval/`: Evaluation harnesses.
*   `scripts/`: Shell scripts to reproduce the experiments in the paper.

## Installation
```bash
# For core Prism implementation only
uv pip install -e .

# For lm-eval evals
uv pip install -e "eval/lm-evaluation-harness["hf", "longbench", "ruler"]"

# For lmms-eval evals
uv pip install -e "eval/lmms-eval["qwen", "metrics"]"

# For baselines
# FlashAttention
uv pip install flash_attn --no-build-isolation
# Minference
uv pip install minference
# XAttention
git clone git@github.com:mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention && uv pip install -e .
```

## Example Usage
A simple example Prism using Qwen3-0.6B with a RULER exmaple.
```python
python -m prism.prism
```

## Evaluation
To reproduce the evaluation results, please refer to the scripts in the `scripts` directory.

```bash
# Example: Running LongBench Evaluation on Qwen3-8B
bash scripts/longbench.sh
```

**Note:** For RULER evaluation, we use Qwen3 with YaRN extrapolation, consistent with the official implementation. Please ensure your `MODEL_ID` points to a model path containing the [modified `config.json`](https://huggingface.co/Qwen/Qwen3-8B#processing-long-texts) required for long-context processing.