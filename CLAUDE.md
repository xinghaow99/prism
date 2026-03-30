# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Prism is a training-free method to accelerate long-context LLM pre-filling via spectral-aware block-sparse attention. It separates attention into high-frequency (positional) and low-frequency (semantic) bands using RoPE structure, then selects important blocks per band using top-p thresholding. Works by monkey-patching transformer attention classes at runtime.

## Build & Install

```bash
uv pip install -e .                                    # Core package
uv pip install -e "eval/lm-evaluation-harness[vllm]"   # Eval harness (if running evals)
```

Dependencies: torch, transformers>=5.0.0, accelerate, triton (for GPU kernels).

## Running Tests

```bash
python -m pytest tests/test_kernels_gpu.py -v   # GPU kernel unit tests (requires GPU)
python -m prism.prism                            # Quick sanity check with Qwen3-0.6B
```

SGLang integration tests: `tests/test_sglang_prism.py`, `tests/smoke_test_sglang.py`.

## Running Evaluations

All evaluation scripts are in `scripts/` and invoke `eval/run_lm_eval.py`:

```bash
# Example: LongBench with Prism
MODEL="Qwen/Qwen3-8B" GPUS=8 bash scripts/longbench.sh

# Available scripts: ruler.sh, longbench.sh, pg19.sh, longvideobench.sh, videomme.sh, efficiency.sh
# PATCH_TYPE options: none, prism, xattn, flexprefill, minference
```

Results are saved to `results/{task}/{model}_{patch_type}/`.

## Architecture

### Core Algorithm (`prism/prism.py`)

- **`prism_attention_forward()`** — Replaces standard attention forward. Handles RoPE, KV caching, GQA. Falls back to dense attention for decoding (q_len=1). Pads sequences to block-aligned lengths.
- **`prism_block_estimate()`** — Block mask estimation pipeline:
  1. Mean-pool Q/K to block level
  2. Extract high-freq (edge indices) and low-freq (center indices) components from RoPE dimension layout
  3. Energy-based calibration (RMS scaling to restore attenuated signals)
  4. Dual-band softmax → top-p block selection
  5. Force sink/recent block constraints

### Configuration via Environment Variables

All Prism parameters are set via env vars (read at import time in `prism/prism.py`):

| Variable | Default | Description |
|---|---|---|
| `LOW_FREQ_DIM` | 96 | Low-frequency band dimension |
| `HIGH_FREQ_DIM` | 64 | High-frequency band dimension |
| `BLOCK_SIZE` | 128 | Attention block size |
| `LOW_FREQ_THRESHOLD` | 0.95 | Top-p threshold for low-freq band |
| `HIGH_FREQ_THRESHOLD` | 0.95 | Top-p threshold for high-freq band |
| `CALIBRATE` | true | Enable energy-based calibration |
| `FORCE_SINK` / `FORCE_RECENT` | true | Always attend to first/last block |
| `USE_TRITON_SELECT` / `USE_TRITON_LOGITS` | true | Use Triton kernels vs PyTorch fallback |
| `COLLECT_DENSITY` | false | Enable sparsity statistics collection |
| `PATCH_TYPE` | prism | Which attention method (none/prism/xattn/flexprefill/minference) |
| `MODEL_ID` | — | HuggingFace model ID (used by eval and patching) |

### Triton Kernels (`prism/kernels/`)

- `prism_estimation.py` — Fused dual-band softmax and top-p selection kernels
- `block_sparse_attn.py` — Block-sparse attention forward (3-stage: off-diagonal, causal diagonal, non-causal)
- `block_sparse_attn_paged.py` — Paged variant for SGLang serving with `req_to_token` mapping

### Model Patching (`prism/utils/patch.py`)

- `apply_patch(forward_fn, model_id)` — Monkey-patches attention classes for a given model
- `get_attention_classes(model_id)` — Resolves model name to transformer attention classes
- Supported models defined in `_SUPPORTED_MODELS`: `("qwen3_vl", "qwen3", "llama")`
- To add a new model: add its name to `_SUPPORTED_MODELS` and ensure class naming matches the pattern

### Baselines (`baselines/`)

Each baseline (Minference, FlexPrefill, XAttention) follows the same pattern: exports a `*_attention_forward()` function and a `STAT_COLLECTOR` instance.

### Evaluation (`eval/`)

- `run_lm_eval.py` — Entry point. Loads patch via `PATCH_TYPE` env var, applies monkey-patch, runs lm-evaluation-harness
- `efficiency_comparison.py` — Benchmarks attention methods by capturing and replaying attention calls
- `lm-evaluation-harness/` and `lmms-eval/` — Bundled evaluation framework submodules

### Key Tensor Shapes

- Q/K/V: `(batch, num_heads, seq_len, head_dim)`
- Block mask: `(batch, num_heads, num_q_blocks, num_kv_blocks)` — boolean
- Pooled Q/K: `(batch, num_heads, num_blocks, head_dim)`
