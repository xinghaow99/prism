import argparse
import gc
import importlib
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add workspace root to sys.path
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EVAL_DIR)
sys.path.append(ROOT_DIR)

from prism.utils.patch import _SUPPORTED_MODELS, apply_patch, get_attention_classes
from prism import prism as prism_module
import baselines.XAttention as xattn_module
import baselines.FlexPrefill as flexprefill_module
import baselines.Minference as minference_module

class StopForward(Exception):
    pass

def restore_all_attention(orig_forwards):
    for cls, orig in orig_forwards.items():
        cls.forward = orig

def collect_attention_calls(model, input_ids, model_id, layer_idx=None):
    attn_classes = get_attention_classes(
        model_id=model_id,
        supported_models=_SUPPORTED_MODELS,
    )
    current_forwards = {cls: cls.forward for cls in attn_classes}
    calls = []

    def make_capture_forward(cls_forward):
        def capture_forward(self, *args, **kwargs):
            should_capture = True
            if layer_idx is not None:
                if hasattr(self, "layer_idx"):
                    should_capture = self.layer_idx == layer_idx
                else:
                    should_capture = False
            
            if should_capture:
                # Detach and clone tensors to save memory and avoid tracking gradients/activations
                def detach_tensors(x):
                    if isinstance(x, torch.Tensor):
                        return x.detach().clone()
                    elif isinstance(x, list):
                        return [detach_tensors(v) for v in x]
                    elif isinstance(x, tuple):
                        return tuple(detach_tensors(v) for v in x)
                    elif isinstance(x, dict):
                        return {k: detach_tensors(v) for k, v in x.items()}
                    return x

                new_args = detach_tensors(args)
                new_kwargs = detach_tensors(kwargs)
                calls.append((cls_forward, self, new_args, new_kwargs))
                
                # If we only need one layer, we can stop the forward pass immediately
                if layer_idx is not None:
                    raise StopForward()
                
            return cls_forward(self, *args, **kwargs)
        return capture_forward

    for cls in attn_classes:
        cls.forward = make_capture_forward(current_forwards[cls]).__get__(None, cls)

    with torch.no_grad():
        try:
            model(input_ids, use_cache=False)
        except StopForward:
            pass
        except TypeError:
            try:
                model(input_ids)
            except StopForward:
                pass

    restore_all_attention(current_forwards)
    return calls

def _reload_patch_modules() -> None:
    importlib.reload(prism_module)
    importlib.reload(xattn_module)
    importlib.reload(flexprefill_module)
    importlib.reload(minference_module)

def _build_method_registry():
    return {
        "FlashAttention": {
            "forward_fn": None,
            "stat_collector": None,
        },
        "Minference": {
            "forward_fn": minference_module.minference_attention_forward,
            "stat_collector": minference_module.STAT_COLLECTOR,
        },
        "FlexPrefill": {
            "forward_fn": flexprefill_module.flexprefill_attention_forward,
            "stat_collector": flexprefill_module.STAT_COLLECTOR,
        },
        "XAttention": {
            "forward_fn": xattn_module.xattn_attention_forward,
            "stat_collector": xattn_module.STAT_COLLECTOR,
        },
        "Prism": {
            "forward_fn": prism_module.prism_attention_forward,
            "stat_collector": prism_module.STAT_COLLECTOR,
        },
    }


def _configure_stat_collector(
    stat_collector,
    collect_density: bool,
    collect_select_time: bool,
) -> None:
    if stat_collector is None:
        return
    stat_collector.collect_density = collect_density
    stat_collector.collect_select_time = collect_select_time
    stat_collector.reset()


def _reset_select_time_collection(stat_collector) -> None:
    if stat_collector is not None:
        stat_collector.reset_select_time()


def _drain_select_time_ms(stat_collector) -> float:
    if stat_collector is None:
        return 0.0
    return stat_collector.drain_select_time_ms()


def _get_density_value(method_name: str, stat_collector) -> Optional[float]:
    if method_name == "FlashAttention":
        return 1.0
    if stat_collector is None:
        return None
    summary = stat_collector.summary()
    if summary is None:
        return None
    return summary["total"]


def _set_or_unset_env(name: str, value: Optional[object]) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)

def get_real_inputs(batch_size, seq_len, tokenizer, data_path, data_idx, model_device):
    print(f"  Loading real data from {data_path} (index {data_idx})...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict data formats
    if isinstance(data, dict):
        sample = data[str(data_idx)]
    else:
        sample = data[int(data_idx)]
        
    prompt = sample["origin_prompt"][0]["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    
    actual_len = input_ids.shape[1]
    if actual_len >= seq_len:
        input_ids = input_ids[:, :seq_len]
    else:
        # Repeat tokens to reach requested length if the sample is too short
        repeats = (seq_len // actual_len) + 1
        input_ids = input_ids.repeat(1, repeats)[:, :seq_len]
    
    input_ids = input_ids.repeat(batch_size, 1).to(model_device)
    return input_ids

def benchmark_prefill(
    calls,
    num_warmup=3,
    num_runs=5,
    collect_select_time=False,
    stat_collector=None,
):
    try:
        if not calls:
            raise RuntimeError(
                "No attention calls provided for benchmarking."
            )

        for _ in range(num_warmup):
            with torch.no_grad():
                for _, module, args, kwargs in calls:
                    # Call the current forward method of the module's class
                    # which will be the patched version if it was patched.
                    type(module).forward(module, *args, **kwargs)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if collect_select_time:
            _reset_select_time_collection(stat_collector)

        attn_latencies = []
        select_latencies = [] if collect_select_time else None

        for _ in range(num_runs):
            if collect_select_time:
                _reset_select_time_collection(stat_collector)
            events = []
            with torch.no_grad():
                for _, module, args, kwargs in calls:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    out = type(module).forward(module, *args, **kwargs)
                    end.record()
                    events.append((start, end))
                    del out
            torch.cuda.synchronize()

            attn_time = 0
            for start, end in events:
                attn_time += start.elapsed_time(end)
            attn_latencies.append(attn_time)
            if collect_select_time:
                select_latencies.append(_drain_select_time_ms(stat_collector))

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB
        select_stats = None
        if collect_select_time:
            select_stats = (np.mean(select_latencies), np.std(select_latencies))
        return (np.mean(attn_latencies), np.std(attn_latencies)), select_stats, peak_memory
    finally:
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Efficiency comparison for different attention mechanisms")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_lens", type=str, default="8192,16384,32768", help="Comma-separated sequence lengths")
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--data_path", type=str, default="data/ruler_examples/ruler_niah_multikey_3_32k_0.json")
    parser.add_argument("--data_idx", type=str, default="0")
    parser.add_argument(
        "--attn_only_layer_idx",
        type=int,
        default=-1,
        help="Only cache the specified layer index (-1 means all layers).",
    )
    
    # Prism args
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--low_freq_dim", type=int, default=None)
    parser.add_argument("--high_freq_dim", type=int, default=None)
    parser.add_argument("--low_freq_threshold", type=float, default=0.9)
    parser.add_argument("--high_freq_threshold", type=float, default=0.9)

    parser.add_argument("--force_sink", dest="force_sink", action="store_true")
    parser.add_argument("--no_force_sink", dest="force_sink", action="store_false")
    parser.set_defaults(force_sink=True)

    parser.add_argument("--force_recent", dest="force_recent", action="store_true")
    parser.add_argument("--no_force_recent", dest="force_recent", action="store_false")
    parser.set_defaults(force_recent=True)

    parser.add_argument("--calibrate", dest="calibrate", action="store_true")
    parser.add_argument("--no_calibrate", dest="calibrate", action="store_false")
    parser.set_defaults(calibrate=True)

    parser.add_argument("--use_triton_select", dest="use_triton_select", action="store_true")
    parser.add_argument("--no_use_triton_select", dest="use_triton_select", action="store_false")
    parser.set_defaults(use_triton_select=True)

    parser.add_argument("--use_triton_logits", dest="use_triton_logits", action="store_true")
    parser.add_argument("--no_use_triton_logits", dest="use_triton_logits", action="store_false")
    parser.set_defaults(use_triton_logits=True)

    parser.add_argument(
        "--no_density",
        action="store_true",
        help="Disable density collection and reporting.",
    )
    parser.add_argument(
        "--no_select_time",
        action="store_true",
        help="Disable selection time collection and reporting.",
    )
    
    # XAttention args
    parser.add_argument("--xattn_stride", type=int, default=8)
    parser.add_argument("--xattn_threshold", type=float, default=0.9)

    # FlexPrefill args
    parser.add_argument("--flexprefill_gamma", type=float, default=0.9)
    parser.add_argument("--flexprefill_tau", type=float, default=0.0)
    parser.add_argument("--flexprefill_min_budget", type=int, default=None)
    parser.add_argument("--flexprefill_max_budget", type=int, default=None)
    parser.add_argument("--flexprefill_gqa_interleave", action="store_true")

    # Minference args
    parser.add_argument("--minference_vertical_size", type=int, default=1000)
    parser.add_argument("--minference_slash_size", type=int, default=6096)
    parser.add_argument("--minference_adaptive_budget", type=float, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default="FlashAttention,Minference,FlexPrefill,XAttention,Prism",
        help="Comma-separated methods to benchmark",
    )
    
    args = parser.parse_args()
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    if args.attn_only_layer_idx < -1:
        raise ValueError("--attn_only_layer_idx must be -1 or a non-negative integer.")
    attn_only_layer_idx = None if args.attn_only_layer_idx == -1 else args.attn_only_layer_idx
    print("Attention-only benchmark: capture full-forward inputs and replay attention only.")
    if attn_only_layer_idx is not None:
        print(f"  Cached layer idx: {attn_only_layer_idx}")
    else:
        print("  Cached layer idx: all layers")
    
    results = []

    # Set environment variables for the patched forwards
    os.environ["BLOCK_SIZE"] = str(args.block_size)
    _set_or_unset_env("LOW_FREQ_DIM", args.low_freq_dim)
    _set_or_unset_env("HIGH_FREQ_DIM", args.high_freq_dim)
    os.environ["LOW_FREQ_THRESHOLD"] = str(args.low_freq_threshold)
    os.environ["HIGH_FREQ_THRESHOLD"] = str(args.high_freq_threshold)
    os.environ["FORCE_SINK"] = "true" if args.force_sink else "false"
    os.environ["FORCE_RECENT"] = "true" if args.force_recent else "false"
    os.environ["CALIBRATE"] = "true" if args.calibrate else "false"
    os.environ["USE_TRITON_SELECT"] = "true" if args.use_triton_select else "false"
    os.environ["USE_TRITON_LOGITS"] = "true" if args.use_triton_logits else "false"

    collect_density = not args.no_density
    collect_select_time = not args.no_select_time
    os.environ["COLLECT_DENSITY"] = "true" if collect_density else "false"
    os.environ["COLLECT_SELECT_TIME"] = "true" if collect_select_time else "false"

    os.environ["XATTN_STRIDE"] = str(args.xattn_stride)
    os.environ["XATTN_THRESHOLD"] = str(args.xattn_threshold)
    os.environ["FLEX_PREFILL_GAMMA"] = str(args.flexprefill_gamma)
    os.environ["FLEX_PREFILL_TAU"] = str(args.flexprefill_tau)
    _set_or_unset_env("FLEX_PREFILL_MIN_BUDGET", args.flexprefill_min_budget)
    _set_or_unset_env("FLEX_PREFILL_MAX_BUDGET", args.flexprefill_max_budget)
    os.environ["GQA_INTERLEAVE"] = "true" if args.flexprefill_gqa_interleave else "false"
    os.environ["MINFERENCE_VERTICAL_SIZE"] = str(args.minference_vertical_size)
    os.environ["MINFERENCE_SLASH_SIZE"] = str(args.minference_slash_size)
    _set_or_unset_env("MINFERENCE_ADAPTIVE_BUDGET", args.minference_adaptive_budget)

    _reload_patch_modules()
    method_registry = _build_method_registry()
    method_order = ["FlashAttention", "Minference", "FlexPrefill", "XAttention", "Prism"]
    selected_method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = []
    for method_name in method_order:
        if method_name in selected_method_names:
            cfg = method_registry[method_name]
            methods.append((method_name, cfg["forward_fn"], cfg["stat_collector"]))

    if not methods:
        print(f"No valid methods selected from: {args.methods}")
        return

    attn_classes = get_attention_classes(
        model_id=args.model_id,
        supported_models=_SUPPORTED_MODELS,
    )
    original_forwards = {cls: cls.forward for cls in attn_classes}

    for seq_len in seq_lens:
        print(f"\nBenchmarking SeqLen: {seq_len}")
        
        # Load model if it's not present (e.g. after being deleted in previous iteration)
        if 'model' not in locals() or model is None:
            print(f"Loading model {args.model_id} on {device}...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id, 
                torch_dtype=torch.bfloat16, 
                device_map=device,
                trust_remote_code=True
            )
            model.eval()

        # Capture shared inputs
        restore_all_attention(original_forwards)
        input_ids = get_real_inputs(args.batch_size, seq_len, tokenizer, args.data_path, args.data_idx, model.device)
        print("  Capturing shared attention inputs...")
        shared_calls = collect_attention_calls(
            model,
            input_ids,
            model_id=args.model_id,
            layer_idx=attn_only_layer_idx,
        )
        
        if not shared_calls:
            print(f"  Failed to capture any attention calls for SeqLen {seq_len}. Check --attn_only_layer_idx.")
            continue

        # Aggressively release model memory if we are only testing specific layers
        if attn_only_layer_idx is not None:
            print("  Releasing model weights to free up GPU memory...")
            del model
            model = None
            gc.collect()
            torch.cuda.empty_cache()
        
        # --- Added: Global initialization to avoid first-run peak pollution ---
        if methods:
            # Run one call to initialize CUDA/Triton state
            with torch.no_grad():
                _m_f, _m, _a, _kw = shared_calls[0]
                _m_f(_m, *_a, **_kw)
            gc.collect()
            torch.cuda.empty_cache()
        # ----------------------------------------------------------------------

        baseline_a = None
        baseline_mem = None
        
        for name, forward_fn, stat_collector in methods:
            # Clean up before each method
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"  Method: {name}")
            
            # Patch
            if forward_fn is not None:
                apply_patch(
                    forward_fn=forward_fn,
                    model_id=args.model_id,
                    supported_models=_SUPPORTED_MODELS,
                )
            else:
                restore_all_attention(original_forwards)
            _configure_stat_collector(
                stat_collector,
                collect_density=collect_density,
                collect_select_time=collect_select_time,
            )
            
            try:
                # Use shared_calls instead of re-capturing
                (a_mean, a_std), select_stats, peak_mem = benchmark_prefill(
                    shared_calls,
                    args.num_warmup,
                    args.num_runs,
                    collect_select_time=collect_select_time,
                    stat_collector=stat_collector,
                )
                if name == "FlashAttention":
                    baseline_a = a_mean
                    baseline_mem = peak_mem
                
                a_accel = baseline_a / a_mean if (baseline_a is not None and a_mean > 0) else None
                mem_overhead = peak_mem - baseline_mem if baseline_mem is not None else None
                
                density_value = None
                if collect_density:
                    density_value = _get_density_value(name, stat_collector)
                density_str = "-" if density_value is None else f"{density_value:.4f}"
                select_str = "-"
                if collect_select_time and select_stats is not None:
                    select_str = f"{select_stats[0]:.2f} ± {select_stats[1]:.2f}"
                
                results.append({
                    "Method": name,
                    "SeqLen": seq_len,
                    "Attn Latency (ms)": f"{a_mean:.2f} ± {a_std:.2f}",
                    "Select Latency (ms)": select_str,
                    "Attn Accel": f"{a_accel:.2f}x" if a_accel is not None else "-",
                    "Peak Memory (GB)": f"{peak_mem:.2f}",
                    "Mem Overhead (GB)": f"{mem_overhead:.2f}" if mem_overhead is not None else "-",
                    "Density": density_str
                })
                
                accel_str = f"{a_accel:.2f}x" if a_accel is not None else "-"
                mem_oh_str = f"{mem_overhead:.2f}" if mem_overhead is not None else "-"
                print(f"    Attn: {a_mean:.2f} ms ({accel_str}), Est.: {select_str}, Peak Mem: {peak_mem:.2f} GB (Overhead: {mem_oh_str} GB), Density: {density_str}")
                
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at SeqLen {seq_len} for {name}")
                results.append({
                    "Method": name,
                    "SeqLen": seq_len,
                    "Attn Latency (ms)": "OOM",
                    "Select Latency (ms)": "OOM" if collect_select_time else "-",
                    "Attn Accel": "-",
                    "Peak Memory (GB)": "OOM",
                    "Mem Overhead (GB)": "-",
                    "Density": "-"
                })
                torch.cuda.empty_cache()
            
            # Clean up
            if stat_collector is not None:
                stat_collector.collect_density = False
                stat_collector.collect_select_time = False

        # Clean up shared calls and cache before next sequence length
        del shared_calls
        gc.collect()
        torch.cuda.empty_cache()

    # Print results table
    print("\n" + "="*140)
    print(f"{'Method':<18} | {'SeqLen':<8} | {'Attn Latency (ms)':<18} | {'Est. Time (ms)':<20} | {'Attn Accel':<12} | {'Mem (GB)':<8} | {'Mem Overhead':<12} | {'Density':<8}")
    print("-" * 140)
    for res in results:
        print(f"{res['Method']:<18} | {res['SeqLen']:<8} | {res['Attn Latency (ms)']:<18} | {res['Select Latency (ms)']:<20} | {res['Attn Accel']:<12} | {res['Peak Memory (GB)']:<8} | {res['Mem Overhead (GB)']:<12} | {res['Density']:<8}")
    print("="*140)

if __name__ == "__main__":
    main()
