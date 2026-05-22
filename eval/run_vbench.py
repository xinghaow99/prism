"""VBench evaluation orchestrator for HunyuanVideo with sparse attention.

Loads the model ONCE and iterates over all prompts in-process, avoiding
the ~3 min model reload per prompt. Supports resume and prompt range
selection for parallelization across GPUs.

Usage:
    python eval/run_vbench.py \
        --attn xattn --threshold 0.95 --stride 8 \
        --prompt_start 0 --prompt_end 100 \
        --save_path results/vbench/HunyuanVideo_xattn
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="VBench evaluation orchestrator")
    parser.add_argument("--prompt_file", type=str,
                        default="eval/HunyuanVideo/all_dimension_longer.txt",
                        help="Path to VBench prompts file (one prompt per line)")
    parser.add_argument("--prompt_start", type=int, default=0,
                        help="Start index (inclusive) for prompt range")
    parser.add_argument("--prompt_end", type=int, default=-1,
                        help="End index (exclusive) for prompt range. -1 for all.")
    parser.add_argument("--save_path", type=str, default="results/vbench",
                        help="Directory to save generated videos")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_seed", type=int, default=1,
                        help="Number of seeds to use per prompt")

    # Video generation params
    parser.add_argument("--video_size", type=int, nargs=2, default=[720, 1280],
                        help="Video height and width")
    parser.add_argument("--video_length", type=int, default=129,
                        help="Number of video frames")
    parser.add_argument("--infer_steps", type=int, default=50,
                        help="Number of denoising steps")

    # Model params
    parser.add_argument("--model_base", type=str, default="eval/HunyuanVideo/ckpts",
                        help="Path to model checkpoints")
    parser.add_argument("--dit_weight", type=str,
                        default="eval/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
                        help="Path to DiT weights")

    # Sparse attention params
    parser.add_argument("--attn", type=str, default="flash",
                        choices=["flash", "xattn", "prism"],
                        help="Attention mode")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Top-p threshold for sparse attention")
    parser.add_argument("--stride", type=int, default=8,
                        help="XAttention estimation stride")
    parser.add_argument("--attn_warmup_steps", type=int, default=5,
                        help="Dense warmup steps before sparse")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Sparse attention block size")

    return parser.parse_args()


def build_attn_suffix(args):
    """Build a descriptive suffix for the output directory."""
    if args.attn == "flash":
        return "flash"
    return f"{args.attn}_warmup={args.attn_warmup_steps}_threshold={args.threshold}_stride={args.stride}"


def main():
    args = parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    hunyuan_dir = script_dir / "HunyuanVideo"

    # Resolve model paths to absolute
    args.model_base = str((root_dir / args.model_base).resolve())
    args.dit_weight = str((root_dir / args.dit_weight).resolve())

    # Read prompts
    prompt_file = root_dir / args.prompt_file
    with open(prompt_file) as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    # Apply prompt range
    end = args.prompt_end if args.prompt_end > 0 else len(all_prompts)
    prompts = all_prompts[args.prompt_start:end]
    print(f"Processing {len(prompts)} prompts (indices {args.prompt_start} to {end})")

    # Output directory
    attn_suffix = build_attn_suffix(args)
    save_dir = root_dir / args.save_path / attn_suffix / "all_dimension_longer"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check which prompts need generation (resume support)
    prompts_to_generate = []
    for idx, prompt in enumerate(prompts):
        global_idx = args.prompt_start + idx
        video_dir = save_dir / str(global_idx)
        existing = list(video_dir.glob("*.mp4")) if video_dir.exists() else []
        if existing:
            print(f"[{global_idx}] Skipping (already exists)")
            continue
        prompts_to_generate.append((global_idx, prompt))

    if not prompts_to_generate:
        print("All prompts already generated. Nothing to do.")
        return

    print(f"{len(prompts_to_generate)} prompts to generate, {len(prompts) - len(prompts_to_generate)} skipped")

    # Add HunyuanVideo to path and load model ONCE
    sys.path.insert(0, str(hunyuan_dir))
    os.chdir(hunyuan_dir)

    from hyvideo.config import parse_args as hv_parse_args
    from hyvideo.inference import HunyuanVideoSampler
    from hyvideo.utils.file_utils import save_videos_grid
    from loguru import logger

    # Build HunyuanVideo args — override sys.argv so hv_parse_args() sees only these
    saved_argv = sys.argv
    sys.argv = [
        "sample_video.py",
        "--model-base", args.model_base,
        "--dit-weight", args.dit_weight,
        "--video-size", str(args.video_size[0]), str(args.video_size[1]),
        "--video-length", str(args.video_length),
        "--infer-steps", str(args.infer_steps),
        "--flow-reverse",
        "--embedded-cfg-scale", "6.0",
        "--seed", str(args.seed),
        "--attn", args.attn,
        "--threshold", str(args.threshold),
        "--stride", str(args.stride),
        "--attn-warmup-steps", str(args.attn_warmup_steps),
    ]
    hv_args = hv_parse_args()
    sys.argv = saved_argv

    print("Loading model (one-time)...")
    load_start = time.time()
    sampler = HunyuanVideoSampler.from_pretrained(Path(args.model_base), args=hv_args)
    hv_args = sampler.args
    print(f"Model loaded in {time.time() - load_start:.1f}s")

    # Generate videos
    for global_idx, prompt in prompts_to_generate:
        video_dir = save_dir / str(global_idx)
        video_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{global_idx}] Generating: {prompt[:80]}...")

        for seed_offset in range(args.num_seed):
            seed = args.seed + seed_offset
            try:
                outputs = sampler.predict(
                    prompt=prompt,
                    height=args.video_size[0],
                    width=args.video_size[1],
                    video_length=args.video_length,
                    seed=seed,
                    negative_prompt=hv_args.neg_prompt,
                    infer_steps=args.infer_steps,
                    guidance_scale=hv_args.cfg_scale,
                    num_videos_per_prompt=1,
                    flow_shift=hv_args.flow_shift,
                    batch_size=1,
                    embedded_guidance_scale=hv_args.embedded_cfg_scale,
                )
                samples = outputs['samples']
                for i, sample in enumerate(samples):
                    sample = samples[i].unsqueeze(0)
                    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                    cur_save_path = f"{video_dir}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
                    save_videos_grid(sample, cur_save_path, fps=24)
                    logger.info(f'Sample save to: {cur_save_path}')
            except Exception as e:
                print(f"[{global_idx}] ERROR: {e}")
                continue

    print(f"Done. Videos saved to: {save_dir}")


if __name__ == "__main__":
    main()
