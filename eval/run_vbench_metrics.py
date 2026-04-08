"""Compute PSNR/SSIM/LPIPS between dense and sparse generated videos.

Compares video pairs generated with dense (flash) attention and a sparse
method (xattn), frame by frame. Outputs per-video and aggregate metrics.
Supports incremental caching: already-computed per-video results are loaded
from the output JSON and skipped, so re-runs only compute new videos.

Usage:
    python eval/run_vbench_metrics.py \
        --dense_dir results/vbench/flash/all_dimension_longer \
        --sparse_dir results/vbench/xattn_warmup=5_threshold=0.95_stride=8/all_dimension_longer \
        --output results/vbench/metrics_xattn_0.95.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def load_video_frames(video_path: str) -> np.ndarray:
    """Load video as numpy array of shape (T, H, W, 3) with values in [0, 1]."""
    try:
        import imageio.v3 as iio
        frames = iio.imread(video_path, plugin="pyav")
    except ImportError:
        import imageio
        reader = imageio.get_reader(video_path)
        frames = np.stack([frame for frame in reader])
        reader.close()
    return frames.astype(np.float64) / 255.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images in [0, 1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images in [0, 1]."""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=-1, data_range=1.0)


_lpips_model = None

def _get_lpips_model(device: str = "cuda"):
    """Get or create the LPIPS model (singleton)."""
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device)
    return _lpips_model

def compute_lpips_batch(frames1: np.ndarray, frames2: np.ndarray, device: str = "cuda") -> float:
    """Compute average LPIPS over all frame pairs."""
    loss_fn = _get_lpips_model(device)

    total_lpips = 0.0
    n_frames = min(len(frames1), len(frames2))

    with torch.no_grad():
        for i in range(n_frames):
            # Convert to (1, 3, H, W) tensor in [-1, 1]
            t1 = torch.from_numpy(frames1[i]).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
            t2 = torch.from_numpy(frames2[i]).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
            total_lpips += loss_fn(t1, t2).item()

    return total_lpips / n_frames if n_frames > 0 else 0.0


def find_video_in_dir(directory: Path) -> Path | None:
    """Find the first .mp4 file in a directory."""
    mp4s = list(directory.glob("*.mp4"))
    return mp4s[0] if mp4s else None


def load_cached_results(output_path: str) -> dict:
    """Load previously computed per-video results from JSON, keyed by idx."""
    if not os.path.exists(output_path):
        return {}
    try:
        with open(output_path) as f:
            data = json.load(f)
        return {r["idx"]: r for r in data.get("per_video", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute VBench fidelity metrics")
    parser.add_argument("--dense_dir", type=str, required=True,
                        help="Directory with dense baseline videos (flash)")
    parser.add_argument("--sparse_dir", type=str, required=True,
                        help="Directory with sparse method videos (xattn)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path. Default: <sparse_dir>/../metrics.json")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for LPIPS computation")
    parser.add_argument("--max_prompts", type=int, default=-1,
                        help="Max number of prompts to evaluate (-1 for all)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Ignore cached results and recompute everything")
    return parser.parse_args()


def main():
    args = parse_args()

    dense_dir = Path(args.dense_dir)
    sparse_dir = Path(args.sparse_dir)

    if not dense_dir.exists():
        raise FileNotFoundError(f"Dense directory not found: {dense_dir}")
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Sparse directory not found: {sparse_dir}")

    output_path = args.output or str(sparse_dir.parent / "metrics.json")

    # Load cached per-video results
    cached = {} if args.no_cache else load_cached_results(output_path)
    if cached:
        print(f"Loaded {len(cached)} cached results from {output_path}")

    # Collect matching video pairs by subdirectory index
    prompt_dirs = sorted([d for d in dense_dir.iterdir() if d.is_dir()],
                         key=lambda p: int(p.name) if p.name.isdigit() else 0)

    if args.max_prompts > 0:
        prompt_dirs = prompt_dirs[:args.max_prompts]

    results = []
    computed = 0
    for prompt_dir in prompt_dirs:
        idx = prompt_dir.name

        # Use cached result if available
        if idx in cached:
            results.append(cached[idx])
            continue

        dense_video = find_video_in_dir(prompt_dir)
        sparse_video = find_video_in_dir(sparse_dir / idx)

        if dense_video is None or sparse_video is None:
            continue

        print(f"[{idx}] Computing metrics...")

        frames_dense = load_video_frames(str(dense_video))
        frames_sparse = load_video_frames(str(sparse_video))

        n_frames = min(len(frames_dense), len(frames_sparse))
        if n_frames == 0:
            print(f"[{idx}] WARNING: Empty video, skipping")
            continue

        frames_dense = frames_dense[:n_frames]
        frames_sparse = frames_sparse[:n_frames]

        # Skip if resolution mismatch or unexpected resolution
        if frames_dense[0].shape != frames_sparse[0].shape:
            print(f"[{idx}] Skipping (resolution mismatch: {frames_dense[0].shape} vs {frames_sparse[0].shape})")
            continue
        if frames_dense[0].shape[0] != 720 or frames_dense[0].shape[1] != 1280:
            print(f"[{idx}] Skipping (unexpected resolution: {frames_dense[0].shape[0]}x{frames_dense[0].shape[1]})")
            continue

        # Per-frame PSNR and SSIM
        psnr_vals = [compute_psnr(frames_dense[i], frames_sparse[i]) for i in range(n_frames)]
        ssim_vals = [compute_ssim(frames_dense[i], frames_sparse[i]) for i in range(n_frames)]

        # LPIPS (batch)
        lpips_val = compute_lpips_batch(frames_dense, frames_sparse, device=args.device)

        result = {
            "idx": idx,
            "n_frames": n_frames,
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
            "lpips": float(lpips_val),
        }
        results.append(result)
        computed += 1
        print(f"[{idx}] PSNR={result['psnr']:.2f} SSIM={result['ssim']:.4f} LPIPS={result['lpips']:.4f}")

        # Save incrementally after each video
        _save_summary(results, dense_dir, sparse_dir, output_path)

    if not results:
        print("No valid video pairs found.")
        return

    print(f"\nComputed {computed} new, {len(results) - computed} cached")
    _save_summary(results, dense_dir, sparse_dir, output_path)

    summary = _make_summary(results, dense_dir, sparse_dir)
    print(f"\n{'='*60}")
    print(f"Aggregate ({len(results)} videos):")
    print(f"  PSNR:  {summary['psnr_mean']:.2f}")
    print(f"  SSIM:  {summary['ssim_mean']:.4f}")
    print(f"  LPIPS: {summary['lpips_mean']:.4f}")
    print(f"{'='*60}")
    print(f"Metrics saved to: {output_path}")


def _make_summary(results, dense_dir, sparse_dir):
    return {
        "num_prompts": len(results),
        "psnr_mean": float(np.mean([r["psnr"] for r in results])),
        "ssim_mean": float(np.mean([r["ssim"] for r in results])),
        "lpips_mean": float(np.mean([r["lpips"] for r in results])),
        "dense_dir": str(dense_dir),
        "sparse_dir": str(sparse_dir),
        "per_video": results,
    }


def _save_summary(results, dense_dir, sparse_dir, output_path):
    summary = _make_summary(results, dense_dir, sparse_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
