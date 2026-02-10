import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_QB": 4, "BLOCK_KB": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QB": 8, "BLOCK_KB": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QB": 4, "BLOCK_KB": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_QB": 8, "BLOCK_KB": 64}, num_warps=8, num_stages=2),
    ],
    key=["K_BLOCKS", "Q_BLOCKS", "HIGH_DIM", "LOW_DIM"],
)
@triton.jit
def _dual_band_softmax_kernel(
    QH_ptr, QL_ptr, KH_ptr, KL_ptr, PROBS_ptr,
    stride_qh_b, stride_qh_h, stride_qh_qb, stride_qh_d,
    stride_ql_b, stride_ql_h, stride_ql_qb, stride_ql_d,
    stride_kh_b, stride_kh_h, stride_kh_kb, stride_kh_d,
    stride_kl_b, stride_kl_h, stride_kl_kb, stride_kl_d,
    stride_p_b, stride_p_h, stride_p_qb, stride_p_kb,
    B, H,
    scale_h_ptr, scale_l_ptr, stride_scale_b, stride_scale_h,
    scale_h, scale_l,
    USE_SCALE_TENSOR: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    BLOCK_QB: tl.constexpr,
    BLOCK_KB: tl.constexpr,
    HIGH_DIM: tl.constexpr,
    LOW_DIM: tl.constexpr,
    BLOCK_DH: tl.constexpr,
    BLOCK_DL: tl.constexpr,
):
    pid_qb = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H
    if b >= B:
        return

    qb_start = pid_qb * BLOCK_QB
    q_idx = qb_start + tl.arange(0, BLOCK_QB)
    mask_q = q_idx < Q_BLOCKS

    if USE_SCALE_TENSOR:
        scale_h_val = tl.load(scale_h_ptr + b * stride_scale_b + h * stride_scale_h).to(tl.float32)
        scale_l_val = tl.load(scale_l_ptr + b * stride_scale_b + h * stride_scale_h).to(tl.float32)
    else:
        scale_h_val = tl.full((), scale_h, tl.float32)
        scale_l_val = tl.full((), scale_l, tl.float32)

    d_h = tl.arange(0, BLOCK_DH)
    d_l = tl.arange(0, BLOCK_DL)

    qh_ptrs = (
        QH_ptr
        + b * stride_qh_b
        + h * stride_qh_h
        + q_idx[:, None] * stride_qh_qb
        + d_h[None, :] * stride_qh_d
    )
    ql_ptrs = (
        QL_ptr
        + b * stride_ql_b
        + h * stride_ql_h
        + q_idx[:, None] * stride_ql_qb
        + d_l[None, :] * stride_ql_d
    )
    qh = tl.load(qh_ptrs, mask=mask_q[:, None] & (d_h[None, :] < HIGH_DIM), other=0.0)
    ql = tl.load(ql_ptrs, mask=mask_q[:, None] & (d_l[None, :] < LOW_DIM), other=0.0)

    offs_k = tl.arange(0, BLOCK_KB)
    shift = K_BLOCKS - Q_BLOCKS
    max_kv = q_idx + shift
    mask_value = tl.full((), -1.0e9, tl.float32)

    m_h = tl.full((BLOCK_QB,), mask_value, tl.float32)
    m_l = tl.full((BLOCK_QB,), mask_value, tl.float32)
    l_h = tl.zeros((BLOCK_QB,), tl.float32)
    l_l = tl.zeros((BLOCK_QB,), tl.float32)

    for k_start in range(0, K_BLOCKS, BLOCK_KB):
        k_idx = k_start + offs_k
        mask_k = k_idx < K_BLOCKS
        mask_kv = mask_q[:, None] & mask_k[None, :] & (k_idx[None, :] <= max_kv[:, None])

        kh_ptrs = (
            KH_ptr
            + b * stride_kh_b
            + h * stride_kh_h
            + k_idx[None, :] * stride_kh_kb
            + d_h[:, None] * stride_kh_d
        )
        kl_ptrs = (
            KL_ptr
            + b * stride_kl_b
            + h * stride_kl_h
            + k_idx[None, :] * stride_kl_kb
            + d_l[:, None] * stride_kl_d
        )
        kh = tl.load(kh_ptrs, mask=(d_h[:, None] < HIGH_DIM) & mask_k[None, :], other=0.0)
        kl = tl.load(kl_ptrs, mask=(d_l[:, None] < LOW_DIM) & mask_k[None, :], other=0.0)

        logits_h = tl.dot(qh, kh).to(tl.float32) / scale_h_val
        logits_l = tl.dot(ql, kl).to(tl.float32) / scale_l_val
        logits_h = tl.where(mask_kv, logits_h, mask_value)
        logits_l = tl.where(mask_kv, logits_l, mask_value)

        max_h = tl.maximum(m_h, tl.max(logits_h, axis=1))
        max_l = tl.maximum(m_l, tl.max(logits_l, axis=1))

        exp_h = tl.exp(logits_h - max_h[:, None])
        exp_l = tl.exp(logits_l - max_l[:, None])
        exp_h = tl.where(mask_kv, exp_h, 0.0)
        exp_l = tl.where(mask_kv, exp_l, 0.0)

        l_h = l_h * tl.exp(m_h - max_h) + tl.sum(exp_h, axis=1)
        l_l = l_l * tl.exp(m_l - max_l) + tl.sum(exp_l, axis=1)
        m_h = max_h
        m_l = max_l

    l_h = tl.maximum(l_h, 1.0e-9)
    l_l = tl.maximum(l_l, 1.0e-9)

    for k_start in range(0, K_BLOCKS, BLOCK_KB):
        k_idx = k_start + offs_k
        mask_k = k_idx < K_BLOCKS
        mask_kv = mask_q[:, None] & mask_k[None, :] & (k_idx[None, :] <= max_kv[:, None])

        kh_ptrs = (
            KH_ptr
            + b * stride_kh_b
            + h * stride_kh_h
            + k_idx[None, :] * stride_kh_kb
            + d_h[:, None] * stride_kh_d
        )
        kl_ptrs = (
            KL_ptr
            + b * stride_kl_b
            + h * stride_kl_h
            + k_idx[None, :] * stride_kl_kb
            + d_l[:, None] * stride_kl_d
        )
        kh = tl.load(kh_ptrs, mask=(d_h[:, None] < HIGH_DIM) & mask_k[None, :], other=0.0)
        kl = tl.load(kl_ptrs, mask=(d_l[:, None] < LOW_DIM) & mask_k[None, :], other=0.0)

        logits_h = tl.dot(qh, kh).to(tl.float32) / scale_h_val
        logits_l = tl.dot(ql, kl).to(tl.float32) / scale_l_val
        logits_h = tl.where(mask_kv, logits_h, mask_value)
        logits_l = tl.where(mask_kv, logits_l, mask_value)

        p_h = tl.exp(logits_h - m_h[:, None]) / l_h[:, None]
        p_l = tl.exp(logits_l - m_l[:, None]) / l_l[:, None]
        p_h = tl.where(mask_kv, p_h, 0.0)
        p_l = tl.where(mask_kv, p_l, 0.0)

        p_h_ptrs = (
            PROBS_ptr
            + b * stride_p_b
            + h * stride_p_h
            + q_idx[:, None] * stride_p_qb
            + k_idx[None, :] * stride_p_kb
        )
        p_l_ptrs = (
            PROBS_ptr
            + b * stride_p_b
            + h * stride_p_h
            + (q_idx + Q_BLOCKS)[:, None] * stride_p_qb
            + k_idx[None, :] * stride_p_kb
        )
        store_mask = mask_q[:, None] & mask_k[None, :]
        tl.store(p_h_ptrs, p_h, mask=store_mask)
        tl.store(p_l_ptrs, p_l, mask=store_mask)


def dual_band_softmax_triton(
    q_high: torch.Tensor,
    q_low: torch.Tensor,
    k_high: torch.Tensor,
    k_low: torch.Tensor,
    scale_h: torch.Tensor | float,
    scale_l: torch.Tensor | float,
) -> torch.Tensor:
    if not q_high.is_cuda:
        raise RuntimeError("dual_band_softmax_triton expects CUDA tensors.")
    q_high = q_high.contiguous()
    q_low = q_low.contiguous()
    k_high = k_high.contiguous()
    k_low = k_low.contiguous()

    bsz, heads, q_blocks, high_dim = q_high.shape
    _, _, k_blocks, low_dim = k_low.shape
    block_dh = triton.next_power_of_2(high_dim)
    block_dl = triton.next_power_of_2(low_dim)

    probs = torch.empty(
        (bsz, heads, 2 * q_blocks, k_blocks),
        device=q_high.device,
        dtype=torch.float32,
    )

    use_scale_tensor = isinstance(scale_h, torch.Tensor)
    if use_scale_tensor != isinstance(scale_l, torch.Tensor):
        raise RuntimeError("scale_h and scale_l must both be tensors or floats.")
    if use_scale_tensor:
        scale_h_t = scale_h.to(device=q_high.device, dtype=torch.float32, non_blocking=True)
        scale_l_t = scale_l.to(device=q_high.device, dtype=torch.float32, non_blocking=True)
        stride_scale_b, stride_scale_h = scale_h_t.stride()[:2]
        scale_h_ptr = scale_h_t
        scale_l_ptr = scale_l_t
        scale_h_val = 1.0
        scale_l_val = 1.0
    else:
        scale_h_ptr = q_high
        scale_l_ptr = q_low
        stride_scale_b = 0
        stride_scale_h = 0
        scale_h_val = float(scale_h)
        scale_l_val = float(scale_l)

    grid = lambda META: (triton.cdiv(q_blocks, META["BLOCK_QB"]), bsz * heads)
    _dual_band_softmax_kernel[grid](
        q_high,
        q_low,
        k_high,
        k_low,
        probs,
        q_high.stride(0),
        q_high.stride(1),
        q_high.stride(2),
        q_high.stride(3),
        q_low.stride(0),
        q_low.stride(1),
        q_low.stride(2),
        q_low.stride(3),
        k_high.stride(0),
        k_high.stride(1),
        k_high.stride(2),
        k_high.stride(3),
        k_low.stride(0),
        k_low.stride(1),
        k_low.stride(2),
        k_low.stride(3),
        probs.stride(0),
        probs.stride(1),
        probs.stride(2),
        probs.stride(3),
        bsz,
        heads,
        scale_h_ptr,
        scale_l_ptr,
        stride_scale_b,
        stride_scale_h,
        scale_h_val,
        scale_l_val,
        USE_SCALE_TENSOR=use_scale_tensor,
        Q_BLOCKS=q_blocks,
        K_BLOCKS=k_blocks,
        HIGH_DIM=high_dim,
        LOW_DIM=low_dim,
        BLOCK_DH=block_dh,
        BLOCK_DL=block_dl,
    )
    return probs


@triton.jit
def _top_p_selection_kernel(
    PROBS_ptr,
    MASK_ptr,
    stride_p_b, stride_p_h, stride_p_q, stride_p_k,
    stride_m_b, stride_m_h, stride_m_q, stride_m_k,
    B, H, Q_BLOCKS, K_BLOCKS,
    high_threshold,
    low_threshold,
    BLOCK_K: tl.constexpr,  # Must be a power of two >= K_BLOCKS
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    if pid_q >= Q_BLOCKS:
        return

    b = pid_bh // H
    h = pid_bh - b * H

    # Offset calculation
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K_BLOCKS

    # --- Process high-frequency row ---
    p_h_ptr = (
        PROBS_ptr
        + b * stride_p_b
        + h * stride_p_h
        + pid_q * stride_p_q
        + offs_k * stride_p_k
    )
    probs_h = tl.load(p_h_ptr, mask=mask_k, other=-1.0)
    
    # Keep sorting stable by slightly decreasing larger-index values
    p_eps_h = probs_h - offs_k.to(tl.float32) * 1e-9
    sorted_h = tl.sort(p_eps_h, descending=True)
    cumsum_h = tl.cumsum(sorted_h)
    
    # Use the minimum selected epsilon value as threshold
    is_selected_h = (cumsum_h - sorted_h) < high_threshold
    threshold_v_h = tl.min(tl.where(is_selected_h & (sorted_h >= -0.5), sorted_h, 2.0))
    selected_h_mask = p_eps_h >= threshold_v_h

    # --- Process low-frequency row ---
    p_l_ptr = (
        PROBS_ptr
        + b * stride_p_b
        + h * stride_p_h
        + (pid_q + Q_BLOCKS) * stride_p_q
        + offs_k * stride_p_k
    )
    probs_l = tl.load(p_l_ptr, mask=mask_k, other=-1.0)
    
    p_eps_l = probs_l - offs_k.to(tl.float32) * 1e-9
    sorted_l = tl.sort(p_eps_l, descending=True)
    cumsum_l = tl.cumsum(sorted_l)
    
    is_selected_l = (cumsum_l - sorted_l) < low_threshold
    threshold_v_l = tl.min(tl.where(is_selected_l & (sorted_l >= -0.5), sorted_l, 2.0))
    selected_l_mask = p_eps_l >= threshold_v_l

    # --- Merge and write back ---
    final_mask = selected_h_mask | selected_l_mask
    
    out_ptr = (
        MASK_ptr
        + b * stride_m_b
        + h * stride_m_h
        + pid_q * stride_m_q
        + offs_k * stride_m_k
    )
    tl.store(out_ptr, final_mask.to(tl.int8), mask=mask_k)


def top_p_selection_triton(
    probs: torch.Tensor,
    high_threshold: float,
    low_threshold: float,
) -> torch.Tensor:
    bsz, heads, combined_q_blocks, k_blocks = probs.shape
    q_blocks = combined_q_blocks // 2

    # Triton tl.sort has known edge-case issues on some versions when the sort length
    # is extremely small (e.g., 1). Handle trivial shapes early to avoid JIT failures.
    if q_blocks == 0 or k_blocks == 0:
        return torch.zeros((bsz, heads, q_blocks, k_blocks), device=probs.device, dtype=torch.bool)
    
    mask = torch.zeros(
        (bsz, heads, q_blocks, k_blocks),
        device=probs.device,
        dtype=torch.int8,
    )

    # Ensure BLOCK_K is at least 2 to avoid tl.sort corner cases when k_blocks == 1.
    block_k = max(2, triton.next_power_of_2(k_blocks))
    
    grid = (q_blocks, bsz * heads)
    _top_p_selection_kernel[grid](
        probs,
        mask,
        probs.stride(0), probs.stride(1), probs.stride(2), probs.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        bsz, heads, q_blocks, k_blocks,
        high_threshold,
        low_threshold,
        BLOCK_K=block_k,
    )
    return mask.to(torch.bool)


if __name__ == "__main__":
    import argparse
    import math
    import os
    import sys
    import time

    # Add project root to sys.path to allow importing prism as a package
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from prism.prism import mean_pool, get_high_freq_components, get_low_freq_components, top_p_select

    parser = argparse.ArgumentParser(description="Quick correctness/perf check for dual_band_softmax_triton.")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--Q", type=int, default=65536)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--low-dim", type=int, default=96)
    parser.add_argument("--high-dim", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--low-temp", type=float, default=1.0)
    parser.add_argument("--high-temp", type=float, default=1.0)
    parser.add_argument("--perf", action="store_true", help="Run timing against PyTorch baseline.")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--use-scale-tensor", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    q = torch.randn(args.B, args.H, args.Q, args.D, device=device, dtype=torch.float16)
    k = torch.randn(args.B, args.H, args.Q, args.D, device=device, dtype=torch.float16)

    q_blocks = mean_pool(q, args.block_size)
    k_blocks = mean_pool(k, args.block_size)
    q_high = get_high_freq_components(q_blocks, args.high_dim)
    q_low = get_low_freq_components(q_blocks, args.low_dim)
    k_high = get_high_freq_components(k_blocks, args.high_dim)
    k_low = get_low_freq_components(k_blocks, args.low_dim)

    scale_h = math.sqrt(args.high_dim) * args.high_temp
    scale_l = math.sqrt(args.low_dim) * args.low_temp
    if args.use_scale_tensor:
        scale_h = torch.full((args.B, args.H, 1, 1), scale_h, device=device, dtype=torch.float32)
        scale_l = torch.full((args.B, args.H, 1, 1), scale_l, device=device, dtype=torch.float32)

    q_blocks_n = q_blocks.shape[2]
    k_blocks_n = k_blocks.shape[2]
    q_block_idx = torch.arange(q_blocks_n, device=device).unsqueeze(1)
    kv_block_idx = torch.arange(k_blocks_n, device=device).unsqueeze(0)
    shift = k_blocks_n - q_blocks_n
    causal_mask = (kv_block_idx <= q_block_idx + shift).unsqueeze(0).unsqueeze(0)

    def baseline_probs():
        logits = torch.empty((args.B, args.H, 2 * q_blocks_n, k_blocks_n), device=device)
        logits[:, :, :q_blocks_n, :] = (
            torch.einsum("bhqd,bhkd->bhqk", q_high, k_high) / scale_h
        ).masked_fill(~causal_mask, float("-inf"))
        logits[:, :, q_blocks_n:, :] = (
            torch.einsum("bhqd,bhkd->bhqk", q_low, k_low) / scale_l
        ).masked_fill(~causal_mask, float("-inf"))
        return torch.softmax(logits, dim=-1, dtype=torch.float32)

    if not args.skip_correctness:
        probs_ref = baseline_probs()
        probs_triton = dual_band_softmax_triton(q_high, q_low, k_high, k_low, scale_h, scale_l)
        max_abs_diff = (probs_ref - probs_triton).abs().max().item()
        print("max_abs_diff", max_abs_diff)

    if args.perf:
        for _ in range(args.warmup):
            dual_band_softmax_triton(q_high, q_low, k_high, k_low, scale_h, scale_l)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            dual_band_softmax_triton(q_high, q_low, k_high, k_low, scale_h, scale_l)
        torch.cuda.synchronize()
        print("triton_avg_ms", (time.time() - t0) * 1000 / args.iters)

        for _ in range(args.warmup):
            _ = baseline_probs()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            _ = baseline_probs()
        torch.cuda.synchronize()
        print("pytorch_avg_ms", (time.time() - t0) * 1000 / args.iters)

    # --- Selection Test ---
    print("\n--- Testing top_p_selection_triton (with PyTorch Probs to eliminate Softmax error) ---")
    high_thr = 0.95
    low_thr = 0.95
    
    if not args.skip_correctness:
        # 1. Use PyTorch reference probabilities (probs_ref)
        probs_h_ref = probs_ref[:, :, :q_blocks_n, :]
        probs_l_ref = probs_ref[:, :, q_blocks_n:, :]
        
        # Reference Mask (PyTorch)
        mask_h_ref = top_p_select(probs_h_ref, high_thr, causal=True)
        mask_l_ref = top_p_select(probs_l_ref, low_thr, causal=True)
        mask_ref = mask_h_ref | mask_l_ref
        
        # 2. Feed PyTorch probabilities to the Triton selection kernel
        # This removes numerical differences from the softmax stage and tests selection logic only
        mask_triton_clean = top_p_selection_triton(probs_ref, high_thr, low_thr)
        
        # Compare
        diff_clean = (mask_ref != mask_triton_clean).sum().item()
        print(f"Selection Logic Correctness (Pure): diff={diff_clean} / {mask_ref.numel()}")
        if diff_clean == 0:
            print("Pure Selection Logic check passed: Triton selection is bit-accurate to PyTorch when given same inputs.")

        # 3. Also show the original comparison (including softmax error)
        mask_triton_original = top_p_selection_triton(probs_triton, high_thr, low_thr)
        diff_orig = (mask_ref != mask_triton_original).sum().item()
        print(f"Combined Correctness (Softmax + Select): diff={diff_orig} / {mask_ref.numel()} (ratio: {diff_orig/mask_ref.numel():.6f})")

    if args.perf:
        # Triton Warmup
        for _ in range(args.warmup):
            top_p_selection_triton(probs_triton, high_thr, low_thr)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            top_p_selection_triton(probs_triton, high_thr, low_thr)
        torch.cuda.synchronize()
        print(f"triton_selection_avg_ms: {((time.time() - t0) * 1000 / args.iters):.4f}")

        # PyTorch Baseline Warmup
        for _ in range(args.warmup):
            m_h = top_p_select(probs_triton[:, :, :q_blocks_n, :], high_thr, causal=True)
            m_l = top_p_select(probs_triton[:, :, q_blocks_n:, :], low_thr, causal=True)
            _ = m_h | m_l
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            m_h = top_p_select(probs_triton[:, :, :q_blocks_n, :], high_thr, causal=True)
            m_l = top_p_select(probs_triton[:, :, q_blocks_n:, :], low_thr, causal=True)
            _ = m_h | m_l
        torch.cuda.synchronize()
        print(f"pytorch_selection_avg_ms: {((time.time() - t0) * 1000 / args.iters):.4f}")
