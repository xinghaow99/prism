import torch
import triton
import triton.language as tl
import math
import time


@triton.jit
def _block_sparse_attn_fwd_inner(
    acc, l_i, m_i,
    q,
    qo_len,
    kv_len,
    K_ptrs,
    V_ptrs,
    K_block_indices_ptrs,
    stride_seq_k,
    stride_seq_v,
    stride_seqk_k_block_indices,  # stride along kv-block dimension of mask
    pid_seq,
    offs_m,
    offs_n,
    softmax_scale,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr,
    STAGE: tl.constexpr,
):
    """
    Inner loop for block sparse attention.
    STAGE 1: Process off-diagonal blocks (no causal mask needed within block)
    STAGE 2: Process on-diagonal blocks (need causal mask within block)
    STAGE 3: Process all blocks without causal consideration (non-causal attention)
    """
    # Determine the range of KV blocks to process based on STAGE
    if STAGE == 1:
        # Off-diagonal: process blocks before the current query's diagonal block
        lo = 0
        hi = (pid_seq * BLOCK_M // LOGICAL_BLOCK_SIZE) * LOGICAL_BLOCK_SIZE
    elif STAGE == 2:
        # On-diagonal: process only the diagonal block
        lo = (pid_seq * BLOCK_M // LOGICAL_BLOCK_SIZE) * LOGICAL_BLOCK_SIZE
        hi = tl.minimum(lo + LOGICAL_BLOCK_SIZE, kv_len)
    else:  # STAGE == 3, non-causal
        lo, hi = 0, kv_len

    K_ptrs_cur = tl.advance(K_ptrs, (0, lo))
    V_ptrs_cur = tl.advance(V_ptrs, (lo, 0))

    # Only process if there are blocks to process
    for kv_seq_start in range(lo, hi, BLOCK_N):
        # Load block selection indicator
        block_idx_offset = (kv_seq_start // LOGICAL_BLOCK_SIZE) * stride_seqk_k_block_indices
        k_block_idx = tl.load(K_block_indices_ptrs + block_idx_offset)
        
        if k_block_idx:
            # Load K block
            k = tl.load(K_ptrs_cur, boundary_check=(0, 1), padding_option="zero")  # (HEAD_DIM, BLOCK_N)
            
            # Compute QK
            qk = tl.dot(q, k)
            
            # Apply masking based on stage
            cols = kv_seq_start + offs_n
            kv_mask = cols < kv_len

            if STAGE == 2:
                # On-diagonal: apply causal mask within block and mask padding
                causal_mask = offs_m[:, None] >= cols[None, :]
                combined_mask = causal_mask & kv_mask[None, :]
                qk = qk * softmax_scale + tl.where(combined_mask, 0, -1.0e6)
            else:
                # Off-diagonal or non-causal: only boundary mask
                qk = qk * softmax_scale + tl.where(kv_mask[None, :], 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            
            # Correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            
            # Scale accumulator
            acc = acc * alpha[:, None]
            
            # Load V and accumulate
            v = tl.load(V_ptrs_cur, boundary_check=(0, 1), padding_option="zero")
            p = p.to(dtype)
            acc += tl.dot(p, v)
            
            # Update statistics
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        K_ptrs_cur = tl.advance(K_ptrs_cur, (0, BLOCK_N))
        V_ptrs_cur = tl.advance(V_ptrs_cur, (BLOCK_N, 0))

    return acc, l_i, m_i


def _prune_invalid_configs(configs, named_args, **kwargs):
    logical_bs = kwargs.get('LOGICAL_BLOCK_SIZE', None)
    if logical_bs is None:
        logical_bs = named_args.get('LOGICAL_BLOCK_SIZE', None)
    try:
        logical_bs = int(logical_bs)
    except Exception:
        logical_bs = None
    if logical_bs is None or logical_bs <= 0:
        return configs
    pruned = []
    for conf in configs:
        bm = conf.kwargs.get('BLOCK_M', 0)
        bn = conf.kwargs.get('BLOCK_N', 0)
        if bm == 0 or bn == 0:
            continue
        if (logical_bs % bm == 0) and (logical_bs % bn == 0):
            pruned.append(conf)
    return pruned


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    ],
    key=['H', 'HEAD_DIM', 'LOGICAL_BLOCK_SIZE', 'num_kv_groups'],
    prune_configs_by={'early_config_prune': _prune_invalid_configs},
)
@triton.jit
def _block_sparse_attn_fwd(
    Q, K, V, O,  # (batch_size, num_q_heads(num_kv_heads), q_len(kv_len), head_dim)
    K_block_indices,  # (batch_size, num_q_heads, num_q_blocks, num_k_blocks)
    stride_bz_q, stride_h_q, stride_seq_q, stride_d_q,
    stride_bz_k, stride_h_k, stride_seq_k, stride_d_k,
    stride_bz_v, stride_h_v, stride_seq_v, stride_d_v,
    stride_bz_o, stride_h_o, stride_seq_o, stride_d_o,
    stride_bz_k_block_indices, stride_h_k_block_indices, stride_seqq_k_block_indices, stride_seqk_k_block_indices,
    qo_len, kv_len,
    softmax_scale,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr,
    STAGE: tl.constexpr,  # 3 for causal, 1 for non-causal
):
    # Enforce tile alignment with logical block size at compile-time
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_M) == 0)
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_N) == 0)

    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_bz = tl.program_id(2).to(tl.int64)

    offs_m = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    dtype = Q.type.element_ty

    # Init ptrs
    Q_ptrs = tl.make_block_ptr(
        base=Q + pid_bz * stride_bz_q + pid_h * stride_h_q,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_seq_q, stride_d_q),
        offsets=(pid_seq * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    K_ptrs = tl.make_block_ptr(
        base=K + pid_bz * stride_bz_k + (pid_h // num_kv_groups) * stride_h_k,
        shape=(HEAD_DIM, kv_len),
        strides=(stride_d_k, stride_seq_k),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0)
    )
    V_ptrs = tl.make_block_ptr(
        base=V + pid_bz * stride_bz_v + (pid_h // num_kv_groups) * stride_h_v,
        shape=(kv_len, HEAD_DIM),
        strides=(stride_seq_v, stride_d_v),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    O_ptrs = tl.make_block_ptr(
        base=O + pid_bz * stride_bz_o + pid_h * stride_h_o,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_seq_o, stride_d_o),
        offsets=(pid_seq * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )

    # Calculate the logical query block this program belongs to
    q_start_index = pid_seq * BLOCK_M
    logical_q_block_idx = q_start_index // LOGICAL_BLOCK_SIZE
    K_block_indices_ptrs = (
        K_block_indices
        + pid_bz * stride_bz_k_block_indices
        + pid_h * stride_h_k_block_indices
        + logical_q_block_idx * stride_seqq_k_block_indices
    )

    # Init accumulators
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Scale for numerical stability (use log2 for exp2)
    qk_scale = softmax_scale * 1.44269504  # 1/log(2)

    # Load Q
    q = tl.load(Q_ptrs, boundary_check=(0, 1), padding_option="zero")

    # Stage 1: Off-diagonal blocks (no causal mask needed)
    # For causal=True (STAGE=3), process blocks before the diagonal
    if STAGE & 1:
        acc, l_i, m_i = _block_sparse_attn_fwd_inner(
            acc, l_i, m_i,
            q,
            qo_len,
            kv_len,
            K_ptrs,
            V_ptrs,
            K_block_indices_ptrs,
            stride_seq_k,
            stride_seq_v,
            stride_seqk_k_block_indices,
            pid_seq,
            offs_m,
            offs_n,
            qk_scale,
            dtype,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            LOGICAL_BLOCK_SIZE,
            4 - STAGE,  # If STAGE=3, inner gets 1; if STAGE=1, inner gets 3
        )

    # Stage 2: On-diagonal blocks (need causal mask)
    # For causal=True (STAGE=3), process the diagonal block with masking
    if STAGE & 2:
        acc, l_i, m_i = _block_sparse_attn_fwd_inner(
            acc, l_i, m_i,
            q,
            qo_len,
            kv_len,
            K_ptrs,
            V_ptrs,
            K_block_indices_ptrs,
            stride_seq_k,
            stride_seq_v,
            stride_seqk_k_block_indices,
            pid_seq,
            offs_m,
            offs_n,
            qk_scale,
            dtype,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            LOGICAL_BLOCK_SIZE,
            2,
        )

    # Epilogue: normalize and store
    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(dtype), boundary_check=(0, 1))


def block_sparse_attention(
    Q: torch.Tensor,  # (batch_size, num_q_heads, q_len, head_dim)
    K: torch.Tensor,  # (batch_size, num_kv_heads, kv_len, head_dim)
    V: torch.Tensor,  # (batch_size, num_kv_heads, kv_len, head_dim)
    block_mask: torch.Tensor,  # (batch_size, num_q_heads, num_q_blocks, num_k_blocks) bool
    block_size: int = 128,
    causal: bool = True,
) -> torch.Tensor:
    """
    Block sparse attention using Triton.
    
    Args:
        Q: Query tensor of shape (batch_size, num_q_heads, q_len, head_dim)
        K: Key tensor of shape (batch_size, num_kv_heads, kv_len, head_dim)
        V: Value tensor of shape (batch_size, num_kv_heads, kv_len, head_dim)
        block_mask: Boolean mask of shape (batch_size, num_q_heads, num_q_blocks, num_k_blocks)
                    True means the block is selected for attention.
        block_size: Size of each block (default 128)
        causal: Whether to apply causal masking (default True)
    
    Returns:
        Output tensor of shape (batch_size, num_q_heads, q_len, head_dim)
    """
    batch_size, num_q_heads, q_len, head_dim = Q.shape
    _, num_kv_heads, kv_len, _ = K.shape

    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
    num_kv_groups = num_q_heads // num_kv_heads

    # Pad Q, K, V to be divisible by block_size
    pad_q = (block_size - q_len % block_size) % block_size
    pad_kv = (block_size - kv_len % block_size) % block_size

    if pad_q > 0:
        Q = torch.nn.functional.pad(Q, (0, 0, 0, pad_q))
    if pad_kv > 0:
        K = torch.nn.functional.pad(K, (0, 0, 0, pad_kv))
        V = torch.nn.functional.pad(V, (0, 0, 0, pad_kv))

    padded_q_len = Q.shape[2]
    padded_kv_len = K.shape[2]

    # Convert block_mask (bool) to int for Triton
    K_block_indices = block_mask.to(torch.int32).contiguous()

    # Allocate output
    O = torch.empty_like(Q)

    # Softmax scale
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # STAGE: 3 for causal, 1 for non-causal
    stage = 3 if causal else 1

    # Grid
    grid = lambda META: (
        triton.cdiv(q_len, META['BLOCK_M']),
        num_q_heads,
        batch_size
    )

    # Launch kernel
    _block_sparse_attn_fwd[grid](
        Q, K, V, O,
        K_block_indices,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        K_block_indices.stride(0), K_block_indices.stride(1), K_block_indices.stride(2), K_block_indices.stride(3),
        q_len, kv_len,
        softmax_scale,
        H=num_q_heads,
        num_kv_groups=num_kv_groups,
        HEAD_DIM=head_dim,
        LOGICAL_BLOCK_SIZE=block_size,
        STAGE=stage,
    )

    # Remove padding
    O = O[:, :, :q_len, :]

    return O


def block_sparse_attention_torch_naive(
    Q: torch.Tensor,  # (batch_size, num_q_heads, q_len, head_dim)
    K: torch.Tensor,  # (batch_size, num_kv_heads, kv_len, head_dim)
    V: torch.Tensor,  # (batch_size, num_kv_heads, kv_len, head_dim)
    block_mask: torch.Tensor,  # (batch_size, num_q_heads, num_q_blocks, num_k_blocks) bool
    block_size: int = 128,
    causal: bool = True,
) -> torch.Tensor:
    """
    Naive PyTorch implementation of block sparse attention for reference/testing.
    """
    batch_size, num_q_heads, q_len, head_dim = Q.shape
    _, num_kv_heads, kv_len, _ = K.shape

    num_kv_groups = num_q_heads // num_kv_heads
    num_q_blocks = (q_len + block_size - 1) // block_size
    num_k_blocks = (kv_len + block_size - 1) // block_size

    O = torch.zeros_like(Q)

    for b in range(batch_size):
        for h in range(num_q_heads):
            kv_h = h // num_kv_groups
            for q_block_idx in range(num_q_blocks):
                q_start = q_block_idx * block_size
                q_end = min(q_start + block_size, q_len)
                q_block = Q[b, h, q_start:q_end, :]

                # Find selected key blocks
                selected_k_block_indices = torch.where(block_mask[b, h, q_block_idx, :])[0]

                if len(selected_k_block_indices) == 0:
                    continue

                # Concatenate selected key/value tokens
                selected_k_blocks = []
                selected_v_blocks = []
                k_positions = []

                for k_block_idx in selected_k_block_indices:
                    k_start = k_block_idx.item() * block_size
                    k_end = min(k_start + block_size, kv_len)
                    selected_k_blocks.append(K[b, kv_h, k_start:k_end, :])
                    selected_v_blocks.append(V[b, kv_h, k_start:k_end, :])
                    k_positions.append(torch.arange(k_start, k_end, device=Q.device))

                concat_k = torch.cat(selected_k_blocks, dim=0)
                concat_v = torch.cat(selected_v_blocks, dim=0)
                concat_k_pos = torch.cat(k_positions, dim=0)

                # Query positions
                q_positions = torch.arange(q_start, q_end, device=Q.device)

                # Compute attention scores
                scores = torch.matmul(q_block, concat_k.transpose(-2, -1)) / math.sqrt(head_dim)

                # Apply causal mask
                if causal:
                    causal_mask = q_positions.unsqueeze(-1) >= concat_k_pos.unsqueeze(0)
                    scores = scores.masked_fill(~causal_mask, float('-inf'))

                # Softmax
                attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)

                # Output
                output_block = torch.matmul(attn_weights, concat_v)
                O[b, h, q_start:q_end, :] = output_block

    return O


if __name__ == "__main__":
    # Test the implementation
    torch.manual_seed(42)

    batch_size = 2
    num_heads = 8
    seq_len = 12345
    head_dim = 64
    block_size = 128

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)

    num_q_blocks = (seq_len + block_size - 1) // block_size
    num_k_blocks = num_q_blocks

    # Create a random block mask (with causal constraint)
    block_mask = torch.zeros(batch_size, num_heads, num_q_blocks, num_k_blocks, dtype=torch.bool, device="cuda")
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(num_q_blocks):
                # Always include causal blocks and some random blocks
                for j in range(i + 1):
                    if torch.rand(1).item() > 0.3:  # 70% chance to include
                        block_mask[b, h, i, j] = True

    print("Testing block sparse attention...")
    print(f"Q shape: {Q.shape}, block_mask shape: {block_mask.shape}")

    # Test Triton implementation
    out_triton = block_sparse_attention(Q, K, V, block_mask, block_size, causal=True)
    print(f"Triton output shape: {out_triton.shape}")

    # Test naive implementation
    out_naive = block_sparse_attention_torch_naive(Q, K, V, block_mask, block_size, causal=True)
    print(f"Naive output shape: {out_naive.shape}")

    # Compare
    diff = (out_triton - out_naive).abs().max().item()
    print(f"Max difference: {diff}")

    if diff < 0.05:  # Relaxed threshold for bfloat16
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")
        
    # Also test non-causal
    print("\nTesting non-causal attention...")
    out_triton_nc = block_sparse_attention(Q, K, V, block_mask, block_size, causal=False)
    out_naive_nc = block_sparse_attention_torch_naive(Q, K, V, block_mask, block_size, causal=False)
    diff_nc = (out_triton_nc - out_naive_nc).abs().max().item()
    print(f"Max difference (non-causal): {diff_nc}")
    if diff_nc < 0.05:  # Relaxed threshold for bfloat16
        print("✓ Non-causal test passed!")
    else:
        print("✗ Non-causal test failed!")
