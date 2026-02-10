import torch
import triton
import triton.language as tl
import math

@triton.jit
def _block_sparse_attn_paged_fwd_inner(
    acc, l_i, m_i,
    q,
    cur_batch_req_idx,
    cur_seq_len,
    cur_prefix_len,
    K_Pool,
    V_Pool,
    Req_to_tokens,
    K_block_indices_ptrs,
    stride_req_b,
    stride_pool_bs, stride_pool_h, stride_pool_d,
    stride_seqk_k_block_indices,
    pid_seq,
    offs_m,
    offs_n,
    offs_d,
    cur_kv_head,
    softmax_scale,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr,
    STAGE: tl.constexpr,
):
    if STAGE == 1:
        # Off-diagonal: process all prefix blocks + blocks before the current query's diagonal block
        # The query tokens start at cur_prefix_len.
        # pid_seq is the query block index within the extension part.
        lo = 0
        hi = cur_prefix_len + (pid_seq * BLOCK_M // LOGICAL_BLOCK_SIZE) * LOGICAL_BLOCK_SIZE
    elif STAGE == 2:
        # On-diagonal: process only the diagonal block
        lo = cur_prefix_len + (pid_seq * BLOCK_M // LOGICAL_BLOCK_SIZE) * LOGICAL_BLOCK_SIZE
        hi = tl.minimum(lo + LOGICAL_BLOCK_SIZE, cur_seq_len)
    else:
        # Non-causal: process all blocks
        lo, hi = 0, cur_seq_len

    for kv_seq_start in range(lo, hi, BLOCK_N):
        # Load block selection indicator
        # K_block_indices covers the entire sequence (prefix + extension)
        block_idx_offset = (kv_seq_start // LOGICAL_BLOCK_SIZE) * stride_seqk_k_block_indices
        k_block_idx = tl.load(K_block_indices_ptrs + block_idx_offset)
        
        if k_block_idx:
            offs_n_cur = kv_seq_start + offs_n
            mask_n = offs_n_cur < cur_seq_len
            
            token_indices = tl.load(
                Req_to_tokens + cur_batch_req_idx * stride_req_b + offs_n_cur,
                mask=mask_n, other=0
            )
            
            off_k = token_indices[None, :] * stride_pool_bs + cur_kv_head * stride_pool_h + offs_d[:, None] * stride_pool_d
            k = tl.load(K_Pool + off_k, mask=mask_n[None, :], other=0.0)
            
            qk = tl.dot(q, k)
            
            if STAGE == 2:
                # Absolute position of query tokens: cur_prefix_len + offs_m
                causal_mask = (cur_prefix_len + offs_m[:, None]) >= offs_n_cur[None, :]
                combined_mask = causal_mask & mask_n[None, :]
                qk = qk * softmax_scale + tl.where(combined_mask, 0, -1.0e6)
            else:
                qk = qk * softmax_scale + tl.where(mask_n[None, :], 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            
            # Correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            
            # Scale accumulator
            acc = acc * alpha[:, None]
            
            # Load V
            off_v = token_indices[:, None] * stride_pool_bs + cur_kv_head * stride_pool_h + offs_d[None, :] * stride_pool_d
            v = tl.load(V_Pool + off_v, mask=mask_n[:, None], other=0.0)
            
            p = p.to(dtype)
            acc += tl.dot(p, v)
            
            # Update statistics
            l_i = l_i * alpha + l_ij
            m_i = m_ij

    return acc, l_i, m_i

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['HEAD_DIM', 'LOGICAL_BLOCK_SIZE'],
)
@triton.jit
def _block_sparse_attn_paged_fwd(
    Q, K_Pool, V_Pool, O,
    Req_to_tokens, B_req_idx, B_Seq_Len, B_Prefix_Len, B_Start_Loc,
    K_block_indices,
    stride_qbs, stride_qh, stride_qd,
    stride_req_b,
    stride_pool_bs, stride_pool_h, stride_pool_d,
    stride_obs, stride_oh, stride_od,
    stride_bz_k_mask, stride_h_k_mask, stride_sq_k_mask, stride_sk_k_mask,
    num_kv_groups,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr,
    STAGE: tl.constexpr,  # 3 for causal, 1 for non-causal
):
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_bz = tl.program_id(2)

    cur_batch_req_idx = tl.load(B_req_idx + pid_bz)
    cur_seq_len = tl.load(B_Seq_Len + pid_bz)
    cur_prefix_len = tl.load(B_Prefix_Len + pid_bz)
    cur_extend_len = cur_seq_len - cur_prefix_len
    cur_q_start_loc = tl.load(B_Start_Loc + pid_bz)
    cur_kv_head = pid_h // num_kv_groups

    # Check if this query block is within the extension length
    if pid_seq * BLOCK_M >= cur_extend_len:
        return

    offs_m = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q block from extension part
    off_q = (cur_q_start_loc + offs_m[:, None]) * stride_qbs + pid_h * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_extend_len, other=0.0)

    # Calculate logical block index within the extension for block mask lookup
    # block_mask dimension 2 is indexed 0..num_q_blocks-1 (extension blocks)
    logical_q_idx = (pid_seq * BLOCK_M) // LOGICAL_BLOCK_SIZE
    K_block_indices_ptrs = (
        K_block_indices 
        + pid_bz * stride_bz_k_mask 
        + pid_h * stride_h_k_mask 
        + logical_q_idx * stride_sq_k_mask
    )

    # Init accumulators
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Scale for numerical stability (use log2 for exp2)
    qk_scale = softmax_scale * 1.44269504  # 1/log(2)

    # Stage 1: Off-diagonal blocks
    if STAGE & 1:
        acc, l_i, m_i = _block_sparse_attn_paged_fwd_inner(
            acc, l_i, m_i,
            q,
            cur_batch_req_idx,
            cur_seq_len,
            cur_prefix_len,
            K_Pool,
            V_Pool,
            Req_to_tokens,
            K_block_indices_ptrs,
            stride_req_b,
            stride_pool_bs, stride_pool_h, stride_pool_d,
            stride_sk_k_mask,
            pid_seq,
            offs_m,
            offs_n,
            offs_d,
            cur_kv_head,
            qk_scale,
            Q.type.element_ty,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            LOGICAL_BLOCK_SIZE,
            1
        )

    # Stage 2: On-diagonal blocks
    if STAGE & 2:
        acc, l_i, m_i = _block_sparse_attn_paged_fwd_inner(
            acc, l_i, m_i,
            q,
            cur_batch_req_idx,
            cur_seq_len,
            cur_prefix_len,
            K_Pool,
            V_Pool,
            Req_to_tokens,
            K_block_indices_ptrs,
            stride_req_b,
            stride_pool_bs, stride_pool_h, stride_pool_d,
            stride_sk_k_mask,
            pid_seq,
            offs_m,
            offs_n,
            offs_d,
            cur_kv_head,
            qk_scale,
            Q.type.element_ty,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            LOGICAL_BLOCK_SIZE,
            2
        )

    # Epilogue: normalize and store
    acc = acc / l_i[:, None]
    off_o = (cur_q_start_loc + offs_m[:, None]) * stride_obs + pid_h * stride_oh + offs_d[None, :] * stride_od
    tl.store(O + off_o, acc.to(Q.type.element_ty), mask=offs_m[:, None] < cur_extend_len)


def block_sparse_attention_paged(
    q: torch.Tensor,          # [total_q_tokens, num_heads, head_dim]
    k_pool: torch.Tensor,     # [pool_size, num_kv_heads, head_dim]
    v_pool: torch.Tensor,     # [pool_size, num_kv_heads, head_dim]
    block_mask: torch.Tensor, # [batch_size, num_heads, num_q_blocks, num_k_blocks]
    req_to_token: torch.Tensor, # [max_batch_size, max_seq_len]
    b_req_idx: torch.Tensor,    # [batch_size]
    b_seq_len: torch.Tensor,    # [batch_size]
    b_prefix_len: torch.Tensor, # [batch_size]
    b_start_loc: torch.Tensor,  # [batch_size]
    block_size: int = 128,
    causal: bool = True
) -> torch.Tensor:
    total_q_tokens, num_heads, head_dim = q.shape
    batch_size = b_req_idx.shape[0]
    num_kv_heads = k_pool.shape[1]
    num_kv_groups = num_heads // num_kv_heads
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Grid: [max_blocks_per_seq, num_heads, batch_size]
    max_extend_len = (b_seq_len - b_prefix_len).max().item()
    
    grid = lambda META: (
        triton.cdiv(max_extend_len, META['BLOCK_M']),
        num_heads,
        batch_size
    )
    
    _block_sparse_attn_paged_fwd[grid](
        q, k_pool, v_pool, o,
        req_to_token, b_req_idx, b_seq_len, b_prefix_len, b_start_loc,
        block_mask.to(torch.int32),
        q.stride(0), q.stride(1), q.stride(2),
        req_to_token.stride(0),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3),
        num_kv_groups, softmax_scale,
        HEAD_DIM=head_dim,
        LOGICAL_BLOCK_SIZE=block_size,
        STAGE=3 if causal else 1
    )
    return o


def block_sparse_attention_paged_torch_naive(
    q, k_pool, v_pool, block_mask, req_to_token, b_req_idx, b_seq_len, b_prefix_len, b_start_loc,
    block_size=128, causal=True
):
    """Naive PyTorch implementation for paged block-sparse attention."""
    total_q_tokens, num_heads, head_dim = q.shape
    batch_size = b_req_idx.shape[0]
    num_kv_heads = k_pool.shape[1]
    num_kv_groups = num_heads // num_kv_heads
    
    o = torch.zeros_like(q)
    
    for b in range(batch_size):
        cur_req_idx = b_req_idx[b].item()
        cur_seq_len = b_seq_len[b].item()
        cur_prefix_len = b_prefix_len[b].item()
        cur_extend_len = cur_seq_len - cur_prefix_len
        cur_start_loc = b_start_loc[b].item()
        
        # Get physical token locations for this sequence (entire sequence)
        token_indices = req_to_token[cur_req_idx, :cur_seq_len]
        
        # Extract KV from pool for this sequence
        cur_k = k_pool[token_indices] # [seq_len, num_kv_heads, head_dim]
        cur_v = v_pool[token_indices] # [seq_len, num_kv_heads, head_dim]
        
        # Repeat KV for GQA if needed
        if num_kv_groups > 1:
            cur_k = cur_k.repeat_interleave(num_kv_groups, dim=1)
            cur_v = cur_v.repeat_interleave(num_kv_groups, dim=1)
            
        # Get Q for this request (extension only)
        cur_q = q[cur_start_loc:cur_start_loc+cur_extend_len] # [extend_len, num_heads, head_dim]
        
        # Transpose to [heads, seq, dim]
        cur_q = cur_q.transpose(0, 1)
        cur_k = cur_k.transpose(0, 1)
        cur_v = cur_v.transpose(0, 1)
        
        num_q_blocks = (cur_extend_len + block_size - 1) // block_size
        num_k_blocks = (cur_seq_len + block_size - 1) // block_size
        
        cur_o = torch.zeros_like(cur_q)
        
        for h in range(num_heads):
            for q_block_idx in range(num_q_blocks):
                q_start = q_block_idx * block_size
                q_end = min(q_start + block_size, cur_extend_len)
                q_slice = cur_q[h, q_start:q_end, :]
                
                # Find selected key blocks (block_mask covers entire sequence)
                selected_k_block_indices = torch.where(block_mask[b, h, q_block_idx, :])[0]
                
                if len(selected_k_block_indices) == 0:
                    continue
                    
                selected_k_blocks = []
                selected_v_blocks = []
                k_positions = []
                
                for k_block_idx in selected_k_block_indices:
                    k_start = k_block_idx.item() * block_size
                    if k_start >= cur_seq_len: continue
                    k_end = min(k_start + block_size, cur_seq_len)
                    selected_k_blocks.append(cur_k[h, k_start:k_end, :])
                    selected_v_blocks.append(cur_v[h, k_start:k_end, :])
                    k_positions.append(torch.arange(k_start, k_end, device=q.device))
                
                if not selected_k_blocks: continue
                
                concat_k = torch.cat(selected_k_blocks, dim=0)
                concat_v = torch.cat(selected_v_blocks, dim=0)
                concat_k_pos = torch.cat(k_positions, dim=0)
                
                # Absolute query positions for causal masking
                q_positions = torch.arange(cur_prefix_len + q_start, cur_prefix_len + q_end, device=q.device)
                
                scores = torch.matmul(q_slice, concat_k.transpose(-2, -1)) / math.sqrt(head_dim)
                
                if causal:
                    c_mask = q_positions.unsqueeze(-1) >= concat_k_pos.unsqueeze(0)
                    scores = scores.masked_fill(~c_mask, float('-inf'))
                
                attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                cur_o[h, q_start:q_end, :] = torch.matmul(attn_weights, concat_v)
                
        o[cur_start_loc:cur_start_loc+cur_extend_len] = cur_o.transpose(0, 1)
        
    return o


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    
    # Test parameters
    batch_size = 2
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    block_size = 128
    seq_lens = torch.tensor([500, 1000], device=device, dtype=torch.int32)
    max_seq_len = seq_lens.max().item()
    
    # total tokens in the pool
    pool_size = 5000
    k_pool = torch.randn(pool_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_pool = torch.randn(pool_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    # Flattened Q
    total_q_tokens = seq_lens.sum().item()
    q = torch.randn(total_q_tokens, num_heads, head_dim, device=device, dtype=dtype)
    
    # Metadata
    b_req_idx = torch.tensor([0, 1], device=device, dtype=torch.int32)
    b_start_loc = torch.zeros(batch_size, device=device, dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)
    
    # Req to token mapping (random physical indices)
    req_to_token = torch.zeros(batch_size, pool_size, device=device, dtype=torch.int32)
    all_indices = torch.randperm(pool_size, device=device)
    req_to_token[0, :seq_lens[0]] = all_indices[:seq_lens[0]]
    req_to_token[1, :seq_lens[1]] = all_indices[seq_lens[0]:seq_lens[0]+seq_lens[1]]
    
    # Block mask
    num_q_blocks = (max_seq_len + block_size - 1) // block_size
    num_k_blocks = num_q_blocks
    block_mask = torch.zeros(batch_size, num_heads, num_q_blocks, num_k_blocks, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        cur_num_blocks = (seq_lens[b].item() + block_size - 1) // block_size
        for h in range(num_heads):
            for i in range(cur_num_blocks):
                # Randomly select some blocks
                for j in range(i + 1):
                    if torch.rand(1).item() > 0.3:
                        block_mask[b, h, i, j] = True
                        
    print("Testing block_sparse_attention_paged...")
    print(f"Total tokens: {total_q_tokens}, heads: {num_heads}, kv_heads: {num_kv_heads}")
    
    # Run Triton kernel
    out_triton = block_sparse_attention_paged(
        q, k_pool, v_pool, block_mask, req_to_token, b_req_idx, seq_lens, b_start_loc,
        block_size=block_size, causal=True
    )
    
    # Run Naive reference
    out_naive = block_sparse_attention_paged_torch_naive(
        q, k_pool, v_pool, block_mask, req_to_token, b_req_idx, seq_lens, b_start_loc,
        block_size=block_size, causal=True
    )
    
    # Compare
    diff = (out_triton - out_naive).abs().max().item()
    print(f"Max difference: {diff}")
    
    if diff < 0.05:
        print("✓ Paged block-sparse attention test passed!")
    else:
        print("✗ Paged block-sparse attention test failed!")
