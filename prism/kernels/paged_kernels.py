import torch
import triton
import triton.language as tl
import math

@triton.jit
def _mean_pool_paged_kernel(
    K_Pool, Req_to_tokens, B_req_idx, B_Seq_Len, Out,
    stride_pool_bs, stride_pool_h, stride_pool_d,
    stride_req_b,
    stride_out_bz, stride_out_h, stride_out_nb, stride_out_d,
    BLOCK_SIZE: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_nb = tl.program_id(0) # prism block index
    pid_h = tl.program_id(1)  # head index
    pid_bz = tl.program_id(2) # batch index

    cur_seq_len = tl.load(B_Seq_Len + pid_bz)
    
    # Check if this block index is within sequence bounds
    if pid_nb * BLOCK_SIZE >= cur_seq_len:
        return

    req_idx = tl.load(B_req_idx + pid_bz)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    # Count valid tokens in this block (tail block may be partial)
    block_start = pid_nb * BLOCK_SIZE
    valid = tl.maximum(tl.minimum(cur_seq_len - block_start, BLOCK_SIZE), 0)
    
    # Mean pool over BLOCK_SIZE tokens
    # We use a simple loop here because tokens are logically contiguous but physically scattered
    for i in range(0, BLOCK_SIZE):
        token_idx_in_seq = pid_nb * BLOCK_SIZE + i
        mask = token_idx_in_seq < cur_seq_len
        
        # Load physical token index from sglang req_to_token table
        token_physical_idx = tl.load(Req_to_tokens + req_idx * stride_req_b + token_idx_in_seq, mask=mask, other=0)
        
        # Load K from pool
        off_k = token_physical_idx * stride_pool_bs + pid_h * stride_pool_h + tl.arange(0, BLOCK_D) * stride_pool_d
        k = tl.load(K_Pool + off_k, mask=mask & (tl.arange(0, BLOCK_D) < HEAD_DIM), other=0.0)
        
        acc += k
    
    # Compute mean
    denom = tl.maximum(tl.full((), valid, tl.int32), 1).to(tl.float32)
    acc /= denom
    
    # Store to contiguous output: [bsz, heads, num_blocks, head_dim]
    off_out = (pid_bz * stride_out_bz + 
               pid_h * stride_out_h + 
               pid_nb * stride_out_nb + 
               tl.arange(0, BLOCK_D) * stride_out_d)
    
    tl.store(Out + off_out, acc.to(Out.type.element_ty), mask=tl.arange(0, BLOCK_D) < HEAD_DIM)

def mean_pool_paged(k_pool, req_to_token, b_req_idx, b_seq_len, block_size=128):
    """
    Computes mean pool of keys directly from sglang's paged KV pool.
    
    Returns:
        torch.Tensor: [batch_size, num_heads, num_blocks, head_dim]
    """
    batch_size = b_req_idx.shape[0]
    num_heads, head_dim = k_pool.shape[1], k_pool.shape[2]
    max_seq_len = b_seq_len.max().item()
    num_blocks = (max_seq_len + block_size - 1) // block_size
    
    # Use zeros to avoid leaving uninitialized blocks for shorter sequences in the batch.
    out = torch.zeros(
        (batch_size, num_heads, num_blocks, head_dim),
        device=k_pool.device,
        dtype=k_pool.dtype,
    )
    
    grid = (num_blocks, num_heads, batch_size)
    block_d = triton.next_power_of_2(head_dim)
    
    _mean_pool_paged_kernel[grid](
        k_pool, req_to_token, b_req_idx, b_seq_len, out,
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        req_to_token.stride(0),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d
    )
    
    return out
