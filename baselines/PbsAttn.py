"""PBS-Attn (Permuted Block-Sparse Attention) baseline.

Vendored from upstream pbs_attn package (kernels + helpers + orchestrator) and
wrapped with the prism baseline forward template.
"""

import os
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack

from prism.utils.patch import get_rotary_fn
from prism.utils.stat_collector import StatCollector

STAT_COLLECTOR = StatCollector.from_env(method_name="pbs_attn")


def reset_select_time_collection() -> None:
    STAT_COLLECTOR.reset_select_time()


def drain_select_time_ms() -> float:
    return STAT_COLLECTOR.drain_select_time_ms()


# =============================================================================
# Triton kernels
# (vendored from pbs_attn/src/kernels/permuted_block_sparse_attention.py)
# =============================================================================

@triton.jit
def _permuted_block_sparse_attn_fwd_inner(
    acc, l_i, m_i,
    q,
    qo_len,
    kv_len,
    K_base,                          # raw pointer to K (already offset by batch+kv-head)
    V_base,                          # raw pointer to V (already offset by batch+kv-head)
    K_block_indices_ptrs,
    perm_Q_indices_ptrs,
    perm_K_indices_ptrs,
    stride_seq_k, stride_d_k,        # K strides for indirect gather
    stride_seq_v, stride_d_v,        # V strides for indirect gather
    stride_bz_perm_q_indices, stride_h_perm_q_indices, stride_seq_perm_q_indices,
    stride_bz_perm_k_indices, stride_h_perm_k_indices, stride_seq_perm_k_indices,
    pid_seq,
    RANGE_Q_SEQ,
    RANGE_KV_SEQ,
    softmax_scale,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr,
    STAGE: tl.constexpr,
):
    segment_id =(pid_seq*BLOCK_M) // SEGMENT_SIZE
    if STAGE == 1:
        lo, hi = 0, (segment_id * SEGMENT_SIZE)
        # perm_Q_indices = None
    elif STAGE == 2:
        lo = segment_id * SEGMENT_SIZE
        hi = tl.minimum(lo + SEGMENT_SIZE, kv_len)

        perm_Q_indices = tl.load(perm_Q_indices_ptrs, boundary_check=(0,1))
    elif STAGE == 3:
        lo, hi = 0, kv_len
        # perm_Q_indices = None

    perm_K_indices_ptrs_cur = tl.advance(perm_K_indices_ptrs, (0, lo))

    head_offsets = tl.arange(0, HEAD_DIM)

    if not (STAGE == 1 and segment_id == 0):
        for kv_seq_start in range(lo, hi, BLOCK_N):
            k_block_idx = tl.load(K_block_indices_ptrs + kv_seq_start // LOGICAL_BLOCK_SIZE)
            if k_block_idx:

                kv_mask = RANGE_KV_SEQ[None, :] >= (kv_len - kv_seq_start) # True if exceed kv_len
                # Load perm_K_indices once: drives both K and V indirect gathers
                # plus the on-diagonal causal mask in STAGE == 2.
                perm_K_indices = tl.load(
                    perm_K_indices_ptrs_cur, boundary_check=(0,1), padding_option="zero"
                )  # (1, BLOCK_N)
                k_rows = perm_K_indices.reshape((BLOCK_N,))
                k_inbounds = k_rows < kv_len

                # Indirect K gather: K is loaded as (HEAD_DIM, BLOCK_N) for the
                # dot product, transposed-style.
                k_offsets = (
                    head_offsets[:, None] * stride_d_k
                    + k_rows[None, :] * stride_seq_k
                )  # (HEAD_DIM, BLOCK_N)
                k = tl.load(
                    K_base + k_offsets,
                    mask=k_inbounds[None, :],
                    other=0.0,
                ).to(dtype)
                qk = tl.dot(q, k)
                qk *= softmax_scale
                if STAGE == 2:
                    # On-diagonal segments, apply mask within segment
                    mask = perm_Q_indices < perm_K_indices # (BLOCK_M, BLOCK_N)
                    kv_mask |= mask

                qk = qk + tl.where(kv_mask, -1e6, 0)
                local_m = tl.max(qk, 1)
                m_ij = tl.maximum(m_i, local_m)
                qk -= m_ij[:, None]

                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(m_i-m_ij)

                acc = acc * alpha[:, None]

                # Indirect V gather: read BLOCK_N rows from the unpermuted V
                # tensor using perm_K_indices as row offsets.
                v_offsets = (
                    k_rows[:, None] * stride_seq_v
                    + head_offsets[None, :] * stride_d_v
                )  # (BLOCK_N, HEAD_DIM)
                v = tl.load(
                    V_base + v_offsets,
                    mask=k_inbounds[:, None],
                    other=0.0,
                ).to(dtype)
                p = p.to(dtype)

                acc += tl.dot(p, v)
                l_i = l_i * alpha + l_ij
                m_i = m_ij

            perm_K_indices_ptrs_cur = tl.advance(perm_K_indices_ptrs_cur, (0, BLOCK_N))

    return acc, l_i, m_i


def _prune_invalid_configs_permuted(configs, named_args, **kwargs):
    # Triton passes autotune key args via kwargs; fall back to named_args if needed
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
        # Require both tiles to divide logical block size to avoid crossing logical-block boundaries
        if (logical_bs % bm == 0) and (logical_bs % bn == 0):
            pruned.append(conf)
    return pruned


_PBS_FULL_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'num_warps': 4, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 3}),
]
_PBS_FIXED_CONFIG = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 2}),
]
_PBS_NO_AUTOTUNE = os.environ.get("PBS_NO_AUTOTUNE", "false").lower() in ("1", "true", "yes")

@triton.autotune(
    configs=_PBS_FIXED_CONFIG if _PBS_NO_AUTOTUNE else _PBS_FULL_AUTOTUNE_CONFIGS,
    key=['H', 'HEAD_DIM', 'LOGICAL_BLOCK_SIZE', 'SEGMENT_SIZE', 'num_kv_groups'],
    prune_configs_by={'early_config_prune': _prune_invalid_configs_permuted},
)
@triton.jit
def _permuted_block_sparse_attn_fwd(
    perm_Q, perm_K, perm_V, perm_O, # (batch_size, num_q_heads(num_kv_heads), q_len(kv_len), head_dim)
    K_block_indices, # (batch_size, num_q_heads, num_q_blocks, num_k_blocks)
    perm_Q_indices, # (batch_size, num_q_heads, q_len)
    perm_K_indices, # (batch_size, num_kv_heads, kv_len)
    stride_bz_q, stride_h_q, stride_seq_q, stride_d_q,
    stride_bz_k, stride_h_k, stride_seq_k, stride_d_k,
    stride_bz_v, stride_h_v, stride_seq_v, stride_d_v,
    stride_bz_o, stride_h_o, stride_seq_o, stride_d_o,
    stride_bz_k_block_indices, stride_h_k_block_indices, stride_seqq_k_block_indices, stride_seqk_k_block_indices,
    stride_bz_perm_q_indices, stride_h_perm_q_indices, stride_seq_perm_q_indices,
    stride_bz_perm_k_indices, stride_h_perm_k_indices, stride_seq_perm_k_indices,
    qo_len, kv_len,
    softmax_scale,
    H:tl.constexpr,
    num_kv_groups:tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr,
    STAGE: tl.constexpr
):

    # Enforce tile alignment with logical block size at compile-time
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_M) == 0)
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_N) == 0)
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_bz = tl.program_id(2).to(tl.int64)


    range_Q_seq = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    range_KV_seq = tl.arange(0, BLOCK_N)
    range_h = tl.arange(0, HEAD_DIM)

    dtype = perm_Q.type.element_ty

    # Init ptrs
    Q_ptrs = tl.make_block_ptr(
        base=perm_Q+pid_bz*stride_bz_q+pid_h*stride_h_q,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_seq_q, stride_d_q),
        offsets=(pid_seq*BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    # K and V are passed unpermuted; the inner kernel gathers them via
    # perm_K_indices. K is expanded to num_q_heads (apply_permutation needs
    # per-Q-head matmul ranking), so K is indexed by pid_h directly. V is
    # kept compact (num_kv_heads) — addressed by (pid_h // num_kv_groups) so
    # GQA models share KV-head storage.
    K_base = perm_K + pid_bz * stride_bz_k + pid_h * stride_h_k
    V_base = perm_V + pid_bz * stride_bz_v + (pid_h // num_kv_groups) * stride_h_v
    O_ptrs = tl.make_block_ptr(
        base=perm_O + pid_bz * stride_bz_o + pid_h * stride_h_o,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_seq_o, stride_d_o),
        offsets=(pid_seq * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    # Calculate the starting query token index for this program
    q_start_index = pid_seq * BLOCK_M
    # Calculate the logical query block this program belongs to.
    # Because we asserted LOGICAL_BLOCK_SIZE % BLOCK_M == 0, this is guaranteed to be a single value.
    logical_q_block_idx = q_start_index // LOGICAL_BLOCK_SIZE
    K_block_indices_ptrs = K_block_indices + pid_bz*stride_bz_k_block_indices + pid_h*stride_h_k_block_indices + logical_q_block_idx*stride_seqq_k_block_indices
    # K_block_indices_ptrs = K_block_indices + (pid_bz * (H // num_kv_groups) + pid_h // num_kv_groups) * stride_h_k_block_indices + pid_seq*stride_seqq_k_block_indices
    perm_Q_indices_ptrs = tl.make_block_ptr(
        base=perm_Q_indices + pid_bz*stride_bz_perm_q_indices + pid_h*stride_h_perm_q_indices,
        shape=(qo_len, tl.constexpr(1)),
        strides=(stride_seq_perm_q_indices, tl.constexpr(0)),
        offsets=(pid_seq*BLOCK_M, 0),
        block_shape=(BLOCK_M, tl.constexpr(1)),
        order=(1, 0)
    ) # (BLOCK_M, 1)
    perm_K_indices_ptrs = tl.make_block_ptr(
        base=perm_K_indices + pid_bz*stride_bz_perm_k_indices + pid_h*stride_h_perm_k_indices,
        shape=(tl.constexpr(1), kv_len),
        strides=(tl.constexpr(0), stride_seq_perm_k_indices),
        offsets=(0, 0),
        block_shape=(tl.constexpr(1), BLOCK_N),
        order=(1, 0)
    )

    # Init accumulators
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    softmax_scale *= 1.44269504  # 1/log(2)
    # Load Q, iterate over K
    q = tl.load(Q_ptrs, boundary_check=(0,1), padding_option="zero")

    # Off-diagonal segments if causal
    acc, l_i, m_i = _permuted_block_sparse_attn_fwd_inner(
        acc, l_i, m_i,
        q,
        qo_len,
        kv_len,
        K_base,
        V_base,
        K_block_indices_ptrs,
        perm_Q_indices_ptrs,
        perm_K_indices_ptrs,
        stride_seq_k, stride_d_k,
        stride_seq_v, stride_d_v,
        stride_bz_perm_q_indices, stride_h_perm_q_indices, stride_seq_perm_q_indices,
        stride_bz_perm_k_indices, stride_h_perm_k_indices, stride_seq_perm_k_indices,
        pid_seq,
        range_Q_seq,
        range_KV_seq,
        softmax_scale,
        dtype,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        SEGMENT_SIZE,
        LOGICAL_BLOCK_SIZE,
        4 - STAGE,
    )

    if STAGE != 1: # causal==True
        # On-diagonal segments
        acc, l_i, m_i = _permuted_block_sparse_attn_fwd_inner(
            acc, l_i, m_i,
            q,
            qo_len,
            kv_len,
            K_base,
            V_base,
            K_block_indices_ptrs,
            perm_Q_indices_ptrs,
            perm_K_indices_ptrs,
            stride_seq_k, stride_d_k,
            stride_seq_v, stride_d_v,
            stride_bz_perm_q_indices, stride_h_perm_q_indices, stride_seq_perm_q_indices,
            stride_bz_perm_k_indices, stride_h_perm_k_indices, stride_seq_perm_k_indices,
            pid_seq,
            range_Q_seq,
            range_KV_seq,
            softmax_scale,
            dtype,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            SEGMENT_SIZE,
            LOGICAL_BLOCK_SIZE,
            2,
        )

    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(dtype), boundary_check=(0,1))


def _permuted_block_sparse_attn_fwd_torch_naive(
        perm_Q, perm_K, perm_V, perm_O, # (batch_size, num_q_heads(num_kv_heads), q_len(kv_len), head_dim)
        perm_Q_indices, perm_K_indices, perm_V_indices, # (batch_size, num_q_heads, q_len(kv_len))
        block_mask,
        block_size=128,
        segment_size=1024,
        causal=True
):

    batch_size, num_q_heads, q_len, head_dim = perm_Q.shape
    batch_size, num_kv_heads, kv_len, head_dim = perm_K.shape
    assert num_q_heads == num_kv_heads
    assert causal
    num_q_blocks = (q_len + block_size - 1) // block_size
    num_k_blocks = (kv_len + block_size - 1) // block_size
    # print(f"block_mask[0][0][0]: {block_mask[0][0][0]}")
    for b in range(batch_size):
        for h in range(num_q_heads):
            for q_block_idx in range(num_q_blocks):
                # Get query block
                q_start = q_block_idx * block_size
                q_end = min(q_start + block_size, q_len)
                q_block = perm_Q[b, h, q_start:q_end, :]  # (block_size, head_dim)

                # Find selected key blocks
                selected_k_block_indices = torch.where(block_mask[b, h, q_block_idx, :] == True)[0]

                if len(selected_k_block_indices) == 0:
                    # No selected blocks, output zeros
                    perm_O[b, h, q_start:q_end, :] = 0
                    continue

                # Concatenate selected key tokens
                selected_k_blocks = []
                selected_v_blocks = []
                selected_k_indices_blocks = []
                for k_block_idx in selected_k_block_indices:
                    k_start = k_block_idx * block_size
                    k_end = min(k_start + block_size, kv_len)
                    k_block = perm_K[b, h, k_start:k_end, :]  # (block_size, head_dim)
                    v_block = perm_V[b, h, k_start:k_end, :]  # (block_size, head_dim)
                    k_indices_block = perm_K_indices[b, h, k_start:k_end]  # (block_size,)
                    selected_k_blocks.append(k_block)
                    selected_v_blocks.append(v_block)
                    selected_k_indices_blocks.append(k_indices_block)

                # Concatenate all selected blocks
                concat_k = torch.cat(selected_k_blocks, dim=0)  # (total_selected_tokens, head_dim)
                concat_v = torch.cat(selected_v_blocks, dim=0)  # (total_selected_tokens, head_dim)
                concat_k_indices = torch.cat(selected_k_indices_blocks, dim=0)  # (total_selected_tokens,)

                # Get query indices for causal masking
                q_indices = perm_Q_indices[b, h, q_start:q_end]  # (block_size,)

                # Compute attention scores
                scores = torch.matmul(q_block, concat_k.transpose(-2, -1))  # (block_size, total_selected_tokens)
                scores = scores / math.sqrt(head_dim)

                # Apply causal mask: query positions >= key positions
                causal_mask = q_indices.unsqueeze(-1) >= concat_k_indices.unsqueeze(0)  # (block_size, total_selected_tokens)
                scores = scores.masked_fill(~causal_mask, float('-inf'))

                # Apply softmax
                attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(perm_Q.dtype)  # (block_size, total_selected_tokens)

                # Compute output
                output_block = torch.matmul(attn_weights, concat_v)  # (block_size, head_dim)
                # Store output
                perm_O[b, h, q_start:q_end, :] = output_block

    return perm_O


# =============================================================================
# Block pooling and selection helpers
# (vendored from pbs_attn/src/utils.py)
# =============================================================================

def block_pooled_attn(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int,
    mask: torch.Tensor,
    query_pool_mode: str = "mean",
    key_pool_mode: str = "mean",
):
    batch_size, num_q_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, kv_len, _ = key_states.shape
    q_block_num = q_len // block_size
    kv_block_num = kv_len // block_size
    if query_pool_mode == "mean":
        pooled_query_states = query_states.reshape(batch_size, num_q_heads, q_block_num, block_size, head_dim).mean(dim=-2)
    elif query_pool_mode == "max":
        pooled_query_states = query_states.reshape(batch_size, num_q_heads, q_block_num, block_size, head_dim).max(dim=-2).values
    else:
        raise ValueError(f"Invalid query_pool_mode: {query_pool_mode}")
    if key_pool_mode == "mean":
        pooled_key_states = key_states.reshape(batch_size, num_kv_heads, kv_block_num, block_size, head_dim).mean(dim=-2)
    elif key_pool_mode == "max":
        pooled_key_states = key_states.reshape(batch_size, num_kv_heads, kv_block_num, block_size, head_dim).max(dim=-2).values
    else:
        raise ValueError(f"Invalid key_pool_mode: {key_pool_mode}")
    block_attn_scores = torch.einsum("bhqd,bhkd->bhqk", pooled_query_states, pooled_key_states)
    block_attn_scores /= math.sqrt(head_dim)
    block_attn_scores = block_attn_scores.masked_fill(~mask, float("-inf"))
    block_attn_scores = F.softmax(block_attn_scores, dim=-1, dtype=torch.float32)
    return block_attn_scores



def select_blocks(
    block_attn_scores: torch.Tensor,
    threshold: Union[float, torch.Tensor],
    causal: bool = True,
) -> torch.Tensor:
    """
    Select the blocks to attend to based on cumulative attention scores.

    Args:
        block_attn_scores: (batch_size, num_heads, q_block_num, kv_block_num)
        threshold: float, the threshold for cumulative attention scores.
        causal: bool, Must be True. This implementation is only for causal selection.

    Returns:
        block_mask: (batch_size, num_heads, q_block_num, kv_block_num)
    """
    assert causal == True, "This implementation variant strictly supports causal=True."

    batch_size, num_heads, q_block_num, kv_block_num = block_attn_scores.shape
    device = block_attn_scores.device
    # fill nans to zeros
    block_attn_scores = torch.nan_to_num(block_attn_scores, nan=0.0)
    if q_block_num == 0 or kv_block_num == 0:
        return torch.zeros_like(block_attn_scores, dtype=torch.bool, device=device)

    # Step 1: Sort scores and get original indices
    sorted_scores, sorted_indices = torch.sort(block_attn_scores, dim=-1, descending=True)

    # Step 2: Calculate cumulative scores
    cumulative_scores = torch.cumsum(sorted_scores, dim=-1)

    # Step 3: Identify blocks meeting the threshold
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
    exceed_threshold = cumulative_scores >= threshold
    # Step 4: Find the first index (sorted order) where cumulative score exceeds threshold
    indices_first_exceed = torch.argmax(exceed_threshold.int(), dim=-1, keepdim=True)
    # Step 5: Check if threshold was actually met
    any_exceeds = torch.any(exceed_threshold, dim=-1, keepdim=True)

    # Step 6: Create selection mask in sorted order
    ramp = torch.arange(kv_block_num, device=device).view(1, 1, 1, kv_block_num)
    selected_mask_sorted = (ramp <= indices_first_exceed) & any_exceeds

    # Step 7: Scatter selection mask back to original block positions
    output_block_mask = torch.empty_like(block_attn_scores, dtype=torch.bool, device=device)
    output_block_mask.scatter_(dim=-1, index=sorted_indices, src=selected_mask_sorted)

    return output_block_mask


# =============================================================================
# Token permutation helpers
# (vendored from pbs_attn/src/permute_states.py)
# =============================================================================

def last_block_attn_sorting(
    queries: torch.Tensor,
    keys: torch.Tensor,
    block_size: int,
    segment_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort keys per segment by attention to the last query block.

    Args:
        queries: (B, H, S, D) queries.
        keys: (B, H, S, D) keys to permute.
        block_size: Block size for Flash Attention.
        segment_size: Segment size for permutation.

    Returns:
        (permuted_keys, indices) where permuted_keys is (B, H, S, D) and
        indices is (B, H, S).
    """

    batch_size, num_heads, seq_len, head_dim = keys.shape
    device = keys.device

    assert segment_size > 0 and block_size > 0

    num_complete_segments = seq_len // segment_size
    remainder_size = seq_len % segment_size
    complete_seq_len = num_complete_segments * segment_size

    # No-op cases: nothing to sort
    if num_complete_segments == 0 or (num_complete_segments == 1 and remainder_size == 0):
        identity_indices = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        return keys, identity_indices

    states_complete = keys[:, :, :complete_seq_len, :]
    states_remainder = keys[:, :, complete_seq_len:, :] if remainder_size > 0 else None

    # Prepare Queries
    last_block_queries = queries[:, :, -block_size:, :]

    # Compute Scores & Sort
    attn_scores = torch.matmul(last_block_queries, states_complete.transpose(-1, -2)) / math.sqrt(head_dim)
    attn_probs_complete = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
    # Free the matmul output as soon as softmax has consumed it. At 131k context
    # this releases ~1 GB before the mean / argsort / gather steps run, which
    # lowers the post-softmax memory peak. Identical results to the original
    # because the softmax output is unchanged.
    del attn_scores
    avg_key_scores_complete = attn_probs_complete.mean(dim=-2)
    del attn_probs_complete

    scores_reshaped = avg_key_scores_complete.reshape(
        batch_size, avg_key_scores_complete.shape[1], num_complete_segments, segment_size
    )

    sorted_indices = torch.argsort(scores_reshaped, dim=-1, descending=True)

    # Apply Permutation
    states_5d = states_complete.reshape(batch_size, num_heads, num_complete_segments, segment_size, head_dim)
    indices_for_gather = sorted_indices.unsqueeze(-1).expand_as(states_5d)
    sorted_states_5d = torch.gather(states_5d, dim=3, index=indices_for_gather)
    sorted_complete = sorted_states_5d.reshape(batch_size, num_heads, complete_seq_len, head_dim)

    # Final Assembly
    final_result = (
        torch.cat([sorted_complete, states_remainder], dim=2)
        if remainder_size > 0
        else sorted_complete
    )

    segment_start_offsets = torch.arange(0, complete_seq_len, segment_size, device=device).view(1, 1, num_complete_segments, 1)
    global_indices_complete = (sorted_indices + segment_start_offsets).reshape(batch_size, num_heads, complete_seq_len)

    final_indices_to_return = (
        torch.cat([
            global_indices_complete,
            torch.arange(complete_seq_len, seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        ], dim=2)
        if remainder_size > 0
        else global_indices_complete
    )

    return final_result, final_indices_to_return



def apply_permutation(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int,
    segment_size: int,

) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Permute keys using last_block_attn_sorting with the given queries.

    Returns (permuted_keys, indices).
    """
    permuted_key_states, key_indices = last_block_attn_sorting(
        queries=query_states,
        keys=key_states,
        block_size=block_size,
        segment_size=segment_size,
    )

    return permuted_key_states, key_indices


# =============================================================================
# Top-level orchestration helpers
# (vendored from pbs_attn/src/pbs.py)
# =============================================================================

def first_token_mask(
    key_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Get the block index of the first token in the first segment.

    Args:
        key_indices: Tensor of shape (batch_size, num_heads, padded_kv_len)
        block_size: Size of the block

    Returns:
        Tensor of shape (batch_size, num_heads, 1, num_kv_blocks)
    """
    first_token_mask = (key_indices.view(key_indices.shape[0], key_indices.shape[1], -1, block_size) == 0).any(dim=-1)
    return first_token_mask[:, :, None, :]



# Cache for the per-layer-invariant block-level masks. Keyed on the layout
# (q_block_num, kv_block_num, num_blocks_per_segment, causal, device-string).
# The two returned tensors are read-only — callers `view()`/broadcast over them
# but never mutate. Reused across all 32 layers of a forward pass and across
# samples that share the same context length, eliminating ~3 MB of redundant
# bool-tensor construction per layer at 131k context.
_PBS_BLOCK_MASK_CACHE: dict = {}


def _get_block_mask_layout(
    q_block_num: int,
    kv_block_num: int,
    num_blocks_per_segment: int,
    causal: bool,
    device: torch.device,
):
    key = (q_block_num, kv_block_num, num_blocks_per_segment, bool(causal), str(device))
    cached = _PBS_BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    q_block_indices = torch.arange(q_block_num, device=device).unsqueeze(1)
    kv_block_indices = torch.arange(kv_block_num, device=device).unsqueeze(0)
    segment_mask = (q_block_indices // num_blocks_per_segment) == (
        kv_block_indices // num_blocks_per_segment
    )
    mask = ~segment_mask
    if causal:
        causal_mask = kv_block_indices <= q_block_indices + (kv_block_num - q_block_num)
        mask = mask & causal_mask

    _PBS_BLOCK_MASK_CACHE[key] = (mask, segment_mask)
    return mask, segment_mask


def permuted_block_selection(
    permuted_query_states: torch.Tensor,
    permuted_key_states: torch.Tensor,
    query_indices: torch.Tensor,
    key_indices: torch.Tensor,
    block_size: int,
    segment_size: int,
    threshold: float = 0.9,
    causal: bool = True,
    force_select_first_block: bool = True,
    query_pool_mode: str = "mean",
    key_pool_mode: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform permuted block selection using the given permuted query and key states.

    Args:
        permuted_query_states (torch.Tensor): Permuted query states of shape
                                              (batch_size, num_q_heads, q_len, head_dim).
        permuted_key_states (torch.Tensor): Permuted key states of shape
                                              (batch_size, num_kv_heads, kv_len, head_dim).
        query_indices (torch.Tensor): Query indices of shape (batch_size, num_q_heads, q_len).
        key_indices (torch.Tensor): Key indices of shape (batch_size, num_kv_heads, kv_len).
        block_size (int): Block size.
        segment_size (int): Segment size.
        threshold (float): Threshold for block selection.
        causal (bool): Whether to use causal attention.

    Returns:
        tuple: (block_attn_scores, block_mask, segment_mask)
    """
    # PBS: 2.1 Block Selection (Padding)
    batch_size, num_q_heads, q_len, head_dim = permuted_query_states.shape
    batch_size, num_kv_heads, kv_len, head_dim = permuted_key_states.shape

    assert num_q_heads == num_kv_heads
    assert q_len == kv_len, "Only support prefilling for now"
    assert segment_size % block_size == 0, "segment_size must be a multiple of block_size"
    q_num_to_pad = ((q_len + block_size - 1) // block_size) * block_size - q_len
    kv_num_to_pad = ((kv_len + block_size - 1) // block_size) * block_size - kv_len

    if q_num_to_pad > 0:
        padded_query_states = torch.nn.functional.pad(permuted_query_states, (0, 0, 0, q_num_to_pad), value=0)
    else:
        padded_query_states = permuted_query_states

    if kv_num_to_pad > 0:
        padded_key_states = torch.nn.functional.pad(permuted_key_states, (0, 0, 0, kv_num_to_pad), value=0)
        pad_indices = torch.arange(kv_len, kv_len + kv_num_to_pad, device=permuted_key_states.device)
        pad_indices = pad_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_kv_heads, -1)
        pad_key_indices = torch.cat([key_indices, pad_indices], dim=-1)
    else:
        padded_key_states = permuted_key_states
        pad_key_indices = key_indices

    padded_q_len = q_len + q_num_to_pad
    padded_kv_len = kv_len + kv_num_to_pad

    # PBS: 2.2 Block Selection (Mask Init)
    q_block_num = padded_q_len // block_size
    kv_block_num = padded_kv_len // block_size
    num_blocks_per_segment = segment_size // block_size

    mask, segment_mask = _get_block_mask_layout(
        q_block_num,
        kv_block_num,
        num_blocks_per_segment,
        causal,
        permuted_query_states.device,
    )

    # PBS: 2.3 Block Selection (Mean Pooling & Attention)
    block_attn_scores = block_pooled_attn(
        padded_query_states,
        padded_key_states,
        block_size,
        mask,
        query_pool_mode=query_pool_mode,
        key_pool_mode=key_pool_mode,
    )

    # PBS: 2.4 Block Selection (Select Blocks)
    if isinstance(threshold, int) and threshold == 1:
        block_mask = mask.view(1, 1, q_block_num, kv_block_num).expand_as(block_attn_scores)
    else:
        block_mask = select_blocks(block_attn_scores, threshold, causal)

    if force_select_first_block:
        # block_mask[:, :, :, 0] = True
        block_mask |= first_token_mask(pad_key_indices, block_size)


    return block_attn_scores, block_mask, segment_mask


def permuted_block_sparse_attn_fwd(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int,
    segment_size: int,
    threshold: float,
    causal: bool = True,
    force_select_first_block: bool = True,
    use_triton: bool = True,
    query_pool_mode: str = "mean",
    key_pool_mode: str = "mean",
    return_mask: bool = False,
    num_key_value_groups: int = 1,
):
    """
    Perform permuted block sparse attention forward pass.

    Adapted from upstream pbs_attn.src.pbs.permuted_block_sparse_attn_fwd with an
    extra `return_mask` flag so the wrapper can record block-mask density.

    K/V may be passed with fewer heads than Q (GQA). K is expanded only for the
    per-query-head permutation; V is gathered from the compact KV-head tensor to
    avoid materializing both repeated and permuted V at the same time.
    """
    SEGMENT_SIZE = segment_size
    LOGICAL_BLOCK_SIZE = block_size

    batch_size, num_q_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, kv_len, _ = key_states.shape
    assert num_key_value_groups >= 1
    assert num_q_heads == num_kv_heads * num_key_value_groups
    assert causal
    assert q_len == kv_len

    # Fall back to regular attention if not permuting
    if q_len <= segment_size:
        if num_key_value_groups > 1:
            key_states = _repeat_kv(key_states, num_key_value_groups)
            value_states = _repeat_kv(value_states, num_key_value_groups)
        attn_outputs = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=causal,
        )
        if return_mask:
            return attn_outputs, None
        return attn_outputs

    if num_key_value_groups > 1:
        key_states_expanded = _repeat_kv(key_states, num_key_value_groups)
    else:
        key_states_expanded = key_states

    select_start, select_end = STAT_COLLECTOR.maybe_start_select_timer(query_states)

    # PBS: 1. Permutation Phase
    perm_key_states, perm_key_indices = apply_permutation(
        query_states=query_states,
        key_states=key_states_expanded,
        block_size=block_size,
        segment_size=segment_size,
    )
    # not permuting queries
    perm_query_states = query_states
    perm_query_indices = torch.arange(q_len, device=query_states.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_q_heads, -1)
    # PBS: 2. Block Selection (consumes perm_key_states for pooled K)
    block_attn_scores, block_mask, segment_mask = permuted_block_selection(
        permuted_query_states=perm_query_states,
        permuted_key_states=perm_key_states,
        query_indices=perm_query_indices,
        key_indices=perm_key_indices,
        block_size=block_size,
        segment_size=segment_size,
        threshold=threshold,
        causal=causal,
        force_select_first_block=force_select_first_block,
        query_pool_mode=query_pool_mode,
        key_pool_mode=key_pool_mode,
    )
    del block_attn_scores
    # perm_key_states no longer needed once block selection is done; the Triton
    # kernel gathers K on-the-fly using perm_key_indices into the unpermuted
    # key_states_expanded tensor. Frees ~1 GB at 131k.
    del perm_key_states

    block_mask = block_mask | segment_mask[None, None, :, :]
    del segment_mask

    STAT_COLLECTOR.finish_select_timer(select_start, select_end)

    # PBS: 3. Attention Computation
    # The Triton kernel gathers both K and V on-the-fly using perm_key_indices,
    # so we never materialize permuted K or V tensors for it.
    perm_attn_outputs = torch.empty_like(perm_query_states, device=perm_query_states.device)
    if use_triton:
        def grid(META):
            return (triton.cdiv(q_len, META["BLOCK_M"]), num_q_heads, batch_size)

        _permuted_block_sparse_attn_fwd[grid](
            perm_query_states, key_states_expanded, value_states, perm_attn_outputs,
            block_mask,
            perm_query_indices, perm_key_indices,
            perm_query_states.stride(0), perm_query_states.stride(1), perm_query_states.stride(2), perm_query_states.stride(3),
            key_states_expanded.stride(0), key_states_expanded.stride(1), key_states_expanded.stride(2), key_states_expanded.stride(3),
            value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
            perm_attn_outputs.stride(0), perm_attn_outputs.stride(1), perm_attn_outputs.stride(2), perm_attn_outputs.stride(3),
            block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3),
            perm_query_indices.stride(0), perm_query_indices.stride(1), perm_query_indices.stride(2),
            perm_key_indices.stride(0), perm_key_indices.stride(1), perm_key_indices.stride(2),
            q_len, kv_len,
            1/math.sqrt(head_dim),
            H=num_q_heads,
            # K is pre-expanded to num_q_heads (kernel uses pid_h directly).
            # V is kept compact (num_kv_heads) — kernel uses pid_h // num_kv_groups.
            num_kv_groups=num_key_value_groups,
            HEAD_DIM=head_dim,
            SEGMENT_SIZE=SEGMENT_SIZE,
            LOGICAL_BLOCK_SIZE=LOGICAL_BLOCK_SIZE,
            STAGE=3 if causal else 1,
        )
        del key_states_expanded
        del value_states
    else:
        perm_value_states = _gather_permuted_values(
            value_states,
            perm_key_indices,
            num_key_value_groups,
        )
        # Naive fallback still uses pre-permuted K; recompute it for the test path.
        perm_key_states_naive, _ = apply_permutation(
            query_states=query_states,
            key_states=key_states_expanded,
            block_size=block_size,
            segment_size=segment_size,
        )
        del key_states_expanded
        del value_states
        perm_value_indices = perm_key_indices
        perm_attn_outputs = _permuted_block_sparse_attn_fwd_torch_naive(
            perm_query_states, perm_key_states_naive, perm_value_states, perm_attn_outputs,
            perm_query_indices, perm_key_indices, perm_value_indices,
            block_mask,
            block_size,
            segment_size,
            causal
        )

    if return_mask:
        return perm_attn_outputs, block_mask
    return perm_attn_outputs


# =============================================================================
# Monkey-patch forward integration (used by eval runners)
# =============================================================================

PBS_BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 128))
PBS_SEGMENT_SIZE = int(os.environ.get("PBS_SEGMENT_SIZE", 256))
PBS_THRESHOLD = float(os.environ.get("PBS_THRESHOLD", 0.9))
PBS_USE_TRITON = os.environ.get("PBS_USE_TRITON", "true").lower() in ("1", "true", "yes")


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _gather_permuted_values(
    value_states: torch.Tensor,
    perm_key_indices: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    batch_size, num_kv_heads, kv_len, head_dim = value_states.shape
    _, num_q_heads, perm_len = perm_key_indices.shape

    if num_key_value_groups == 1:
        gather_indices = perm_key_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        return torch.gather(value_states, 2, gather_indices)

    kv_head_indices = (
        torch.arange(num_q_heads, device=perm_key_indices.device, dtype=perm_key_indices.dtype)
        // num_key_value_groups
    )
    flat_indices = perm_key_indices + (kv_head_indices * kv_len).view(1, num_q_heads, 1)
    flat_indices = flat_indices.reshape(batch_size, num_q_heads * perm_len)

    flat_values = value_states.contiguous().view(batch_size, num_kv_heads * kv_len, head_dim)
    flat_indices = flat_indices.unsqueeze(-1).expand(-1, -1, head_dim)
    return torch.gather(flat_values, 1, flat_indices).view(batch_size, num_q_heads, perm_len, head_dim)


def reset_density_collection() -> None:
    STAT_COLLECTOR.reset_density()


def get_density_summary() -> Optional[dict]:
    return STAT_COLLECTOR.summary()


def pbs_attn_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    if hasattr(self, "q_norm") and self.q_norm is not None:
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm") and self.k_norm is not None:
        key_states = self.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    rotary_fn = get_rotary_fn(self.__class__.__module__)
    if rotary_fn is None:
        raise RuntimeError(f"Could not find apply_rotary_pos_emb for {self.__class__.__module__}")

    query_states, key_states = rotary_fn(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    _, heads, q_len, _ = query_states.shape
    _, kv_heads, _, _ = key_states.shape
    num_kv_groups = heads // kv_heads

    if q_len == 1 and past_key_values is not None:
        if num_kv_groups > 1:
            k_rep = _repeat_kv(key_states, num_kv_groups)
            v_rep = _repeat_kv(value_states, num_kv_groups)
        else:
            k_rep, v_rep = key_states, value_states
        attn_output = F.scaled_dot_product_attention(query_states, k_rep, v_rep)
    else:
        if STAT_COLLECTOR.collect_density:
            attn_output, block_mask = permuted_block_sparse_attn_fwd(
                query_states,
                key_states,
                value_states,
                block_size=PBS_BLOCK_SIZE,
                segment_size=PBS_SEGMENT_SIZE,
                threshold=PBS_THRESHOLD,
                causal=True,
                force_select_first_block=True,
                use_triton=PBS_USE_TRITON,
                return_mask=True,
                num_key_value_groups=num_kv_groups,
            )

            if block_mask is not None:
                STAT_COLLECTOR.record_block_mask(block_mask)
        else:
            attn_output = permuted_block_sparse_attn_fwd(
                query_states,
                key_states,
                value_states,
                block_size=PBS_BLOCK_SIZE,
                segment_size=PBS_SEGMENT_SIZE,
                threshold=PBS_THRESHOLD,
                causal=True,
                force_select_first_block=True,
                use_triton=PBS_USE_TRITON,
                return_mask=False,
                num_key_value_groups=num_kv_groups,
            )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None
