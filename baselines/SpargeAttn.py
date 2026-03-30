"""
SpargeAttention baseline adapted for the Prism evaluation framework.

Original: https://github.com/thu-nics/SpargeAttn
Paper: SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference

Key idea: mean-pooling + intra-block cosine similarity filtering + CDF-based
top-p block selection, followed by INT8-quantised block-sparse attention.
"""

import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from prism.utils.stat_collector import StatCollector

STAT_COLLECTOR = StatCollector.from_env(method_name="sparge")


def reset_select_time_collection() -> None:
    STAT_COLLECTOR.reset_select_time()


def drain_select_time_ms() -> float:
    return STAT_COLLECTOR.drain_select_time_ms()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hyperparameter_check(hyper, H, device):
    if type(hyper) == float or type(hyper) == int:
        hyper = torch.full((H,), float(hyper), device=device)
    elif isinstance(hyper, Tensor):
        assert len(hyper.shape) <= 1, "Hyperparameter tensor must be 1D"
        if len(hyper.shape) == 0:
            hyper = torch.full((H,), hyper.item(), device=device)
        assert hyper.numel() == H, f"Hyperparameter tensor must have {H} elements, but has {hyper.numel()}"
        hyper = hyper.to(device)
    else:
        raise ValueError("Hyperparameter must be a float or a tensor")
    return hyper


# ---------------------------------------------------------------------------
# Triton kernels – pooling + intra-block similarity
# ---------------------------------------------------------------------------

@triton.jit
def triton_bmm_pool_sim_simmean(
    x_ptr, pool_ptr, sim_ptr, simthreshd1,
    N: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb * BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    x = tl.load(x_ptrs, mask=xmask)
    BS_ = BS if (N - nb * BS) >= BS else (N - nb * BS)

    cur_h1 = tl.load(simthreshd1 + h)
    x_fp32 = x.to(tl.float32)
    pool = tl.sum(x_fp32, axis=0) / BS_
    x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
    x = (x / x_norm).to(tl.float16)

    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = b * H * NB + h * NB + nb
    tl.store(sim_ptr + sim_offset, cur_sim)


def get_pool_sim_triton_simmean(x, block_size, simthreshd1):
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
    grid = (B, H, nblock)
    triton_bmm_pool_sim_simmean[grid](x, pool, sim_blocks, simthreshd1, N=N, D=D, BS=block_size)
    return pool, sim_blocks


# ---------------------------------------------------------------------------
# Triton kernels – block map construction
# ---------------------------------------------------------------------------

@triton.jit
def triton_fill_block_map_kernel(final_map, num_to_select, sorted_indices, NK: tl.constexpr):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * Q * NK + h * Q * NK + q * NK
    cur_num_to_select = (cur_num_to_select + 1) if cur_num_to_select == 0 else cur_num_to_select
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 1)


def fill_block_map_triton(final_map, num_to_select, sorted_indices):
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map


@triton.jit
def triton_fill_causal_mask(mask, BqdivBk):
    q, k = tl.program_id(0), tl.program_id(1)
    Q, K = tl.num_programs(0), tl.num_programs(1)
    if k >= (q + 1) * BqdivBk:
        tl.store(mask + q * K + k, 0)
    else:
        tl.store(mask + q * K + k, 1)


def fill_causal_mask_triton(mask, BqdivBk: float):
    assert mask.dim() == 2
    triton_fill_causal_mask[mask.shape](mask, BqdivBk)
    return mask


# ---------------------------------------------------------------------------
# Block selection: mean-pooling + similarity + CDF top-p
# ---------------------------------------------------------------------------

def get_block_map_meansim(
    q, k,
    is_causal=False,
    BLKQ=128, BLKK=64,
    simthreshd1=0.1, cdfthreshd=0.9,
    attention_sink=False,
):
    Headnum = q.size(1)
    simthreshd1 = hyperparameter_check(simthreshd1, Headnum, q.device)
    cdfthreshd = hyperparameter_check(cdfthreshd, Headnum, q.device)
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_qblocks, sim_qblocks = get_pool_sim_triton_simmean(q, BLKQ, simthreshd1)
    pooled_kblocks, sim_kblocks = get_pool_sim_triton_simmean(k, BLKK, simthreshd1)

    sim_kblocks = sim_kblocks.unsqueeze(-2).expand(-1, -1, nq, -1)
    sim_qblocks = sim_qblocks.unsqueeze(-1).expand(-1, -1, -1, nk)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5
    pooled_score[~sim_kblocks] = -torch.inf
    if is_causal:
        nq = pooled_qblocks.shape[-2]
        nk = pooled_kblocks.shape[-2]
        empty_mask = torch.empty(nq, nk, device=q.device, dtype=torch.bool)
        causal_mask = fill_causal_mask_triton(empty_mask, BLKQ / BLKK)
        pooled_score = pooled_score.masked_fill(~causal_mask[None, None, ...], -torch.inf)
    pooled_score = pooled_score.softmax(-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape
    cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1).expand(B, -1, Q, 1).contiguous()
    num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)
    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map[~sim_kblocks] = 1
    final_map[~sim_qblocks] = 1
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    if is_causal:
        final_map = final_map * causal_mask[None, None, ...]
    if attention_sink:
        final_map[:, :, :, 0] = 1
    return final_map


# ---------------------------------------------------------------------------
# Triton kernels – per-block INT8 quantization
# ---------------------------------------------------------------------------

@triton.jit
def quant_per_block_int8_kernel(
    Input, Output, Scale, L,
    stride_iz, stride_ih, stride_in,
    stride_oz, stride_oh, stride_on,
    stride_sz, stride_sh,
    sm_scale,
    C: tl.constexpr, BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.0
    scale += 0.0000001
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


def per_block_int8(q, k, BLKQ=128, BLKK=64, sm_scale=None):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    b, h_qo, qo_len, head_dim = q.shape
    _, h_kv, kv_len, _ = k.shape

    stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
    stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
    stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
    stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ, 1), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK, 1), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_int8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ,
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_int8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK,
    )

    return q_int8, q_scale, k_int8, k_scale


# ---------------------------------------------------------------------------
# Triton kernels – INT8 block-sparse attention forward
# ---------------------------------------------------------------------------

@triton.jit
def _attn_fwd_inner(
    acc, l_i, old_m, q, q_scale, kv_len,
    K_ptrs, K_bid_ptr, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
    pvthreshd, start_m,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    elif STAGE == 3:
        lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        kbid = tl.load(K_bid_ptr + start_n // BLOCK_N)
        if kbid:
            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs, mask=k_mask)
            k_scale = tl.load(K_scale_ptr)
            qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
            if STAGE == 2:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = qk + tl.where(mask, 0, -1.0e6)
                local_m = tl.max(qk, 1)
                new_m = tl.maximum(old_m, local_m)
                qk -= new_m[:, None]
            else:
                local_m = tl.max(qk, 1)
                new_m = tl.maximum(old_m, local_m)
                qk = qk - new_m[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(old_m - new_m)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
            p = p.to(tl.float16)
            acc += tl.dot(p, v, out_dtype=tl.float16)
            old_m = new_m
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, old_m


@triton.jit
def _attn_fwd(
    Q, K, K_blkid, V, Q_scale, K_scale, PVThreshd, Out,
    stride_qz, stride_qh, stride_qn,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_on,
    stride_kbidq, stride_kbidk,
    qo_len, kv_len,
    H: tl.constexpr, num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)
    k_bid_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * stride_kbidq
    pvthreshd = tl.load(PVThreshd + off_h)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    K_bid_ptr = K_blkid + k_bid_offset + start_m * stride_kbidk
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q, q_scale, kv_len,
        K_ptrs, K_bid_ptr, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
        pvthreshd, start_m,
        BLOCK_M, HEAD_DIM, BLOCK_N,
        4 - STAGE, offs_m, offs_n,
    )
    if STAGE != 1:
        acc, l_i, _ = _attn_fwd_inner(
            acc, l_i, m_i, q, q_scale, kv_len,
            K_ptrs, K_bid_ptr, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
            pvthreshd, start_m,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2, offs_m, offs_n,
        )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))


def _sparge_attn_forward(q, k, k_block_id, v, q_scale, k_scale, pvthreshd, is_causal=False, output_dtype=torch.float16):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3 if is_causal else 1
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    b, h_qo, qo_len, head_dim = q.shape
    _, h_kv, kv_len, _ = k.shape
    stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
    stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
    stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
    stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)

    assert qo_len == kv_len, "qo_len and kv_len must be equal for causal attention"

    num_kv_groups = h_qo // h_kv

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q, k, k_block_id, v, q_scale, k_scale, pvthreshd, o,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        k_block_id.stride(1), k_block_id.stride(2),
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
        STAGE=stage,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4,
    )
    return o


# ---------------------------------------------------------------------------
# High-level prefill entry point
# ---------------------------------------------------------------------------

def SpargeAttn_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    smooth_k: bool = True,
    simthreshd1: float = 0.3,
    cdfthreshd: float = 0.96,
    pvthreshd: float = 20.0,
    attention_sink: bool = True,
    block_size: int = 128,
    return_mask: bool = False,
):
    assert query_states.size(-2) >= 128, "seq_len should be not less than 128."

    dtype = query_states.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q = query_states.contiguous().to(torch.float16)
        k = key_states.contiguous().to(torch.float16)
        v = value_states.contiguous().to(torch.float16)
    else:
        q = query_states.contiguous().to(torch.bfloat16)
        k = key_states.contiguous().to(torch.bfloat16)
        v = value_states.contiguous().to(torch.float16)

    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)

    select_start, select_end = STAT_COLLECTOR.maybe_start_select_timer(query_states)

    BLKQ = block_size
    BLKK = 64  # SpargeAttn uses 64 for K blocks

    k_block_indices = get_block_map_meansim(
        q, k,
        is_causal=True,
        BLKQ=BLKQ, BLKK=BLKK,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        attention_sink=attention_sink,
    )

    STAT_COLLECTOR.finish_select_timer(select_start, select_end)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, BLKQ=BLKQ, BLKK=BLKK)
    pvthreshd_t = hyperparameter_check(pvthreshd, q.size(1), q.device)

    attn_output = _sparge_attn_forward(
        q_int8, k_int8, k_block_indices, v, q_scale, k_scale, pvthreshd_t,
        is_causal=True, output_dtype=dtype,
    )

    if return_mask:
        return attn_output, k_block_indices
    return attn_output


# ---------------------------------------------------------------------------
# Monkey-patch forward integration (used by eval runners)
# ---------------------------------------------------------------------------
from typing import Optional

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from prism.utils.patch import get_rotary_fn


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


SPARGE_SMOOTH_K = os.environ.get("SPARGE_SMOOTH_K", "true").lower() in ("1", "true", "yes")
SPARGE_SIMTHRESHD = float(os.environ.get("SPARGE_SIMTHRESHD", 0.6))
SPARGE_CDFTHRESHD = float(os.environ.get("SPARGE_CDFTHRESHD", 0.98))
SPARGE_PVTHRESHD = float(os.environ.get("SPARGE_PVTHRESHD", 50.0))
SPARGE_ATTENTION_SINK = os.environ.get("SPARGE_ATTENTION_SINK", "true").lower() in ("1", "true", "yes")
SPARGE_BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 128))


def reset_density_collection() -> None:
    STAT_COLLECTOR.reset_density()


def get_density_summary() -> Optional[dict]:
    return STAT_COLLECTOR.summary()


def sparge_attention_forward(
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

    if heads != kv_heads:
        k_rep = _repeat_kv(key_states, heads // kv_heads)
        v_rep = _repeat_kv(value_states, heads // kv_heads)
    else:
        k_rep = key_states
        v_rep = value_states

    if q_len == 1 and past_key_values is not None:
        attn_output = F.scaled_dot_product_attention(query_states, k_rep, v_rep)
    else:
        if STAT_COLLECTOR.collect_density:
            attn_output, block_mask = SpargeAttn_prefill(
                query_states,
                k_rep,
                v_rep,
                smooth_k=SPARGE_SMOOTH_K,
                simthreshd1=SPARGE_SIMTHRESHD,
                cdfthreshd=SPARGE_CDFTHRESHD,
                pvthreshd=SPARGE_PVTHRESHD,
                attention_sink=SPARGE_ATTENTION_SINK,
                block_size=SPARGE_BLOCK_SIZE,
                return_mask=True,
            )
            if block_mask is not None:
                STAT_COLLECTOR.record_block_mask(block_mask)
        else:
            attn_output = SpargeAttn_prefill(
                query_states,
                k_rep,
                v_rep,
                smooth_k=SPARGE_SMOOTH_K,
                simthreshd1=SPARGE_SIMTHRESHD,
                cdfthreshd=SPARGE_CDFTHRESHD,
                pvthreshd=SPARGE_PVTHRESHD,
                attention_sink=SPARGE_ATTENTION_SINK,
                block_size=SPARGE_BLOCK_SIZE,
                return_mask=False,
            )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None
