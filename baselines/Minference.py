import os
import torch
import torch.nn.functional as F
import math
from minference.ops.pit_sparse_flash_attention_v2 import (
    vertical_slash_sparse_attention,
)
from prism.utils.stat_collector import StatCollector

STAT_COLLECTOR = StatCollector.from_env(method_name="minference")


def reset_select_time_collection() -> None:
    STAT_COLLECTOR.reset_select_time()


def drain_select_time_ms() -> float:
    return STAT_COLLECTOR.drain_select_time_ms()


last_q = 64
arange = torch.arange(last_q, device="cuda")
LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]


def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device)  # Zero matrix used for padding
    mat_padded = torch.cat(
        (zero_mat, mat, zero_mat), -1
    )  # pads the matrix on left and right
    mat_strided = mat_padded.as_strided(
        (1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1)
    )  # Change the strides
    sum_diags = torch.sum(mat_strided, 2)  # Sums the resulting matrix's columns
    return sum_diags[:, :, 1:]



def Minference_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    vertical_size=512,
    slash_size=2048,
    adaptive_budget=None,
    block_size: int = 128,
    return_mask: bool = False,
):
    output = torch.empty_like(query_states)
    key_states = key_states.to(query_states.device)
    value_states = value_states.to(query_states.device)
    if adaptive_budget is not None:
        seq_len = query_states.size(2)
        budget = int(seq_len * adaptive_budget)
        vertical_size = int(budget * 0.2)
        slash_size = int(budget * 0.8)
    seq_len = query_states.size(2)
    num_blocks = math.ceil(seq_len / block_size)
    should_build_mask = return_mask
    if should_build_mask:
        block_mask = torch.zeros(
            query_states.size(0),
            query_states.size(1),
            num_blocks,
            num_blocks,
            dtype=torch.bool,
            device=query_states.device,
        )
        causal_mask = torch.tril(
            torch.ones(num_blocks, num_blocks, dtype=torch.bool, device=query_states.device)
        )

    for head in range(query_states.size(1)):
        q = query_states[:, head, :, :].unsqueeze(1).to(query_states.device)
        k = key_states[:, head, :, :].unsqueeze(1).to(query_states.device)
        v = value_states[:, head, :, :].unsqueeze(1).to(query_states.device)

        select_start, select_end = STAT_COLLECTOR.maybe_start_select_timer(query_states)

        q_len = q.shape[2]
        vertical_size, slash_size = min(q_len, max(vertical_size, 30)), min(
            q_len, max(slash_size, 50)
        )
        last_q = min(64, q_len)
        qk = torch.einsum(f"bhmk, bhnk -> bhmn", q[:, :, -last_q:, :], k) / math.sqrt(
            128
        )  # headdim
        qk[:, :, :, -last_q:] = torch.where(
            LAST_Q_MASK[..., -last_q:, -last_q:].to(q.device),
            qk[:, :, :, -last_q:],
            -torch.inf,
        )
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[..., :30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[..., : -last_q + 1]
        slash[..., -100:] = torch.inf
        slash_topk = slash
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        if should_build_mask:
            # Optimized mask building
            v_blocks = (vertical_topk // block_size).clamp(0, num_blocks - 1)
            offset_blocks = ((q_len - 1 - slash) // block_size).clamp(0, num_blocks - 1)
            
            bsz = query_states.size(0)
            # 1. Vertical mask
            v_cols_mask = torch.zeros(bsz, num_blocks, dtype=torch.bool, device=q.device)
            v_cols_mask.scatter_(1, v_blocks.view(bsz, -1), True)
            block_mask[:, head] |= v_cols_mask.unsqueeze(1) & causal_mask

            # 2. Slash mask
            # For block (r, c), it is a slash block if r - c == offset_block
            # So block_mask[b, head, r, c] = (r - c) in offset_blocks[b]
            slash_offsets_mask = torch.zeros(bsz, num_blocks, dtype=torch.bool, device=q.device)
            slash_offsets_mask.scatter_(1, offset_blocks.view(bsz, -1), True)
            slash_offsets_mask[:, 0] = True # Include diagonal
            
            r_idx = torch.arange(num_blocks, device=q.device).view(num_blocks, 1)
            c_idx = torch.arange(num_blocks, device=q.device).view(1, num_blocks)
            diff = r_idx - c_idx
            valid_diff = (diff >= 0)
            
            # Use advanced indexing to build the 2D mask for each batch
            # slash_offsets_mask: [B, num_blocks]
            # diff: [num_blocks, num_blocks]
            # We want block_mask[b, head, r, c] = slash_offsets_mask[b, diff[r, c]] if diff[r, c] >= 0
            block_mask[:, head] |= slash_offsets_mask[:, diff.clamp(min=0)] & valid_diff
        STAT_COLLECTOR.finish_select_timer(select_start, select_end)
        output[:, head : head + 1, :, :] = vertical_slash_sparse_attention(
            q, k, v, vertical_topk, slash
        )

    if return_mask:
        return output, block_mask
    return output

# -----------------------------------------------------------------------------
# Monkey-patch forward integration (used by eval runners)
# -----------------------------------------------------------------------------
from typing import Optional

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from prism.utils.patch import get_rotary_fn


def _env_to_int(name: str) -> Optional[int]:
    val = os.environ.get(name, "").strip().lower()
    if val in ("", "none", "null"):
        return None
    return int(val)


def _env_to_float(name: str) -> Optional[float]:
    val = os.environ.get(name, "").strip().lower()
    if val in ("", "none", "null"):
        return None
    return float(val)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


MINFERENCE_VERTICAL_SIZE = int(os.environ.get("MINFERENCE_VERTICAL_SIZE", 512))
MINFERENCE_SLASH_SIZE = int(os.environ.get("MINFERENCE_SLASH_SIZE", 2048))
MINFERENCE_ADAPTIVE_BUDGET = _env_to_float("MINFERENCE_ADAPTIVE_BUDGET")
MINFERENCE_BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 128))

def reset_density_collection() -> None:
    STAT_COLLECTOR.reset_density()


def get_density_summary() -> Optional[dict]:
    return STAT_COLLECTOR.summary()


def minference_attention_forward(
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
    _, kv_heads, kv_len, _ = key_states.shape

    if kv_len != q_len:
        if heads != kv_heads:
            k_rep = _repeat_kv(key_states, heads // kv_heads)
            v_rep = _repeat_kv(value_states, heads // kv_heads)
        else:
            k_rep = key_states
            v_rep = value_states
        attn_output = F.scaled_dot_product_attention(query_states, k_rep, v_rep)
    else:
        if heads != kv_heads:
            k_rep = _repeat_kv(key_states, heads // kv_heads)
            v_rep = _repeat_kv(value_states, heads // kv_heads)
        else:
            k_rep = key_states
            v_rep = value_states

        if STAT_COLLECTOR.collect_density:
            attn_output, block_mask = Minference_prefill(
                query_states,
                k_rep,
                v_rep,
                vertical_size=MINFERENCE_VERTICAL_SIZE,
                slash_size=MINFERENCE_SLASH_SIZE,
                adaptive_budget=MINFERENCE_ADAPTIVE_BUDGET,
                block_size=MINFERENCE_BLOCK_SIZE,
                return_mask=True,
            )
            if block_mask is not None:
                STAT_COLLECTOR.record_block_mask(block_mask)
        else:
            attn_output = Minference_prefill(
                query_states,
                k_rep,
                v_rep,
                vertical_size=MINFERENCE_VERTICAL_SIZE,
                slash_size=MINFERENCE_SLASH_SIZE,
                adaptive_budget=MINFERENCE_ADAPTIVE_BUDGET,
                block_size=MINFERENCE_BLOCK_SIZE,
                return_mask=False,
            )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None
