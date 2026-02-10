import torch
from typing import Optional, Union
import math
import json
import os
from .utils.stat_collector import StatCollector
from .utils.patch import _SUPPORTED_MODELS, apply_patch, get_rotary_fn
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
import torch.nn.functional as F
from .kernels.block_sparse_attn import block_sparse_attention
from .kernels.prism_estimation import dual_band_softmax_triton, top_p_selection_triton

LOW_FREQ_DIM = int(os.environ.get("LOW_FREQ_DIM", 96))
HIGH_FREQ_DIM = int(os.environ.get("HIGH_FREQ_DIM", 64))

BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 128))
LOW_FREQ_THRESHOLD = float(os.environ.get("LOW_FREQ_THRESHOLD", 0.95))
HIGH_FREQ_THRESHOLD = float(os.environ.get("HIGH_FREQ_THRESHOLD", 0.95))
LOW_FREQ_TEMP = float(os.environ.get("LOW_FREQ_TEMP", 1.0))
HIGH_FREQ_TEMP = float(os.environ.get("HIGH_FREQ_TEMP", 1.0))
FORCE_SINK = os.environ.get("FORCE_SINK", "true").lower() in ("1", "true", "yes")
FORCE_RECENT = os.environ.get("FORCE_RECENT", "true").lower() in ("1", "true", "yes")
CALIBRATE = os.environ.get("CALIBRATE", "true").lower() == "true"
USE_TRITON_SELECT = os.environ.get("USE_TRITON_SELECT", "true").lower() in ("1", "true", "yes")
USE_TRITON_LOGITS = os.environ.get("USE_TRITON_LOGITS", "true").lower() in ("1", "true", "yes")
STAT_COLLECTOR = StatCollector.from_env()

def mean_pool(x, block_size):
    """Mean pool states before RMS calculation."""
    bsz, heads, length, dim = x.shape
    if length % block_size != 0:
        raise ValueError(f"mean_pool expects length divisible by block_size, got {length} and {block_size}")
    nb = length // block_size
    x = x.view(bsz, heads, nb, block_size, dim)
    return x.mean(dim=-2)

def get_low_freq_components(states, low_freq_dim):
    """
    Extracts the low-frequency components from query or key states.
    low_freq_dim: the number of dimensions to extract (must be even).
    """
    head_dim = states.shape[-1]
    half_head = head_dim // 2
    half_low = low_freq_dim // 2
    part1 = states[..., half_head - half_low : half_head]
    part2 = states[..., head_dim - half_low : head_dim]
    return torch.cat([part1, part2], dim=-1)

def get_high_freq_components(states, high_freq_dim):
    """
    Extracts the high-frequency components from query or key states.
    high_freq_dim: the number of dimensions to extract (must be even).
    """
    head_dim = states.shape[-1]
    half_head = head_dim // 2
    half_high = high_freq_dim // 2
    # Lower i means higher frequency.
    part1 = states[..., : half_high]
    part2 = states[..., half_head : half_head + half_high]
    return torch.cat([part1, part2], dim=-1)

def top_p_select(
    block_attn_scores: torch.Tensor,
    threshold: Union[float, torch.Tensor],
    causal: bool = True,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """
    Select the blocks to attend to based on cumulative attention scores.

    Args:
        block_attn_scores: (batch_size, num_heads, q_block_num, kv_block_num)
        threshold: float, the threshold for cumulative attention scores.
        causal: bool, Must be True. This implementation is only for causal selection.
        topk: int, if provided, select top-k blocks for each query block.

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

    if topk is not None:
        if topk <= 0:
            return torch.zeros_like(block_attn_scores, dtype=torch.bool, device=device)
        k = min(topk, kv_block_num)
        topk_indices = torch.topk(block_attn_scores, k=k, dim=-1, largest=True, sorted=False).indices
        output_block_mask = torch.zeros_like(block_attn_scores, dtype=torch.bool, device=device)
        output_block_mask.scatter_(
            dim=-1,
            index=topk_indices,
            src=torch.ones_like(topk_indices, dtype=torch.bool, device=device),
        )
        return output_block_mask

    # Step 1: Sort scores and get original indices
    sorted_scores, sorted_indices = torch.sort(block_attn_scores, dim=-1, descending=True)

    # Step 2: Calculate cumulative scores
    cumulative_scores = torch.cumsum(sorted_scores, dim=-1)

    # Step 3: Identify blocks meeting the threshold
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
    # Step 3: Thresholding (Top-P)
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
    sorted_mask = (cumulative_scores - sorted_scores) < threshold

    # Step 4: Create selection mask in sorted order
    selected_mask_sorted = sorted_mask

    # Step 7: Scatter selection mask back to original block positions
    output_block_mask = torch.empty_like(block_attn_scores, dtype=torch.bool, device=device)
    output_block_mask.scatter_(dim=-1, index=sorted_indices, src=selected_mask_sorted)

    return output_block_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads for GQA."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@STAT_COLLECTOR.collect
def prism_block_estimate(
    query_states, 
    key_states, 
    low_freq_dim=LOW_FREQ_DIM,
    high_freq_dim=HIGH_FREQ_DIM,
    block_size=BLOCK_SIZE, 
    low_freq_threshold=LOW_FREQ_THRESHOLD, 
    high_freq_threshold=HIGH_FREQ_THRESHOLD, 
    low_freq_temp=LOW_FREQ_TEMP,
    high_freq_temp=HIGH_FREQ_TEMP,
    calibrate: bool = CALIBRATE,
    force_sink: bool = FORCE_SINK,
    force_recent: bool = FORCE_RECENT,
):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]
    # Removed strict q_len == kv_len check to support sglang incremental prefill
    # assert q_len == kv_len

    # Pooling & slicing on block level
    q_blocks = mean_pool(query_states, block_size)
    k_blocks = mean_pool(key_states, block_size)
    q_high = get_high_freq_components(q_blocks, high_freq_dim)
    q_low = get_low_freq_components(q_blocks, low_freq_dim)
    k_high = get_high_freq_components(k_blocks, high_freq_dim)
    k_low = get_low_freq_components(k_blocks, low_freq_dim)
    num_q_blocks, num_k_blocks = q_blocks.shape[2], k_blocks.shape[2]
    # Calibration
    if calibrate:
        eps = 1e-6
        rq = torch.sqrt(torch.mean(q_blocks**2, dim=(-2, -1), keepdim=True))
        rk = torch.sqrt(torch.mean(k_blocks**2, dim=(-2, -1), keepdim=True))
        rq_h = torch.sqrt(torch.mean(q_high**2, dim=(-2, -1), keepdim=True))
        rk_h = torch.sqrt(torch.mean(k_high**2, dim=(-2, -1), keepdim=True))
        rq_l = torch.sqrt(torch.mean(q_low**2, dim=(-2, -1), keepdim=True))
        rk_l = torch.sqrt(torch.mean(k_low**2, dim=(-2, -1), keepdim=True))

        th = math.sqrt(high_freq_dim / head_dim) * (rq_h / rq.clamp(min=eps)) * (rk_h / rk.clamp(min=eps))
        tl = math.sqrt(low_freq_dim / head_dim) * (rq_l / rq.clamp(min=eps)) * (rk_l / rk.clamp(min=eps))
    else:
        th = high_freq_temp
        tl = low_freq_temp

    # Dual-band scoring
    scale_h = math.sqrt(high_freq_dim) * th
    scale_l = math.sqrt(low_freq_dim) * tl
    
    if USE_TRITON_SELECT and USE_TRITON_LOGITS and query_states.is_cuda:
        # Fused scoring + select (partially fused via sequential triton calls)
        probs = dual_band_softmax_triton(q_high, q_low, k_high, k_low, scale_h, scale_l)
        block_mask = top_p_selection_triton(probs, high_freq_threshold, low_freq_threshold)
    elif USE_TRITON_LOGITS and query_states.is_cuda:
        probs = dual_band_softmax_triton(q_high, q_low, k_high, k_low, scale_h, scale_l)
        if high_freq_threshold != low_freq_threshold:
            block_mask_high = top_p_select(probs[:, :, :num_q_blocks, :], high_freq_threshold, causal=True)
            block_mask_low = top_p_select(probs[:, :, num_q_blocks:, :], low_freq_threshold, causal=True)
        else:
            block_mask_high, block_mask_low = top_p_select(probs, high_freq_threshold, causal=True).split(num_q_blocks, dim=2)
        block_mask = block_mask_low | block_mask_high
    else:
        logits = torch.empty((batch_size, num_heads, 2*num_q_blocks, num_k_blocks), device=query_states.device)
        # ... (keep the existing torch fallback)
        q_block_indices = torch.arange(num_q_blocks, device=query_states.device).unsqueeze(1)
        kv_block_indices = torch.arange(num_k_blocks, device=query_states.device).unsqueeze(0)
        causal_mask = kv_block_indices <= q_block_indices + (num_k_blocks - num_q_blocks)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        logits[:, :, :num_q_blocks, :] = (torch.einsum("bhqd,bhkd->bhqk", q_high, k_high) / scale_h).masked_fill(~causal_mask, float("-inf"))
        logits[:, :, num_q_blocks:, :] = (torch.einsum("bhqd,bhkd->bhqk", q_low, k_low) / scale_l).masked_fill(~causal_mask, float("-inf"))

        probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        if high_freq_threshold != low_freq_threshold:
            block_mask_high = top_p_select(probs[:, :, :num_q_blocks, :], high_freq_threshold, causal=True)
            block_mask_low = top_p_select(probs[:, :, num_q_blocks:, :], low_freq_threshold, causal=True)
        else:
            block_mask_high, block_mask_low = top_p_select(probs, high_freq_threshold, causal=True).split(num_q_blocks, dim=2)
        
        block_mask = block_mask_low | block_mask_high

    if force_sink:
        # keep sink block
        block_mask[:, :, :, 0] = True
    if force_recent:
        # keep recent block
        q_block_idx = torch.arange(num_q_blocks, device=block_mask.device)
        kv_block_idx = q_block_idx + (num_k_blocks - num_q_blocks)
        block_mask[:, :, q_block_idx, kv_block_idx] = True

    return block_mask


def prism_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Patched forward with prism attention.
    """
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

    bsz, heads, q_len, head_dim = query_states.shape
    bsz, kv_heads, kv_len, head_dim = key_states.shape
    
    # Expand key/value for GQA
    if heads != kv_heads:
        key_states = repeat_kv(key_states, heads // kv_heads)
        value_states = repeat_kv(value_states, heads // kv_heads)
    
    # Check for decoding phase (q_len == 1 and we have history)
    if q_len == 1 and past_key_values is not None:
        # --- Decoding: Standard Attention (Fallback) ---
        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
        )
        
    else:
        orig_q_len = q_len
        pad_q = (BLOCK_SIZE - (q_len % BLOCK_SIZE)) % BLOCK_SIZE
        pad_kv = (BLOCK_SIZE - (kv_len % BLOCK_SIZE)) % BLOCK_SIZE
        if pad_q:
            query_states = F.pad(query_states, (0, 0, 0, pad_q), value=0.0)
        if pad_kv:
            key_states = F.pad(key_states, (0, 0, 0, pad_kv), value=0.0)
            value_states = F.pad(value_states, (0, 0, 0, pad_kv), value=0.0)

        block_mask = prism_block_estimate(
            query_states, key_states,
            low_freq_temp=LOW_FREQ_TEMP, high_freq_temp=HIGH_FREQ_TEMP,
        )

        attn_output = block_sparse_attention(query_states, key_states, value_states, block_mask, block_size=BLOCK_SIZE, causal=True)
        if pad_q:
            attn_output = attn_output[:, :, :orig_q_len, :]

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Enable density collection for main run
    STAT_COLLECTOR.collect_density = True


    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
    patched_count = apply_patch(
        forward_fn=prism_attention_forward,
        model_id=model_id,
        supported_models=_SUPPORTED_MODELS,
    )
    print(f"Monkey-patch applied to {patched_count} attention classes.")

    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )    
    # Load prompt from JSON
    json_path = os.environ.get("DATA_PATH", "data/ruler_niah_multikey_3_32k_0.json")
    print(f"Loading prompt from {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data_idx = str(os.environ.get("DATA_IDX", "0"))
    print(f"Using data index: {data_idx}")

    prompt = data[data_idx]["origin_prompt"][0]["prompt"]
    gold_answer = data[data_idx]["gold"]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input tokens: {inputs.input_ids.shape[1]}")

    # Generate
    with torch.no_grad():
        generate_ids = model.generate(
            inputs.input_ids, 
            max_new_tokens=100, # Short answer expected
            do_sample=False,
        )
    
    # Decode only the new tokens
    new_tokens = generate_ids[0, inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print("-" * 30)
    print(f"Response: {response}")
    print(f"Gold Answer: {gold_answer}")
    print("-" * 30)

    # Verification
    layer_idx = 0
    attention = model.model.layers[layer_idx].self_attn
    if attention.forward.__name__ == "prism_attention_forward":
        print(f"Verification successful: model.model.layers[0].self_attn.forward is {attention.forward.__name__}")
    else:
        print(f"Verification failed: attention.forward is {attention.forward.__name__}")

    summary = STAT_COLLECTOR.summary()
    if summary:
        print("\n--- Density Statistics ---")
        print(f"Average Total Density:     {summary['total']:.4f}")
        print("-" * 26)
