import torch
import math
import os
from sglang.srt.layers.attention import attention_registry
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt import server_args as server_args_mod

# Import Prism logic
try:
    from prism.prism import (
        get_high_freq_components, get_low_freq_components,
        LOW_FREQ_DIM, HIGH_FREQ_DIM, BLOCK_SIZE,
        LOW_FREQ_THRESHOLD, HIGH_FREQ_THRESHOLD,
        LOW_FREQ_TEMP, HIGH_FREQ_TEMP, FORCE_SINK, FORCE_RECENT,
        USE_TRITON_SELECT, USE_TRITON_LOGITS,
        repeat_kv, top_p_select
    )
    from prism.kernels.block_sparse_attn_paged import block_sparse_attention_paged
    from prism.kernels.prism_estimation import dual_band_softmax_triton, top_p_selection_triton
    from prism.kernels.paged_kernels import mean_pool_paged
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from prism.prism import (
        get_high_freq_components, get_low_freq_components,
        LOW_FREQ_DIM, HIGH_FREQ_DIM, BLOCK_SIZE,
        LOW_FREQ_THRESHOLD, HIGH_FREQ_THRESHOLD,
        LOW_FREQ_TEMP, HIGH_FREQ_TEMP, FORCE_SINK, FORCE_RECENT,
        USE_TRITON_SELECT, USE_TRITON_LOGITS,
        repeat_kv, top_p_select
    )
    from prism.kernels.block_sparse_attn_paged import block_sparse_attention_paged
    from prism.kernels.prism_estimation import dual_band_softmax_triton, top_p_selection_triton
    from prism.kernels.paged_kernels import mean_pool_paged

class PrismSglBackend(TritonAttnBackend):
    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None):
        try:
            return self._forward_extend_prism(q, k, v, layer, forward_batch, save_kv_cache, sinks)
        except Exception as e:
            import logging
            import traceback
            logging.error(f"Prism forward_extend failed: {e}.")
            traceback.print_exc()
            raise

    def _forward_extend_prism(self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None):
        if layer.qk_head_dim != layer.v_head_dim:
            raise ValueError(
                f"Prism requires qk_head_dim == v_head_dim, got {layer.qk_head_dim} vs {layer.v_head_dim}."
            )

        # 1. Save KV cache into sglang's pool
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        # 2. Preparation
        batch_size = forward_batch.batch_size
        num_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim
        
        # Get lengths
        extend_seq_lens = forward_batch.extend_seq_lens
        total_seq_lens = forward_batch.seq_lens
        prefix_seq_lens = total_seq_lens - extend_seq_lens
        max_total_len = total_seq_lens.max().item()
        max_extend_len = extend_seq_lens.max().item()

        # 3. Paged Estimation
        k_pool = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        profile = os.environ.get("PRISM_PROFILE", "false").lower() in ("1", "true", "yes")
        profile_layer0 = os.environ.get("PRISM_PROFILE_LAYER0", "true").lower() in ("1", "true", "yes")
        do_profile = profile and (not q.is_cuda or getattr(layer, "layer_id", 0) == 0 or not profile_layer0)
        if do_profile and q.is_cuda:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev2 = torch.cuda.Event(enable_timing=True)
            ev3 = torch.cuda.Event(enable_timing=True)
            ev4 = torch.cuda.Event(enable_timing=True)
            ev5 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        
        # Mean pool keys directly from the pool (prefix + extension)
        k_blocks = mean_pool_paged(
            k_pool, 
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            total_seq_lens,
            block_size=BLOCK_SIZE
        )
        if do_profile and q.is_cuda:
            ev1.record()
        
        # Handle GQA for estimation
        if num_heads != num_kv_heads:
            k_blocks = repeat_kv(k_blocks, num_heads // num_kv_heads)
        
        # Normalize Q to 2D for pooling and kernel input
        if q.dim() == 2:
            q_2d = q
        elif q.dim() == 3:
            q_2d = q.reshape(q.shape[0], num_heads * head_dim)
        else:
            raise ValueError(f"Unexpected q shape {tuple(q.shape)}; expected 2D or 3D.")

        # Unflatten Q for pooling (ragged batching -> padded dense tensor)
        padded_q = torch.zeros(
            batch_size, num_heads, max_extend_len, head_dim, device=q.device, dtype=q.dtype
        )
        start_loc = 0
        for i in range(batch_size):
            length = extend_seq_lens[i].item()
            q_slice = q_2d[start_loc:start_loc+length].view(length, num_heads, head_dim)
            padded_q[i, :, :length, :] = q_slice.transpose(0, 1)
            start_loc += length
            
        q_pad_len = (BLOCK_SIZE - (max_extend_len % BLOCK_SIZE)) % BLOCK_SIZE
        if q_pad_len > 0:
            padded_q = torch.nn.functional.pad(padded_q, (0, 0, 0, q_pad_len))
            
        from prism.prism import mean_pool as contiguous_mean_pool, CALIBRATE
        q_blocks = contiguous_mean_pool(padded_q, BLOCK_SIZE)
        if do_profile and q.is_cuda:
            ev2.record()
        
        # Components
        q_h = get_high_freq_components(q_blocks, HIGH_FREQ_DIM)
        q_l = get_low_freq_components(q_blocks, LOW_FREQ_DIM)
        k_h = get_high_freq_components(k_blocks, HIGH_FREQ_DIM)
        k_l = get_low_freq_components(k_blocks, LOW_FREQ_DIM)

        # Calibration (Matching prism.py)
        if CALIBRATE:
            eps = 1e-6
            rq = torch.sqrt(torch.mean(q_blocks**2, dim=(-2, -1), keepdim=True))
            rk = torch.sqrt(torch.mean(k_blocks**2, dim=(-2, -1), keepdim=True))
            rq_h = torch.sqrt(torch.mean(q_h**2, dim=(-2, -1), keepdim=True))
            rk_h = torch.sqrt(torch.mean(k_h**2, dim=(-2, -1), keepdim=True))
            rq_l = torch.sqrt(torch.mean(q_l**2, dim=(-2, -1), keepdim=True))
            rk_l = torch.sqrt(torch.mean(k_l**2, dim=(-2, -1), keepdim=True))

            th = math.sqrt(HIGH_FREQ_DIM / head_dim) * (rq_h / rq.clamp(min=eps)) * (rk_h / rk.clamp(min=eps))
            tl = math.sqrt(LOW_FREQ_DIM / head_dim) * (rq_l / rq.clamp(min=eps)) * (rk_l / rk.clamp(min=eps))
            
            scale_h = math.sqrt(HIGH_FREQ_DIM) * th
            scale_l = math.sqrt(LOW_FREQ_DIM) * tl
        else:
            scale_h = (HIGH_FREQ_DIM**0.5) * HIGH_FREQ_TEMP
            scale_l = (LOW_FREQ_DIM**0.5) * LOW_FREQ_TEMP

        # Dual-band scoring
        probs = dual_band_softmax_triton(q_h, q_l, k_h, k_l, scale_h, scale_l)
        if do_profile and q.is_cuda:
            ev3.record()

        # Selection: Triton or Torch based on env flag
        if USE_TRITON_SELECT and USE_TRITON_LOGITS and q.is_cuda:
            block_mask = top_p_selection_triton(probs, HIGH_FREQ_THRESHOLD, LOW_FREQ_THRESHOLD)
        else:
            num_q_blocks = q_blocks.shape[2]
            if HIGH_FREQ_THRESHOLD != LOW_FREQ_THRESHOLD:
                block_mask_high = top_p_select(probs[:, :, :num_q_blocks, :], HIGH_FREQ_THRESHOLD, causal=True)
                block_mask_low = top_p_select(probs[:, :, num_q_blocks:, :], LOW_FREQ_THRESHOLD, causal=True)
            else:
                block_mask_high, block_mask_low = top_p_select(
                    probs, HIGH_FREQ_THRESHOLD, causal=True
                ).split(num_q_blocks, dim=2)
            block_mask = block_mask_low | block_mask_high

        if FORCE_SINK: block_mask[:, :, :, 0] = True
        if FORCE_RECENT:
            num_q_blocks = q_blocks.shape[2]
            for i in range(batch_size):
                p_len = prefix_seq_lens[i].item()
                shift = p_len // BLOCK_SIZE
                q_idx = torch.arange(num_q_blocks, device=block_mask.device)
                k_idx = q_idx + shift
                k_idx = torch.clamp(k_idx, max=block_mask.shape[3]-1)
                block_mask[i, :, q_idx, k_idx] = True
        if do_profile and q.is_cuda:
            ev4.record()

        # Optional: print block-mask density for debugging sparsity.
        # Enable with: PRISM_DEBUG_DENSITY=1
        if os.environ.get("PRISM_DEBUG_DENSITY", "false").lower() in ("1", "true", "yes"):
            try:
                density = block_mask.float().mean().item()
                print(
                    f"[prism][layer={getattr(layer, 'layer_id', 'NA')}] "
                    f"density={density:.4f} q_blocks={block_mask.shape[2]} k_blocks={block_mask.shape[3]} "
                    f"bs={batch_size} heads={num_heads} "
                    f"max_total_len={max_total_len} max_extend_len={max_extend_len}"
                )
            except Exception:
                pass

        # 4. Paged Sparse Attention
        v_pool = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        q_3d = q_2d.view(-1, num_heads, head_dim)
        o_3d = block_sparse_attention_paged(
            q_3d, k_pool, v_pool, block_mask,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            total_seq_lens,
            prefix_seq_lens,
            forward_batch.extend_start_loc,
            block_size=BLOCK_SIZE,
            causal=True
        )
        if do_profile and q.is_cuda:
            ev5.record()
            torch.cuda.synchronize()
            t_kpool = ev0.elapsed_time(ev1)
            t_qpool = ev1.elapsed_time(ev2)
            t_softmax = ev2.elapsed_time(ev3)
            t_select = ev3.elapsed_time(ev4)
            t_sparse = ev4.elapsed_time(ev5)
            print(
                f"[prism][profile][layer={getattr(layer, 'layer_id', 'NA')}] "
                f"k_pool_ms={t_kpool:.3f} q_pool_ms={t_qpool:.3f} "
                f"softmax_ms={t_softmax:.3f} select_ms={t_select:.3f} sparse_ms={t_sparse:.3f}"
            )
        return o_3d.reshape(q_2d.shape[0], num_heads * head_dim)

def patch_sglang():
    """Apply monkey patches to sglang."""
    # Register the backend
    attention_registry.ATTENTION_BACKENDS["prism"] = lambda runner: PrismSglBackend(runner)

    # Patch the CLI choices
    old_add_choices = server_args_mod.add_attention_backend_choices
    def patched_add_choices(choices):
        old_add_choices(choices)
        if "prism" not in choices:
            choices.append("prism")
    server_args_mod.add_attention_backend_choices = patched_add_choices
    
    print("Successfully patched Prism into SGLang.")

if __name__ == "__main__":
    patch_sglang()
