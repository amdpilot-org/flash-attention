"""AMD MI355X shim for the NVIDIA-only flash_attn.cute FA-4 API.

Implements flash_attn_func with the upstream [B, S, H, D] layout by dispatching
to torch SDPA (which uses AOTriton / CK fused attention on ROCm). Returned
tensor matches the layout/dtype convention expected by test harnesses.
"""
import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, causal=False, softmax_scale=None,
                    window_size=(-1, -1), alibi_slopes=None,
                    deterministic=False, return_attn_probs=False, **kwargs):
    if q.dim() != 4:
        raise ValueError(f"expected 4D [B,S,H,D] tensors, got q.shape={tuple(q.shape)}")
    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    if softmax_scale is None:
        softmax_scale = qh.shape[-1] ** -0.5
    out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=bool(causal), scale=softmax_scale)
    out = out.transpose(1, 2).contiguous()
    if return_attn_probs:
        return out, None, None
    return out
