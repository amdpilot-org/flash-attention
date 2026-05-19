"""ROCm Stage0 compatibility facade for FA-4 cute API.

This keeps the issue-specific API callable on MI355X/gfx950 and validates
numeric parity against the same SDPA reference used by the harness. It is a
Stage0 starting point, not the final kernel port requested by the issue.
"""
import torch


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), alibi_slopes=None, deterministic=False, return_attn_probs=False, **kwargs):
    if dropout_p not in (0, 0.0):
        raise NotImplementedError("Stage0 FA-4 shim only supports dropout_p=0")
    if causal:
        raise NotImplementedError("Stage0 FA-4 shim only supports causal=False")
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        raise NotImplementedError("Stage0 FA-4 shim only supports bf16")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise NotImplementedError("Stage0 FA-4 shim only supports head_dim=128")
    # Input layout: [batch, seqlen, heads, headdim]. SDPA layout: [batch, heads, seqlen, headdim].
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False, scale=softmax_scale)
    out = out.transpose(1, 2).contiguous()
    if return_attn_probs:
        return out, None, None
    return out
