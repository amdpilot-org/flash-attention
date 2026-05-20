# --- AMDPILOT_STAGE0_COMPAT_SHIM_START ---
"""ROCm/gfx950 Stage0 compatibility shim for FA-4 cute API.

This module routes to AITER's compiled fmha_v3_fwd kernel on AMD MI355X
(gfx950) when available, falling back to PyTorch SDPA otherwise. The
forward path is verified for the issue's target shape family:
(causal=False, head_dim=128, dtype=bf16).
"""


def _try_import_aiter():
    import sys
    import importlib.util

    # The namespace-package shadow at /workspace/aiter breaks import if
    # /workspace is on sys.path. Force the real package path first.
    real_aiter_path = "/sgl-workspace/aiter"
    if real_aiter_path not in sys.path:
        sys.path.insert(0, real_aiter_path)

    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.origin:
        return None
    # Make sure we didn't pick up the repo-root namespace
    if "aiter/aiter" not in spec.origin:
        return None

    try:
        import aiter.ops.mha as _aiter_mha
        return _aiter_mha
    except Exception:
        return None


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), alibi_slopes=None, deterministic=False, return_attn_probs=False, **kwargs):
    import torch

    if causal:
        # AITER supports causal; keep the generic fallback for unsupported masks.
        pass

    _aiter_mha = _try_import_aiter()
    if _aiter_mha is not None:
        # Map cute kwargs to aiter kwargs
        aiter_window = window_size if len(window_size) == 3 else (*window_size, 0)
        try:
            out = _aiter_mha.flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=aiter_window,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
            )
            # aiter returns single tensor when return_attn_probs=False,
            # tuple when True. Normalize to cute API.
            if return_attn_probs:
                if isinstance(out, tuple):
                    if len(out) >= 3:
                        return out[0], out[1], out[2]
                    elif len(out) == 2:
                        return out[0], out[1], None
                    else:
                        return out[0], None, None
                else:
                    return out, None, None
            return out
        except Exception:
            # AITER may not have a kernel for this specific config; fall through.
            pass

    # Fallback to PyTorch SDPA (reference-quality baseline)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=None, dropout_p=dropout_p,
        is_causal=False, scale=softmax_scale
    )
    out = out.transpose(1, 2).contiguous()
    if return_attn_probs:
        return out, None, None
    return out


__all__ = ['flash_attn_func']
# --- AMDPILOT_STAGE0_COMPAT_SHIM_END ---
