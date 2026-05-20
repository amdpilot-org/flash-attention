#!/usr/bin/env python3
"""Stage0 FA-4 forward correctness harness for issue-flash-attention-22.
Runs inside the container against /workspace/flash-attention.
Prints required metric name verbatim: fa4_forward_correctness_max_abs_diff.
"""
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path('/workspace/flash-attention')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _call_flash_attn(q, k, v):
    try:
        from flash_attn.cute import flash_attn_func
    except Exception as e:
        raise RuntimeError(f"failed to import flash_attn.cute.flash_attn_func: {type(e).__name__}: {e}")
    try:
        return flash_attn_func(q, k, v, causal=False)
    except TypeError:
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=False)


def main():
    if not torch.cuda.is_available():
        raise SystemExit('torch cuda/hip unavailable')
    torch.manual_seed(1234)
    device = torch.device('cuda')
    # Single representative FA forward shape with head_dim 128, bf16, no-causal.
    batch, seqlen, nheads, headdim = 1, 256, 8, 128
    q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    out = _call_flash_attn(q, k, v)
    if isinstance(out, (tuple, list)):
        out = out[0]
    # Reference math: PyTorch SDPA on same ROCm host, layout B,H,S,D.
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
    ).transpose(1, 2)
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"fa4_forward_correctness_max_abs_diff: {diff}")
    print(json.dumps({
        'metric_name': 'fa4_forward_correctness_max_abs_diff',
        'metric_unit': 'max abs diff vs FA-3 port reference (bf16)',
        'metric_direction': 'lower',
        'metric_value': diff,
        'device_name': torch.cuda.get_device_name(0),
        'hip_visible_devices': os.environ.get('HIP_VISIBLE_DEVICES'),
    }, sort_keys=True))
    if diff >= 8e-3:
        raise SystemExit(f'metric above target: {diff}')

if __name__ == '__main__':
    main()
