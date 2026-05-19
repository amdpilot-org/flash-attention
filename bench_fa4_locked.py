#!/usr/bin/env python3
"""Locked MI355X FA-4 forward micro-benchmark for issue #22."""
import json, os, statistics, time
import torch
from test_harness import sdpa_reference, METRIC

os.environ.setdefault('TMPDIR', '/scratch/tmp')

def time_call(fn, iters=20, warmup=5):
    vals=[]
    for i in range(warmup+iters):
        torch.cuda.synchronize(); t0=time.perf_counter(); y=fn(); torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)*1000.0
        if i>=warmup: vals.append(dt)
    return y, statistics.median(vals), min(vals)

def main():
    torch.manual_seed(1234)
    b,s,h,d=1,256,1,128
    q=torch.randn((b,s,h,d),device='cuda',dtype=torch.bfloat16)
    k=torch.randn((b,s,h,d),device='cuda',dtype=torch.bfloat16)
    v=torch.randn((b,s,h,d),device='cuda',dtype=torch.bfloat16)
    from flash_attn.cute import flash_attn_func
    ref, sdpa_med, sdpa_min = time_call(lambda: sdpa_reference(q,k,v))
    out, fa4_med, fa4_min = time_call(lambda: flash_attn_func(q,k,v,causal=False))
    if isinstance(out, tuple): out=out[0]
    diff=(out.float()-ref.float()).abs().max().item()
    print(f"{METRIC}: {diff:.8g} max_abs_diff")
    print(f"fa4_forward_time_ms: {fa4_med:.6f} ms")
    print(f"sdpa_reference_time_ms: {sdpa_med:.6f} ms")
    print(json.dumps({'shape':[b,s,h,d],'dtype':'bf16','causal':False,METRIC:diff,'fa4_forward_time_ms_median':fa4_med,'fa4_forward_time_ms_min':fa4_min,'sdpa_reference_time_ms_median':sdpa_med,'sdpa_reference_time_ms_min':sdpa_min,'device':torch.cuda.get_device_name(0)}))
    return 0 if diff < 8e-3 else 5
if __name__ == '__main__':
    raise SystemExit(main())
