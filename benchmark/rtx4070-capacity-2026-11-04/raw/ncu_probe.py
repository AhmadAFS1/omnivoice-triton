"""Finite optional hardware-counter permission probe, synthetic FP16 GEMM.

This is not model inference or a capacity result. Never change host profiling
permissions; retain a denied-access result if the rental cannot expose counters.
"""
import torch
with torch.inference_mode():
    a=torch.ones((1024,1024),device='cuda',dtype=torch.float16)
    b=torch.ones_like(a)
    c=torch.mm(a,b)
    torch.cuda.synchronize()
