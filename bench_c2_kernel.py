#!/usr/bin/env python3
"""C2 fused y+=(xA)B vs shrink+expand. Run on NPU after compiling sgmv_lora."""
import argparse
import time

import torch
import torch_npu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="npu:0")
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--out", type=int, default=5120)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()
    torch_npu.npu.set_device(args.device)
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    import vllm_ascend.vllm_ascend_C as C  # noqa: F401

    T, H, R, O = args.tokens, args.hidden, args.rank, args.out
    dtype = torch.bfloat16
    dev = args.device
    x = torch.randn(T, H, dtype=dtype, device=dev)
    A = torch.randn(1, 1, R, H, dtype=dtype, device=dev)
    B = torch.randn(1, 1, O, R, dtype=dtype, device=dev)
    y_ref = torch.zeros(T, O, dtype=dtype, device=dev)
    y_c2 = torch.zeros(T, O, dtype=dtype, device=dev)
    idx = torch.zeros(T, dtype=torch.int64, device=dev)
    seq = torch.tensor([T], dtype=torch.int64, device=dev)
    t = torch.zeros(T, R, dtype=torch.float32, device=dev)
    scale = 1.0

    torch.ops._C_ascend.sgmv_shrink(x, A, idx, seq, t, scale)
    torch.ops._C_ascend.sgmv_expand(t, B, idx, seq, y_ref, 0, O, 0)
    torch.ops._C_ascend.sgmv_lora(x, A, B, idx, seq, y_c2, scale, 0, O)
    err = (y_c2.float() - y_ref.float()).abs().max().item()
    print(f"max_abs_err={err:.6e} tokens={T} hidden={H} rank={R} out={O}")

    def bench(fn):
        for _ in range(10):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / args.iters * 1e3

    def run_ref():
        t.zero_()
        y_ref.zero_()
        torch.ops._C_ascend.sgmv_shrink(x, A, idx, seq, t, scale)
        torch.ops._C_ascend.sgmv_expand(t, B, idx, seq, y_ref, 0, O, 0)

    def run_c2():
        y_c2.zero_()
        torch.ops._C_ascend.sgmv_lora(x, A, B, idx, seq, y_c2, scale, 0, O)

    ms_ref = bench(run_ref)
    ms_c2 = bench(run_c2)
    print(f"ref_shrink+expand={ms_ref:.4f} ms  fused={ms_c2:.4f} ms  speedup={ms_ref / ms_c2:.2f}x")


if __name__ == "__main__":
    main()
