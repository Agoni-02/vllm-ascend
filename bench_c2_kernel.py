#!/usr/bin/env python3
"""C2 fused y+=(xA)B vs shrink+expand. Run on NPU after compiling sgmv_lora.

After an NPU Alarm / D-state hang: host-reset the card first.
Then --op shrink (old kernel) before --op lora. Do not serve 27B until shrink returns.
"""
import argparse
import sys
import time

import torch
import torch_npu


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="npu:0")
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--out", type=int, default=5120)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--op",
        choices=("shrink", "expand", "lora", "all"),
        default="all",
        help="shrink/expand = old kernels (health check). lora = C2. all = err then bench.",
    )
    args = p.parse_args()
    torch_npu.npu.set_device(args.device)
    log(f"device={args.device} op={args.op}")
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    import vllm_ascend.vllm_ascend_C as C  # noqa: F401

    log(f"loaded C {getattr(C, '__file__', C)}")

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
    torch.npu.synchronize()
    log("tensors ready")

    if args.op == "shrink":
        log("launch sgmv_shrink")
        torch.ops._C_ascend.sgmv_shrink(x, A, idx, seq, t, scale)
        torch.npu.synchronize()
        log(f"shrink ok t.abs.max={t.abs().max().item():.4f}")
        return
    if args.op == "expand":
        log("launch sgmv_shrink then sgmv_expand")
        torch.ops._C_ascend.sgmv_shrink(x, A, idx, seq, t, scale)
        torch.ops._C_ascend.sgmv_expand(t, B, idx, seq, y_ref, 0, O, 0)
        torch.npu.synchronize()
        log(f"expand ok y.abs.max={y_ref.abs().max().item():.4f}")
        return
    if args.op == "lora":
        log("launch sgmv_lora only")
        torch.ops._C_ascend.sgmv_lora(x, A, B, idx, seq, y_c2, scale, 0, O)
        torch.npu.synchronize()
        log(f"lora ok y.abs.max={y_c2.abs().max().item():.4f}")
        return

    log("launch sgmv_shrink")
    torch.ops._C_ascend.sgmv_shrink(x, A, idx, seq, t, scale)
    torch.npu.synchronize()
    log("shrink ok")
    log("launch sgmv_expand")
    torch.ops._C_ascend.sgmv_expand(t, B, idx, seq, y_ref, 0, O, 0)
    torch.npu.synchronize()
    log("expand ok")
    log("launch sgmv_lora")
    torch.ops._C_ascend.sgmv_lora(x, A, B, idx, seq, y_c2, scale, 0, O)
    torch.npu.synchronize()
    log("lora ok")
    diff = (y_c2.float() - y_ref.float()).abs()
    err = diff.max().item()
    per = diff.amax(dim=1)
    worst = int(per.argmax().item())
    n_bad = int((per > 1e-2).sum().item())
    pos = int(diff.view(-1).argmax().item())
    t_pos, h_pos = divmod(pos, O)
    log(f"max_abs_err={err:.6e} tokens={T} hidden={H} rank={R} out={O}")
    log(
        f"worst_token={worst} token_max={per[worst].item():.6e} "
        f"n_tokens_gt_1e-2={n_bad} argmax=(t={t_pos},h={h_pos}) "
        f"y_ref={y_ref[t_pos, h_pos].float().item():.6e} "
        f"y_c2={y_c2[t_pos, h_pos].float().item():.6e}"
    )
    log(f"token0_max={per[0].item():.6e} token_last_max={per[-1].item():.6e}")

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
    log(f"ref_shrink+expand={ms_ref:.4f} ms  fused={ms_c2:.4f} ms  speedup={ms_ref / ms_c2:.2f}x")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
    sys.stdout.flush()
