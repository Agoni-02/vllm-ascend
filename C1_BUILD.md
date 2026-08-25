# C1 生产编译（分支基于 efc2dd9e9）

不要 `python3 setup.py build_ext --inplace`。

## 现网结论（2026-08-25）

核 `[n,T,R]` + FSL 对齐 CopyOut（`sliceOutBuf_`）已接到：shrink 592→304、2.97→2.20 ms；expand 仍 592。  
热路径 `torch.cat().contiguous()` 使 Cat 48→224，数据集 TPOT 21.0→22.1、tok/s 44.4→42.4。  
**下一步 C1-pack：只改 punica，不重编 `.so`。** 禁止 `is_compiling` 跳过、禁止 `data_ptr`。Cat 须回到约 48。

## 编核（仅改 cpp 时）

kernels 成功后再编 C 须用 `vllm_ascend_C/fast`，否则会二次 `ld.lld` 已 merge 的 `.o`。  
从干净 `build_c1` 一次编两个：`--target vllm_ascend_kernels vllm_ascend_C --parallel 1`。  
只 source `cann-9.1.0`。kernels 预处理不要 `-j8`。`.so` 从 `build_c1/lib/` 和 `build_c1/vllm_ascend_C*.so` 拷到 `vllm_ascend/`，不要用 `import` 日志当路径。

验收 tok/s 只对比 B1，到不了 B0。Insight Totals 不是 TPOT。
