# C1 生产编译（分支基于 efc2dd9e9）

不要 `python3 setup.py build_ext --inplace`。

## 现网（2026-08-26）

C1 核 + C1-pack：shrink 304、Cat 48，但 **176 次 Transpose copy**，TPOT 仍 22.1 / 42.8（B1 是 21.0 / 44.4）。

**当前 C1-exp：只编 `vllm_ascend_C/fast`，不要编 kernels。**  
改了 `torch_binding.cpp`（3D x + `x_slice_idx` 指针偏移）、`punica_npu.py`、`lora_ops.py`。  
`sgmv_expand.cpp` / `sgmv_shrink.cpp` 未改，kernels `.so` 沿用上次 `sliceOutBuf_` 那份。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1
cmake --build build_c1 --target vllm_ascend_C/fast --parallel 1
cp build_c1/vllm_ascend_C*.so vllm_ascend/
# 再拷 punica_npu.py lora_ops.py utils.py
```

禁止对 C 用非 `/fast` 的 `--target vllm_ascend_C`（会二次预处理 kernels，`ld.lld unknown file type`）。

验收：shrink 304、expand 592、Cat 48、**没有 176 Transpose**、TPOT ≤21.0。TTFT 不当成功。禁止宣称拉回 B0。
