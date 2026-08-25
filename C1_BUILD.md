# C1 生产编译（分支基于 efc2dd9e9）

不要 `python3 setup.py build_ext --inplace`（会先编整包 aclnn）。

改了 `sgmv_shrink.cpp` / `torch_binding.cpp` 必须重编两个 target。只改 `punica_npu.py` 才不用编 `.so`。

FSL+TP 后每卡 rank 可能是 2：CopyOut 必须从 32 字节对齐的 UB 写出，禁止 `DataCopyPad(..., yOutLocal[s*R], ...)`。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 若本机还有 cann-9.x 的 set_env.sh 也 source

mkdir -p build_c1 && cd build_c1
cmake .. -DSOC_VERSION=ascend910b1   # 按 npu-smi 改
# kernels 必须 -j1，否则 ld.lld unknown file type
cmake --build . --target vllm_ascend_kernels -j1
cmake --build . --target vllm_ascend_C -j8
```

把编出的 `libvllm_ascend_kernels.so`、`vllm_ascend_C*.so` 拷到当前 `import vllm_ascend` 的目录，覆盖 `punica_npu.py`，重启 serve。

构图/decode 都必须走 packed shrink，**禁止** `torch.compiler.is_compiling()` 跳过。A 用 `torch.cat(...).contiguous()`，不要 `data_ptr` 缓存。不要改 expand。

验收（同一 decode 窗）：`sgmv_shrink` 592→约 304，`sgmv_expand` 仍约 592。tok/s 只对比 B1，到不了 B0。
