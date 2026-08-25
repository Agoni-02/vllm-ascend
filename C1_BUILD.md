# C1 生产编译（分支基于 efc2dd9e9）

不要 `python3 setup.py build_ext --inplace`（会先编整包 aclnn）。

只改 `punica_npu.py` 时：**不要重编 `.so`**，覆盖 Python 后重启即可。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 若本机还有 cann-9.x 的 set_env.sh 也 source

mkdir -p build_c1 && cd build_c1
cmake .. -DSOC_VERSION=ascend910b1   # 按 npu-smi 改
# kernels 必须 -j1，否则 ld.lld unknown file type
cmake --build . --target vllm_ascend_kernels -j1
cmake --build . --target vllm_ascend_C -j8
```

把编出的 `libvllm_ascend_kernels.so`、`vllm_ascend_C*.so` 拷到当前 `import vllm_ascend` 的目录，重启 serve。

改动：`sgmv_shrink` 多截写出 `[n,T,R]`；`punica` packed 层一次 shrink。构图/decode 都必须走这条，**禁止** `torch.compiler.is_compiling()` 跳过融合。A 只用 `torch.cat`，不要 `data_ptr` 缓存。不要改 expand。

验收（同一 decode 窗）：`sgmv_shrink` 592→约 304，`sgmv_expand` 仍约 592。构图图里可以有少量 cat；不应再出现方案1那种每步大量 Cat/Slice/`copy_`。tok/s 只对比 B1，到不了 B0。
