# C1 生产编译（分支基于 efc2dd9e9）

不要 `python3 setup.py build_ext --inplace`（会先编整包 aclnn）。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 若本机还有 cann-9.x 的 set_env.sh 也 source

mkdir -p build_c1 && cd build_c1
cmake .. -DSOC_VERSION=ascend910b1   # 按 npu-smi 改
cmake --build . --target vllm_ascend_kernels vllm_ascend_C -j8
```

把编出的 `libvllm_ascend_kernels.so`、`vllm_ascend_C*.so` 拷到当前 `import vllm_ascend` 的目录，重启 serve。

改动：`sgmv_shrink` 多截写出 `[n,T,R]`；`punica add_shrink` 一次 shrink。不要改 expand。

验收（同一 decode 窗）：`sgmv_shrink` 592→约 304，`sgmv_expand` 仍约 592，不应大量 Cat/Slice。tok/s 只对比 B1，到不了 B0。
