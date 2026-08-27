import importlib
import os

import torch
import vllm

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)

# FSL: all-gather A on rank dim at set_lora so C2 fused kernel sees full rank.
C2_FULL_A_ATTR = "_c2_lora_a_full"
_C3_HOOKS_INSTALLED = False
_SKIP_NAME_PARTS = ("MoE", "Embedding", "Logits", "Vocab")
_LORA_STREAM = None


def refresh_all_lora_classes():
    ascend_classes = (
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *vllm.lora.utils._all_lora_classes,
    )
    install_c3_hooks()


def unshard_lora_a_if_needed(layer) -> None:
    """Eager: gather sharded A to full rank so decode fused kernel matches B."""
    if torch.compiler.is_compiling():
        return
    stacked = getattr(layer, "lora_a_stacked", None)
    bstack = getattr(layer, "lora_b_stacked", None)
    if not isinstance(stacked, (tuple, list)) or not isinstance(bstack, (tuple, list)):
        return
    if not stacked or not bstack or not torch.is_tensor(stacked[0]):
        return
    r_a = stacked[0].size(-2)
    r_b = bstack[0].size(-1)
    if r_a >= r_b:
        setattr(layer, C2_FULL_A_ATTR, tuple(stacked))
        return
    try:
        from vllm.distributed import tensor_model_parallel_all_gather
    except ImportError:
        return
    if tensor_model_parallel_all_gather is None:
        return
    full = []
    for src in stacked:
        t = src.transpose(-1, -2).contiguous()
        g = tensor_model_parallel_all_gather(t, dim=-1)
        full.append(g.transpose(-1, -2).contiguous())
    setattr(layer, C2_FULL_A_ATTR, tuple(full))


def _wrap_after_unshard(orig):
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        unshard_lora_a_if_needed(self)
        return out

    return wrapped


def _iter_linear_lora_classes():
    seen: set[type] = set()
    classes = list(getattr(vllm.lora.utils, "_all_lora_classes", ()))
    extra_targets = (
        ("vllm.lora.layers.base_linear", "BaseLinearLayerWithLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "ColumnParallelLinearWithLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "MergedColumnParallelLinearWithLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "MergedQKVParallelLinearWithLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "ColumnParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "MergedColumnParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "QKVParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "MergedQKVParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers.row_parallel_linear", "RowParallelLinearWithLoRA"),
        ("vllm.lora.layers.row_parallel_linear", "RowParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers", "BaseLinearLayerWithLoRA"),
        ("vllm.lora.layers", "MergedColumnParallelLinearWithLoRA"),
        ("vllm.lora.layers", "MergedQKVParallelLinearWithLoRA"),
        ("vllm.lora.layers", "ColumnParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers", "MergedColumnParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers", "MergedQKVParallelLinearWithShardedLoRA"),
    )
    for mod_name, attr in extra_targets:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, attr, None)
        if isinstance(cls, type):
            classes.append(cls)
    for cls in classes:
        if not isinstance(cls, type) or cls in seen:
            continue
        name = getattr(cls, "__name__", "")
        if any(part in name for part in _SKIP_NAME_PARTS):
            continue
        seen.add(cls)
        yield cls


def _code_names(fn) -> tuple[str, ...]:
    try:
        return fn.__code__.co_names
    except AttributeError:
        return ()


def _c3_overlap_enabled() -> bool:
    # Default off: decode graphs drop side-stream LoRA from y (base-model serve).
    return os.environ.get("VLLM_ASCEND_C3_OVERLAP", "0") == "1"


def _lora_stream():
    global _LORA_STREAM
    if _LORA_STREAM is None:
        _LORA_STREAM = torch.npu.Stream()
    return _LORA_STREAM


def _base_matmul(layer, x, bias):
    if hasattr(layer, "_get_quant_method"):
        return layer._get_quant_method().apply(layer.base_layer, x, bias)
    return layer.base_layer.quant_method.apply(layer.base_layer, x, bias)


def _delta_like(x, output_slices, dtype, device):
    out_h = int(sum(output_slices))
    if x.ndim == 3:
        return torch.zeros(x.size(0), x.size(1), out_h, dtype=dtype, device=device)
    return torch.zeros(x.size(0), out_h, dtype=dtype, device=device)


def _c3_apply_overlap(layer, x, bias, a_use):
    """Cube MatMul on current stream; Vector sgmv_lora on a side stream into delta."""
    slices = layer.output_slices
    main = torch.npu.current_stream()
    side = _lora_stream()
    delta = _delta_like(x, slices, x.dtype, x.device)
    with torch.npu.stream(side):
        side.wait_stream(main)
        x2, d2 = x, delta
        if x.ndim == 3 and delta.ndim == 3:
            d2 = delta.flatten(0, 1)
            x2 = x.flatten(0, 1)
        layer.punica_wrapper.add_lora_linear(
            d2, x2, a_use, layer.lora_b_stacked, 1.0, slices
        )
    output = _base_matmul(layer, x, bias)
    main.wait_stream(side)
    if output.shape != delta.shape:
        output.add_(delta.reshape_as(output))
    else:
        output.add_(delta)
    return output


def _c3_apply_serial(layer, x, bias, a_use):
    output = _base_matmul(layer, x, bias)
    original_shape = output.shape if output.ndim == 3 else None
    x2, y2 = x, output
    if x.ndim == 3 and output.ndim == 3:
        y2 = output.flatten(0, 1)
        x2 = x.flatten(0, 1)
    layer.punica_wrapper.add_lora_linear(
        y2, x2, a_use, layer.lora_b_stacked, 1.0, layer.output_slices
    )
    if original_shape is not None:
        return y2.reshape(original_shape)
    return output


def _c3_apply(layer, x, bias):
    a_use = getattr(layer, C2_FULL_A_ATTR, None) or layer.lora_a_stacked
    if _c3_overlap_enabled() and getattr(layer, "punica_wrapper", None) is not None:
        return _c3_apply_overlap(layer, x, bias, a_use)
    return _c3_apply_serial(layer, x, bias, a_use)


def _patch_mcp_apply():
    for mod_name in (
        "vllm.lora.layers.column_parallel_linear",
        "vllm.lora.layers",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        orig = getattr(mod, "_mcp_apply", None)
        if orig is None or getattr(orig, "_c3_wrapped", False):
            continue

        def _mcp_apply(x, bias, layer, _orig=orig):
            a_full = getattr(layer, C2_FULL_A_ATTR, None)
            b0 = layer.lora_b_stacked[0] if getattr(layer, "lora_b_stacked", None) else None
            if (
                a_full is not None
                and b0 is not None
                and int(a_full[0].size(-2)) == int(b0.size(-1))
            ):
                return _c3_apply(layer, x, bias)
            return _orig(x, bias, layer)

        _mcp_apply._c3_wrapped = True
        mod._mcp_apply = _mcp_apply
        for name in dir(mod):
            cls = getattr(mod, name, None)
            if not isinstance(cls, type):
                continue
            apply = getattr(cls, "apply", None)
            if apply is None or "_mcp_apply" not in _code_names(apply):
                continue

            def apply_mcp(self, x, bias=None, _mcp=_mcp_apply):
                return _mcp(x, bias, self)

            cls.apply = apply_mcp
            cls._c3_apply_wrapped = True


def _patch_add_lora_linear_sites(cls):
    if getattr(cls, "_c3_apply_wrapped", False):
        return
    apply = getattr(cls, "apply", None)
    if apply is None:
        return
    if "_mcp_apply" in _code_names(apply):
        return

    def apply_c3(self, x, bias=None):
        return _c3_apply(self, x, bias)

    cls.apply = apply_c3
    cls._c3_apply_wrapped = True


def install_c3_hooks():
    """C3: unshard A (C2); overlap Cube MatMul with Vector sgmv_lora. No C1-exp pack."""
    global _C3_HOOKS_INSTALLED
    _patch_mcp_apply()
    for cls in _iter_linear_lora_classes():
        if getattr(cls, "_c3_hooked", False):
            _patch_add_lora_linear_sites(cls)
            continue
        for name in ("create_lora_weights", "set_lora", "reset_lora", "set_mapping"):
            orig = getattr(cls, name, None)
            if orig is None or not callable(orig):
                continue
            setattr(cls, name, _wrap_after_unshard(orig))
        _patch_add_lora_linear_sites(cls)
        cls._c3_hooked = True
    _C3_HOOKS_INSTALLED = True
