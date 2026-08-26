import importlib

import torch
import vllm

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)

# Eager-built contiguous A for C1 packed shrink. Graph input; never cat on decode.
C1_PACKED_ATTR = "_c1_packed_lora_a"
# FSL: all-gather A on rank dim at set_lora so C2 fused kernel sees full rank.
C2_FULL_A_ATTR = "_c2_lora_a_full"
_C1_HOOKS_INSTALLED = False
_SKIP_NAME_PARTS = ("MoE", "Embedding", "Logits", "Vocab")


def refresh_all_lora_classes():
    ascend_classes = (
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Append the Ascend classes in a deterministic order.
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *vllm.lora.utils._all_lora_classes,
    )
    install_c1_pack_hooks()


def _c1_can_pack(stacked) -> bool:
    if not isinstance(stacked, (tuple, list)) or len(stacked) <= 1:
        return False
    if any(not torch.is_tensor(t) for t in stacked):
        return False
    r0 = stacked[0].size(-2)
    h0 = stacked[0].size(-1)
    return all(t.size(-2) == r0 and t.size(-1) == h0 for t in stacked)


def pack_lora_a_inplace(layer) -> None:
    """Copy slice A into one contiguous packed tensor. Eager only (set_lora / mapping)."""
    if torch.compiler.is_compiling():
        return
    stacked = getattr(layer, "lora_a_stacked", None)
    if not _c1_can_pack(stacked):
        return
    a0 = stacked[0]
    n = len(stacked)
    r = a0.size(-2)
    want = list(a0.shape)
    want[-2] = n * r
    packed = getattr(layer, C1_PACKED_ATTR, None)
    need_new = (
        packed is None
        or not torch.is_tensor(packed)
        or list(packed.shape) != want
        or packed.dtype != a0.dtype
        or packed.device != a0.device
    )
    if need_new:
        packed = torch.empty(want, dtype=a0.dtype, device=a0.device)
        if isinstance(layer, torch.nn.Module):
            try:
                layer.register_buffer(C1_PACKED_ATTR, packed, persistent=False)
            except (KeyError, TypeError, RuntimeError, AttributeError):
                setattr(layer, C1_PACKED_ATTR, packed)
        else:
            setattr(layer, C1_PACKED_ATTR, packed)
        packed = getattr(layer, C1_PACKED_ATTR)
    for i, src in enumerate(stacked):
        packed.narrow(-2, i * r, r).copy_(src)


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
        # A is [..., rank, hidden]; gather last dim after transpose.
        t = src.transpose(-1, -2).contiguous()
        g = tensor_model_parallel_all_gather(t, dim=-1)
        full.append(g.transpose(-1, -2).contiguous())
    setattr(layer, C2_FULL_A_ATTR, tuple(full))


def _wrap_after(orig):
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        pack_lora_a_inplace(self)
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


def _patch_mcp_apply():
    from vllm.platforms import current_platform

    try:
        from vllm.distributed import tensor_model_parallel_all_gather
    except ImportError:
        tensor_model_parallel_all_gather = None

    for mod_name in (
        "vllm.lora.layers.column_parallel_linear",
        "vllm.lora.layers",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        orig = getattr(mod, "_mcp_apply", None)
        if orig is None or getattr(orig, "_c1_pack_wrapped", False):
            continue

        def _mcp_apply(x, bias, layer, _orig=orig):
            a_full = getattr(layer, C2_FULL_A_ATTR, None)
            b0 = layer.lora_b_stacked[0] if getattr(layer, "lora_b_stacked", None) else None
            if (
                a_full is not None
                and b0 is not None
                and int(a_full[0].size(-2)) == int(b0.size(-1))
            ):
                if hasattr(layer, "_get_quant_method"):
                    output = layer._get_quant_method().apply(layer.base_layer, x, bias)
                else:
                    output = layer.base_layer.quant_method.apply(layer.base_layer, x, bias)
                original_shape = output.shape if output.ndim == 3 else None
                x2, y2 = x, output
                if x.ndim == 3 and output.ndim == 3:
                    y2 = output.flatten(0, 1)
                    x2 = x.flatten(0, 1)
                layer.punica_wrapper.add_lora_linear(
                    y2, x2, a_full, layer.lora_b_stacked, 1.0, layer.output_slices
                )
                if original_shape is not None:
                    return y2.reshape(original_shape)
                return output

            packed = getattr(layer, C1_PACKED_ATTR, None)
            n = getattr(layer, "n_slices", 1)
            if n <= 1 or tensor_model_parallel_all_gather is None:
                return _orig(x, bias, layer)

            n = layer.n_slices
            if hasattr(layer, "_get_quant_method"):
                output = layer._get_quant_method().apply(layer.base_layer, x, bias)
            else:
                output = layer.base_layer.quant_method.apply(layer.base_layer, x, bias)

            x2 = x.view(-1, x.shape[-1])
            output, out_orig_shape = output.view(-1, output.shape[-1]), output.shape
            local_rank = layer.lora_a_stacked[0].shape[2]
            buffers = torch.zeros(
                (n, x2.shape[0], local_rank),
                dtype=torch.float32,
                device=x2.device,
            )
            shrunk = layer.punica_wrapper.add_shrink(
                buffers, x2, layer.lora_a_stacked, 1.0, lora_a_packed=packed
            )
            if not current_platform.can_update_inplace():
                buffers = shrunk
            buffers = tensor_model_parallel_all_gather(buffers)
            lora_output = layer.punica_wrapper.add_expand(
                output,
                buffers,
                layer.lora_b_stacked,
                layer.output_slices,
                offset_start=0,
                add_input=True,
            )
            if not current_platform.can_update_inplace() and lora_output is not None:
                output = lora_output
            return output.view(*out_orig_shape)

        _mcp_apply._c1_pack_wrapped = True
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


def _patch_add_lora_linear_sites(cls):
    from vllm.platforms import current_platform

    def _call_add_lora_linear(self, x, output):
        kwargs = {}
        packed = getattr(self, C1_PACKED_ATTR, None)
        if packed is not None:
            kwargs["lora_a_packed"] = packed
        a_full = getattr(self, C2_FULL_A_ATTR, None)
        if a_full is not None:
            kwargs["lora_a_full"] = a_full
        return self.punica_wrapper.add_lora_linear(
            output,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            1.0,
            self.output_slices,
            **kwargs,
        )

    if hasattr(cls, "_apply_lora_to_output") and not getattr(cls, "_c1_lora_out_wrapped", False):
        def _apply_lora_to_output(self, x, output):
            original_shape = output.shape if output.ndim == 3 else None
            if x.ndim == 3 and output.ndim == 3:
                output = output.flatten(0, 1)
                x = x.flatten(0, 1)
            lora_output = _call_add_lora_linear(self, x, output)
            if not current_platform.can_update_inplace():
                output = lora_output
            if original_shape is not None:
                output = output.reshape(original_shape)
            return output

        cls._apply_lora_to_output = _apply_lora_to_output
        cls._c1_lora_out_wrapped = True
        return

    apply = getattr(cls, "apply", None)
    if apply is None or getattr(cls, "_c1_apply_wrapped", False):
        return
    if "add_lora_linear" not in _code_names(apply):
        return

    def apply_packed(self, x, bias=None):
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        original_shape = output.shape if output.ndim == 3 else None
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)
        lora_output = _call_add_lora_linear(self, x, output)
        if not current_platform.can_update_inplace():
            output = lora_output
        if original_shape is not None:
            output = output.reshape(original_shape)
        return output

    cls.apply = apply_packed
    cls._c1_apply_wrapped = True


def install_c1_pack_hooks():
    """Pack LoRA A at set_lora/mapping; pass packed tensor into compiled shrink."""
    global _C1_HOOKS_INSTALLED
    _patch_mcp_apply()
    for cls in _iter_linear_lora_classes():
        if getattr(cls, "_c1_pack_hooked", False):
            _patch_add_lora_linear_sites(cls)
            continue
        for name in ("create_lora_weights", "set_lora", "reset_lora", "set_mapping"):
            orig = getattr(cls, name, None)
            if orig is None or not callable(orig):
                continue
            setattr(cls, name, _wrap_after(orig))
        _patch_add_lora_linear_sites(cls)
        cls._c1_pack_hooked = True
    _C1_HOOKS_INSTALLED = True
