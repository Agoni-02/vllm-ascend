import importlib

import torch
import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import MergedColumnParallelLinearWithLoRA, MergedQKVParallelLinearWithLoRA
from vllm.lora.layers.utils import _not_fully_sharded_can_replace
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.platforms import current_platform

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)
from vllm_ascend.ops.linear import AscendQKVParallelLinear

FULL_RANK_LORA_A_ATTR = "_full_rank_lora_a"
_LINEAR_LORA_HOOKS_INSTALLED = False
_SKIP_NAME_PARTS = ("MoE", "Embedding", "Logits", "Vocab")


class _PackedLoRAAWeightsMixin(MergedColumnParallelLinearWithLoRA):
    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        super().create_lora_weights(max_loras, lora_config, model_config)
        rank = self.lora_a_stacked[0].size(2)
        self.lora_a_packed = torch.zeros(
            max_loras,
            1,
            self.n_slices * rank,
            self.input_size,
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

    def reset_lora(self, index: int) -> None:
        super().reset_lora(index)
        if hasattr(self, "lora_a_packed"):
            self.lora_a_packed[index].zero_()

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ) -> None:
        super().set_lora(index, lora_a, lora_b)
        rank = self.lora_a_stacked[0].size(2)
        for slice_index, slice_weight in enumerate(self.lora_a_stacked):
            packed_slice = self.lora_a_packed[index, 0].narrow(0, slice_index * rank, rank)
            packed_slice.copy_(slice_weight[index, 0], non_blocking=True)

    def _apply_lora_to_output(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        original_shape = output.shape if output.ndim == 3 else None
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            output,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            1.0,
            self.output_slices,
            packed_lora_a=self.lora_a_packed,
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        if original_shape is not None:
            output = output.reshape(original_shape)
        return output


class AscendMergedColumnParallelLinearWithLoRA(_PackedLoRAAWeightsMixin):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return (
            lora_config.max_loras == 1
            and type(source_layer) is maybe_get_oot_by_class(MergedColumnParallelLinear)
            and len(packed_modules_list) == 2
        )


class AscendMergedQKVParallelLinearWithLoRA(_PackedLoRAAWeightsMixin, MergedQKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return (
            lora_config.max_loras == 1
            and type(source_layer) is AscendQKVParallelLinear
            and len(packed_modules_list) == 3
        )


def refresh_all_lora_classes():
    ascend_classes = (
        AscendMergedColumnParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    existing_classes = tuple(cls for cls in vllm.lora.utils._all_lora_classes if cls not in ascend_classes)
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *existing_classes,
    )
    install_linear_lora_hooks()


def gather_sharded_lora_a(layer) -> None:
    """All-gather LoRA A on the rank dim so fused kernels see the full rank."""
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
        setattr(layer, FULL_RANK_LORA_A_ATTR, tuple(stacked))
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
    setattr(layer, FULL_RANK_LORA_A_ATTR, tuple(full))


def _wrap_after_gather(orig):
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        gather_sharded_lora_a(self)
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
    for mod_name in (
        "vllm.lora.layers.column_parallel_linear",
        "vllm.lora.layers",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        orig = getattr(mod, "_mcp_apply", None)
        if orig is None or getattr(orig, "_fused_lora_wrapped", False):
            continue

        def _mcp_apply(x, bias, layer, _orig=orig):
            a_full = getattr(layer, FULL_RANK_LORA_A_ATTR, None)
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
            return _orig(x, bias, layer)

        _mcp_apply._fused_lora_wrapped = True
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
    if getattr(cls, "_fused_lora_apply_wrapped", False):
        return
    apply = getattr(cls, "apply", None)
    if apply is None:
        return
    if "_mcp_apply" in _code_names(apply):
        return
    if "add_lora_linear" not in _code_names(apply) and not hasattr(cls, "_apply_lora_to_output"):
        return

    orig_apply = apply

    def apply_fused(self, x, bias=None):
        a_full = getattr(self, FULL_RANK_LORA_A_ATTR, None)
        if a_full is None:
            return orig_apply(self, x, bias)
        if hasattr(self, "_get_quant_method"):
            output = self._get_quant_method().apply(self.base_layer, x, bias)
        else:
            output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        original_shape = output.shape if output.ndim == 3 else None
        x2, y2 = x, output
        if x.ndim == 3 and output.ndim == 3:
            y2 = output.flatten(0, 1)
            x2 = x.flatten(0, 1)
        self.punica_wrapper.add_lora_linear(
            y2,
            x2,
            a_full,
            self.lora_b_stacked,
            1.0,
            self.output_slices,
            full_rank_lora_a=a_full,
        )
        if original_shape is not None:
            return y2.reshape(original_shape)
        return output

    cls.apply = apply_fused
    cls._fused_lora_apply_wrapped = True


def install_linear_lora_hooks():
    """Gather sharded A at set_lora; fused sgmv_lora/bgmv_lora on the linear path."""
    global _LINEAR_LORA_HOOKS_INSTALLED
    _patch_mcp_apply()
    for cls in _iter_linear_lora_classes():
        if getattr(cls, "_fused_lora_hooked", False):
            _patch_add_lora_linear_sites(cls)
            continue
        for name in ("create_lora_weights", "set_lora", "reset_lora", "set_mapping"):
            orig = getattr(cls, name, None)
            if orig is None or not callable(orig):
                continue
            setattr(cls, name, _wrap_after_gather(orig))
        _patch_add_lora_linear_sites(cls)
        cls._fused_lora_hooked = True
    _LINEAR_LORA_HOOKS_INSTALLED = True
