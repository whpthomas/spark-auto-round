# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Internal factory for creating compressor instances."""

import os
from typing import Any, Callable, Optional, Union

import torch

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.transforms import normalize_rotation_config as _normalize_any_rotation_config
from auto_round.algorithms.transforms.base import BaseRotationConfig as _BaseRotationConfig
from auto_round.algorithms.transforms.rotation.config import RotationConfig as _NewArchRotationConfig
from auto_round.compressors.data_driven import DataDrivenCompressor
from auto_round.compressors.utils import check_need_act_calibration
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, _parse_scheme


# Compressor-class registry
# ---------------------------------------------------------------------------
# Maps (model_type, base_class_name) -> combined class, created lazily.
_COMPRESSOR_REGISTRY: dict[tuple[str, str], type] = {}


def _get_compressor_class(model) -> type:
    """Return the compressor class, applying MLLM mixin for multimodal models.

    For plain LLM models, returns ``DataDrivenCompressor`` unchanged.
    For MLLM models, dynamically creates a combined class with ``MLLMMixin``
    prepended so that ``_get_calibrator_kind()`` returns ``"mllm"``.
    """
    from auto_round.utils.model.detect import is_mllm_model

    base_cls = DataDrivenCompressor

    # Guard: only attempt MLLM detection for string paths or nn.Module instances
    is_mllm = False
    try:
        if isinstance(model, str):
            is_mllm = is_mllm_model(model)
        elif hasattr(model, "config"):  # nn.Module with config attr
            is_mllm = is_mllm_model(model)
    except Exception:
        # Detection failed (e.g. path not found, no network); assume non-MLLM
        pass

    key = ("mllm" if is_mllm else "llm", base_cls.__name__)

    if key in _COMPRESSOR_REGISTRY:
        return _COMPRESSOR_REGISTRY[key]

    if key[0] == "mllm":
        from auto_round.compressors.mllm_mixin import MLLMMixin

        combined = type(f"Mllm{base_cls.__name__}", (MLLMMixin, base_cls), {})
        _COMPRESSOR_REGISTRY[key] = combined
        return combined

    # Cache the plain LLM class too
    _COMPRESSOR_REGISTRY[key] = base_cls
    return base_cls


def _preview_resolved_attrs(config, scheme=None) -> dict:
    """Resolve scheme attributes without mutating config, for routing decisions."""
    scheme_attr_names = QuantizationScheme.get_attributes()
    user_overrides = {k: getattr(config, k) for k in scheme_attr_names if getattr(config, k, None) is not None}
    try:
        _, final_attrs = _parse_scheme(scheme, user_overrides)
        return final_attrs
    except Exception:
        logger.debug("_preview_resolved_attrs failed for %s", attr, exc_info=True)
        return {}


def _eager_validate_scheme(config, scheme=None) -> None:
    """Eagerly validate scheme/config constraints at construction time."""
    scheme_attr_names = QuantizationScheme.get_attributes()
    user_overrides = {k: getattr(config, k) for k in scheme_attr_names if getattr(config, k, None) is not None}
    try:
        _, final_attrs = _parse_scheme(scheme, user_overrides)
    except (ValueError, NotImplementedError):
        raise
    except Exception:
        logger.debug("_eager_validate_scheme failed for %s", attr, exc_info=True)
        return  # Other parse errors are deferred to post_init

    import copy

    temp_config = copy.copy(config)
    for key, value in final_attrs.items():
        setattr(temp_config, key, value)
    temp_config.check_config()  # raises ValueError / NotImplementedError if invalid


def _resolve_config(config: Union[str, AlgConfig, list]) -> Union[AlgConfig, list[AlgConfig]]:
    """Convert string alias(es) to the corresponding config instance(s)."""
    _CONFIG_ALIASES = {
        "sign_round": SignRoundConfig,
        "signround": SignRoundConfig,
        "hadamard": _NewArchRotationConfig,
    }
    if isinstance(config, str):
        key = config.strip().lower()
        # Handle spinquant/quarot via unified normalizer
        if key in ("spinquant", "quarot"):
            return _normalize_any_rotation_config(key)
        if key not in _CONFIG_ALIASES:
            raise ValueError(f"Unknown config alias '{config}'. " f"Supported: {list(_CONFIG_ALIASES.keys())}")
        return _CONFIG_ALIASES[key]()
    if isinstance(config, list):
        return [_resolve_config(c) for c in config]
    return config


def auto_round_factory(
    alg_configs: Union[str, AlgConfig, list[Union[str, AlgConfig]]] = None,
    model: Union[torch.nn.Module, str] = None,
    tokenizer=None,
    platform="hf",
    format=None,
    scheme="W4A16",
    low_gpu_mem_usage: bool = False,
    device_map: Union[str, torch.device, int, dict] = 0,
    iters: int = None,
    gradient_accumulate_steps: int = 1,
    enable_torch_compile: bool = False,
    seed: int = 42,
    low_cpu_mem_usage: bool = True,
    layer_config=None,
    nsamples: int = None,
    seqlen: int = None,
    **kwargs,
) -> DataDrivenCompressor:
    """Create a DataDrivenCompressor with the given configuration.

    This is the internal factory that both the public ``AutoRound()`` function
    and the backward-compatible ``AutoRoundCompatible()`` function delegate to.
    """
    from auto_round.algorithms.quantization.config import QuantizationConfig

    # Resolve string alias(es) to config instance(s) before routing.
    alg_configs = _resolve_config(alg_configs)

    # When no config is provided, create a default SignRoundConfig.
    if alg_configs is None:
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        alg_configs = SignRoundConfig(iters=iters or 200)

    # Extract the single QuantizationConfig from a list; validate at most one exists.
    if isinstance(alg_configs, list):
        quant_configs = [c for c in alg_configs if isinstance(c, QuantizationConfig)]
        if len(quant_configs) == 0:
            raise ValueError("At least one QuantizationConfig (SignRoundConfig / RTNConfig) is required.")
        if len(quant_configs) > 1:
            raise ValueError(
                f"Only one QuantizationConfig is allowed, but got {len(quant_configs)}: "
                f"{[type(c).__name__ for c in quant_configs]}"
            )
        quant_config = quant_configs[0]
    else:
        quant_config = alg_configs

    # Eagerly validate scheme constraints that do not require model info.
    _eager_validate_scheme(quant_config, scheme)

    # Explicitly build the dict of constructor args to forward to the compressor.
    local_args = dict(
        model=model,
        tokenizer=tokenizer,
        platform=platform,
        format=format,
        scheme=scheme,
        low_gpu_mem_usage=low_gpu_mem_usage,
        device_map=device_map,
        iters=iters,
        gradient_accumulate_steps=gradient_accumulate_steps,
        enable_torch_compile=enable_torch_compile,
        seed=seed,
        low_cpu_mem_usage=low_cpu_mem_usage,
        layer_config=layer_config,
        nsamples=nsamples,
        seqlen=seqlen,
    )

    # Pop kwargs that are only consumed by specific Mixins or are legacy/unused
    # (MLLM kwargs are preserved below and forwarded to the combined class)
    for _k in (
        "disable_opt_rtn", "use_meta_device",
        # Legacy kwargs from old API that are no longer used
        "enable_adam", "extra_config", "not_use_best_mse", "momentum",
        "rotation_config", "algorithm", "enable_alg_ext",
    ):
        kwargs.pop(_k, None)

    # Only SignRoundConfig is supported
    if not isinstance(quant_config, SignRoundConfig):
        raise ValueError(
            f"Only SignRoundConfig is supported, but got {type(quant_config).__name__}. "
            f"RTN and AWQ algorithms have been removed."
        )

    # Select compressor class: DataDrivenCompressor for LLM, MLLM-mixed for multimodal
    compressor_cls = _get_compressor_class(model)

    # Forward MLLM-specific kwargs if the mixin is active
    mllm_kwargs = {}
    if issubclass(compressor_cls, type) and compressor_cls.__name__.startswith("Mllm"):
        for _k in ("processor", "image_processor", "template", "extra_data_dir", "quant_nontext_module"):
            if _k in kwargs:
                mllm_kwargs[_k] = kwargs.pop(_k)

    return compressor_cls(alg_configs, **local_args, **mllm_kwargs, **kwargs)


def _pop_config_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract old-API config kwargs and split them by config type."""
    common_keys = (
        "ignore_layers",
        "quant_lm_head",
        "scale_dtype",
        "super_bits",
        "super_group_size",
        "to_quant_block_names",
    )
    auto_round_only_keys = (
        "nblocks",
        "enable_alg_ext",
        "lr_scheduler",
        "not_use_best_mse",
        "dynamic_max_gap",
        "optimizer",
        "enable_adam",
        "momentum",
    )
    common_kwargs = {}
    auto_round_kwargs = {}
    for key in common_keys:
        if key in kwargs:
            common_kwargs[key] = kwargs.pop(key)
    for key in auto_round_only_keys:
        if key in kwargs:
            auto_round_kwargs[key] = kwargs.pop(key)
    return common_kwargs, auto_round_kwargs


def AutoRoundCompatible(
    model: Union[torch.nn.Module, str],
    tokenizer=None,
    platform: str = "hf",
    scheme: Union[str, dict, QuantizationScheme] = "W4A16",
    layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
    dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
    iters: int = 200,
    seqlen: int = 2048,
    nsamples: int = 128,
    batch_size: int = 8,
    gradient_accumulate_steps: int = 1,
    low_gpu_mem_usage: bool = False,
    device_map: Union[str, torch.device, int, dict] = 0,
    enable_torch_compile: bool = False,
    seed: int = 42,
    low_cpu_mem_usage: bool = True,
    algorithm: str = None,
    **kwargs,
) -> DataDrivenCompressor:
    """Create AutoRoundCompatible instance using new AutoRound architecture.

    This is a backward-compatible factory function that translates the old
    AutoRoundCompatible API to the new architecture.
    """
    device = kwargs.pop("device", None)
    if device is not None:
        logger.warning_once("`device` is deprecated, please use `device_map` instead")
        if device_map in (None, 0):
            device_map = device

    common_config_kwargs, auto_round_config_kwargs = _pop_config_kwargs(kwargs)

    # Extract quantization parameters from kwargs or use defaults
    bits = kwargs.pop("bits", None)
    group_size = kwargs.pop("group_size", None)
    sym = kwargs.pop("sym", None)
    data_type = kwargs.pop("data_type", None)
    act_bits = kwargs.pop("act_bits", None)
    act_group_size = kwargs.pop("act_group_size", None)
    act_sym = kwargs.pop("act_sym", None)
    act_data_type = kwargs.pop("act_data_type", None)
    act_dynamic = kwargs.pop("act_dynamic", None)
    enable_opt_rtn = kwargs.pop("enable_opt_rtn", None)
    lr = kwargs.pop("lr", None)
    minmax_lr = kwargs.pop("minmax_lr", None)
    enable_minmax_tuning = kwargs.pop("enable_minmax_tuning", True)
    enable_norm_bias_tuning = kwargs.pop("enable_norm_bias_tuning", False)
    enable_quanted_input = kwargs.pop("enable_quanted_input", True)

    # Always use SignRoundConfig
    config = SignRoundConfig(
        iters=iters,
        gradient_accumulate_steps=gradient_accumulate_steps,
        bits=bits,
        group_size=group_size,
        sym=sym,
        data_type=data_type,
        act_bits=act_bits,
        act_group_size=act_group_size,
        act_sym=act_sym,
        act_data_type=act_data_type,
        act_dynamic=act_dynamic,
        lr=lr,
        minmax_lr=minmax_lr,
        enable_minmax_tuning=enable_minmax_tuning,
        enable_norm_bias_tuning=enable_norm_bias_tuning,
        enable_quanted_input=enable_quanted_input,
        **common_config_kwargs,
        **auto_round_config_kwargs,
    )

    # Determine output format if specified
    format = kwargs.pop("format", None)

    # Extract rotation_config (old-API kwarg) and thread it into alg_configs.
    _rotation_config_raw = kwargs.pop("rotation_config", None)
    if _rotation_config_raw is not None:
        if isinstance(_rotation_config_raw, _BaseRotationConfig):
            _rc = _rotation_config_raw
        elif isinstance(_rotation_config_raw, dict):
            _rc = _normalize_any_rotation_config(_rotation_config_raw)
        elif isinstance(_rotation_config_raw, str):
            _rc = _normalize_any_rotation_config(_rotation_config_raw)
        else:
            _rc = _NewArchRotationConfig()
        if _rc is not None:
            config = [config, _rc]

    # Create AutoRound instance using new architecture
    compressor = auto_round_factory(
        alg_configs=config,
        model=model,
        tokenizer=tokenizer,
        platform=platform,
        format=format,
        scheme=scheme,
        dataset=dataset,
        iters=iters,
        gradient_accumulate_steps=gradient_accumulate_steps,
        low_gpu_mem_usage=low_gpu_mem_usage,
        device_map=device_map,
        enable_torch_compile=enable_torch_compile,
        seed=seed,
        low_cpu_mem_usage=low_cpu_mem_usage,
        layer_config=layer_config,
        nsamples=nsamples,
        seqlen=seqlen,
        batch_size=batch_size,
        # Pass remaining kwargs
        **kwargs,
    )

    return compressor
